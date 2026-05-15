import argparse
import copy
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import threading
from datetime import datetime, timezone
from vllm_react_agent import MultiTurnReactAgent, VllmServerError
from qwen_vllm_agent import QwenVllmReactAgent
from deepseek_vllm_agent import DeepSeekVllmReactAgent
import time
import math
import re
import traceback
from typing import Dict, Optional


def sanitize_tag_value(value: str) -> str:
    return str(value).strip().replace(".", "p").replace("/", "-").replace(" ", "_")


def sanitize_file_stem(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._")
    return sanitized or "task"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_context_management_strategy(strategy: str) -> str:
    normalized = (strategy or "none").strip().lower().replace("-", "_")
    if normalized in {"discard", "discard_all"}:
        return "discard_all"
    if normalized in {
        "fold_then_discard",
        "fold_tool_then_discard",
        "fold_then_reset",
    }:
        return "fold_then_discard"
    if normalized in {
        "fold_tool",
        "fold_tools",
        "fold_tool_call",
        "fold_tool_calls",
        "fold_tool_message",
        "fold_tool_messages",
    }:
        return "fold_tool"
    if normalized == "summary":
        return "summary"
    return "none"


def is_qwen_model(model_name: str) -> bool:
    return "qwen" in os.path.basename(model_name.rstrip("/")).lower()


def is_deepseek_model(model_name: str) -> bool:
    return "deepseek" in os.path.basename(model_name.rstrip("/")).lower()


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def remove_file_if_exists(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def cleanup_task_artifacts(task_info: dict) -> None:
    remove_file_if_exists(task_info["running_path"])
    for key in ("finished_path", "to_eval_path", "errored_path", "interrupted_path"):
        path = task_info.get(key)
        if path:
            remove_file_if_exists(path)


def has_auto_judge_score(payload: dict) -> bool:
    auto_judge = payload.get("auto_judge")
    if not isinstance(auto_judge, dict):
        return False
    return isinstance(auto_judge.get("score"), (int, float))


def is_error_result(payload: dict) -> bool:
    if payload.get("error"):
        return True
    if payload.get("prediction") == "[Failed]":
        return True
    return False


def is_interrupted_result(payload: dict) -> bool:
    if payload.get("interrupted") is True:
        return True
    return payload.get("termination") in {"timeout", "interrupted"}


def terminal_path_for_result(task_info: dict, payload: dict) -> str:
    if is_interrupted_result(payload):
        return task_info["interrupted_path"]
    if is_error_result(payload):
        return task_info["errored_path"]
    if has_auto_judge_score(payload):
        return task_info["finished_path"]
    return task_info["to_eval_path"]


def hard_exit_due_to_vllm_error(message: str, exit_code: int = 2) -> None:
    print(message, flush=True)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        # Hard-exit so a dead vLLM server does not leave the evaluator writing junk records.
        os._exit(exit_code)


def make_task_id(item: dict, global_idx: int, rollout_idx: int, output_tag: str) -> str:
    raw_task_id = item.get("id") or f"task_{global_idx:04d}"
    base_task_id = sanitize_file_stem(raw_task_id)
    return f"{base_task_id}_{sanitize_file_stem(output_tag)}_iter{rollout_idx}"


def build_progress_writer(task_info: dict, abort_event: Optional[threading.Event] = None):
    resume_state = task_info.get("resume_state") or {}
    lifecycle = {"started_at": resume_state.get("started_at")}

    def _write(snapshot: dict, final: bool = False):
        if abort_event is not None and abort_event.is_set():
            return
        payload = copy.deepcopy(snapshot)
        now = utc_now_iso()

        if lifecycle["started_at"] is None:
            lifecycle["started_at"] = payload.get("started_at") or now

        payload["task_id"] = task_info["task_id"]
        payload["rollout_idx"] = task_info["rollout_idx"]
        payload["task_index"] = task_info["task_index"]
        payload["output_tag"] = task_info["output_tag"]
        payload["started_at"] = lifecycle["started_at"]
        payload["updated_at"] = now

        if final:
            payload["status"] = "interrupted" if is_interrupted_result(payload) else "finished"
            payload["finished_at"] = now
            terminal_path = terminal_path_for_result(task_info, payload)
            for key in ("finished_path", "to_eval_path", "errored_path", "interrupted_path"):
                path = task_info.get(key)
                if path and path != terminal_path:
                    remove_file_if_exists(path)
            write_json_atomic(terminal_path, payload)
            try:
                os.remove(task_info["running_path"])
            except FileNotFoundError:
                pass
            return

        payload.setdefault("status", "running")
        write_json_atomic(task_info["running_path"], payload)

    return _write


def load_running_resume_state(running_path: str, question: str) -> Optional[Dict]:
    enabled = os.getenv("WEBEXPLORER_RESUME_RUNNING", "1").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return None
    if not os.path.exists(running_path):
        return None
    try:
        with open(running_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: cannot resume from running snapshot {running_path}: {e}")
        return None

    if state.get("status") != "running":
        return None
    if not isinstance(state.get("messages"), list) or not state["messages"]:
        return None
    if (state.get("question") or "").strip() != question.strip():
        print(f"Warning: ignoring stale running snapshot with mismatched question: {running_path}")
        return None
    return state


def collect_processed_queries(
    output_file: str,
    finished_dir: str,
    rollout_idx: int,
    to_eval_dir: Optional[str] = None,
) -> set:
    processed_queries = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "question" in data and "error" not in data:
                            processed_queries.add(data["question"].strip())
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid line in output file: {line.strip()}")
        except FileNotFoundError:
            pass

    terminal_dirs = [finished_dir]
    if to_eval_dir:
        terminal_dirs.append(to_eval_dir)
    for terminal_dir in terminal_dirs:
        if not os.path.isdir(terminal_dir):
            continue
        for filename in os.listdir(terminal_dir):
            if not filename.endswith(f"_iter{rollout_idx}.json"):
                continue
            path = os.path.join(terminal_dir, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Skipping invalid finished file {path}: {e}")
                continue
            if "question" in data and "error" not in data:
                processed_queries.add(data["question"].strip())

    return processed_queries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dataset", type=str, default="bc_1_per_6")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--roll_out_count", type=int, default=3)
    parser.add_argument("--total_splits", type=int, default=1)
    parser.add_argument("--worker_split", type=int, default=1)
    parser.add_argument("--auto_judge", action="store_true", help="Enable automatic judging of answers")
    parser.add_argument("--judge_engine", type=str, default="deepseekchat", choices=["deepseekchat", "geminiflash", "openai"], help="LLM engine for auto judging")
    args = parser.parse_args()

    model = args.model
    output_base = args.output
    roll_out_count = args.roll_out_count
    total_splits = args.total_splits
    worker_split = args.worker_split

    # Validate worker_split
    if worker_split < 1 or worker_split > total_splits:
        print(f"Error: worker_split ({worker_split}) must be between 1 and total_splits ({total_splits})")
        exit(1)

    model_name = os.path.basename(model.rstrip('/'))

    context_strategy = normalize_context_management_strategy(
        os.getenv("CONTEXT_MANAGEMENT_STRATEGY", "none")
    )
    context_reset_threshold = os.getenv("CONTEXT_RESET_THRESHOLD", "0.3")
    context_summary_trigger_tokens = os.getenv(
        "CONTEXT_SUMMARY_TRIGGER_TOKENS", "32768"
    )
    context_total_token_limit = os.getenv("CONTEXT_TOTAL_TOKEN_LIMIT", "1000000")
    tool_context_max = os.getenv("TOOL_CONTEXT_MAX", os.getenv("QWEN_TOOL_CONTEXT_MAX", "32000"))
    tool_context_target = os.getenv("TOOL_CONTEXT_TARGET", os.getenv("QWEN_TOOL_CONTEXT_TARGET", "5000"))
    discard_prompt_threshold_ratio = os.getenv("DISCARD_PROMPT_THRESHOLD_RATIO", "0.85")
    discard_history_tool_tokens = os.getenv("DISCARD_HISTORY_TOOL_TOKENS", "0")
    discard_history_min_rounds = os.getenv("DISCARD_HISTORY_MIN_ROUNDS", "0")
    discard_history_max_rounds = os.getenv("DISCARD_HISTORY_MAX_ROUNDS", "0")
    if is_deepseek_model(model):
        default_max_llm_call_per_run = "500"
    elif is_qwen_model(model):
        default_max_llm_call_per_run = "200"
    else:
        default_max_llm_call_per_run = "100"
    max_llm_call_per_run = os.getenv(
        "MAX_LLM_CALL_PER_RUN", default_max_llm_call_per_run
    )
    output_tag = f"ctx-{sanitize_tag_value(context_strategy)}"
    if context_strategy == "summary":
        output_tag += (
            f"_sumctx-{sanitize_tag_value(context_summary_trigger_tokens)}"
            f"_tot-{sanitize_tag_value(context_total_token_limit)}"
        )
    elif context_strategy in {"fold_tool", "fold_then_discard"}:
        output_tag += (
            f"_toolmax-{sanitize_tag_value(tool_context_max)}"
            f"_tooltarget-{sanitize_tag_value(tool_context_target)}"
        )
        if context_strategy == "fold_then_discard":
            output_tag += (
                f"_discardthr-{sanitize_tag_value(discard_prompt_threshold_ratio)}"
                f"_histtool-{sanitize_tag_value(discard_history_tool_tokens)}"
                f"_minr-{sanitize_tag_value(discard_history_min_rounds)}"
                f"_maxr-{sanitize_tag_value(discard_history_max_rounds)}"
            )
    else:
        output_tag += f"_thr-{sanitize_tag_value(context_reset_threshold)}"
    output_tag += f"_turns-{sanitize_tag_value(max_llm_call_per_run)}"
    if is_deepseek_model(model):
        deepseek_thinking_mode = os.getenv("DEEPSEEK_THINKING_MODE", "think")
        output_tag += f"_think-{sanitize_tag_value(deepseek_thinking_mode)}"

    model_dir = os.path.join(output_base, f"{model_name}")
    dataset_base_dir = os.path.join(model_dir, args.dataset)
    dataset_dir = os.path.join(dataset_base_dir, output_tag)
    running_dir = os.path.join(dataset_dir, "running")
    finished_dir = os.path.join(dataset_dir, "finished")
    to_eval_dir = os.path.join(dataset_dir, "to_eval")
    errored_dir = os.path.join(dataset_dir, "errored")
    interrupted_dir = os.path.join(dataset_dir, "interrupted")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(running_dir, exist_ok=True)
    os.makedirs(finished_dir, exist_ok=True)
    os.makedirs(to_eval_dir, exist_ok=True)
    os.makedirs(errored_dir, exist_ok=True)
    os.makedirs(interrupted_dir, exist_ok=True)

    print(f"Model name: {model_name}")
    print(f"Data set name: {args.dataset}")
    print(f"Output directory: {dataset_dir}")
    print(f"Number of rollouts: {roll_out_count}")
    print(f"Data splitting: {worker_split}/{total_splits}")
    print(f"Auto judge enabled: {args.auto_judge}")
    if args.auto_judge:
        print(f"Judge engine: {args.judge_engine}")
    print(f"Output tag: {output_tag}")

    data_filepath = f"eval_data/{args.dataset}.jsonl"
    try:
        if data_filepath.endswith(".json"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                raise ValueError("Input JSON must be a list of objects.")
            if items and not isinstance(items[0], dict):
                raise ValueError("Input JSON list items must be objects.")
        elif data_filepath.endswith(".jsonl"):
            with open(data_filepath, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f]
        else:
            raise ValueError("Unsupported file extension. Please use .json or .jsonl files.")
        items = items
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_filepath}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading or parsing input file {data_filepath}: {e}")
        exit(1)

    # Apply data splitting
    total_items = len(items)
    items_per_split = math.ceil(total_items / total_splits)
    start_idx = (worker_split - 1) * items_per_split
    end_idx = min(worker_split * items_per_split, total_items)
    
    # Split the dataset
    items = items[start_idx:end_idx]
    
    print(f"Total items in dataset: {total_items}")
    print(f"Processing items {start_idx} to {end_idx-1} ({len(items)} items)")

    if total_splits > 1:
        # Add split suffix to output files when using splits
        output_files = {
            i: os.path.join(
                dataset_dir,
                f"iter{i}_split{worker_split}of{total_splits}.jsonl"
            )
            for i in range(1, roll_out_count + 1)
        }
    else:
        output_files = {
            i: os.path.join(dataset_dir, f"iter{i}.jsonl")
            for i in range(1, roll_out_count + 1)
        }
    
    processed_queries_per_rollout = {}
    abort_event = threading.Event()

    for rollout_idx in range(1, roll_out_count + 1):
        output_file = output_files[rollout_idx]
        processed_queries = collect_processed_queries(
            output_file=output_file,
            finished_dir=finished_dir,
            rollout_idx=rollout_idx,
            to_eval_dir=to_eval_dir,
        )
        processed_queries_per_rollout[rollout_idx] = processed_queries

    tasks_to_run_all = []
    per_rollout_task_counts = {i: 0 for i in range(1, roll_out_count + 1)}
    planning_ports_env = os.getenv("WEBEXPLORER_VLLM_PORTS", "6001,6002,6003,6004,6005,6006,6007,6008")
    planning_ports = [int(port.strip()) for port in planning_ports_env.split(",") if port.strip()]
    if not planning_ports:
        raise ValueError("WEBEXPLORER_VLLM_PORTS must contain at least one port")
    # Round-robin state
    planning_rr_idx = 0
    summary_rr_idx = 0
    # Sticky assignment per question
    question_to_ports = {}
    for rollout_idx in range(1, roll_out_count + 1):
        processed_queries = processed_queries_per_rollout[rollout_idx]
        for local_idx, item in enumerate(items):
            question = item.get("question", "").strip()
            if question == "":
                try:
                    user_msg = item["messages"][1]["content"]
                    question = user_msg.split("User:")[1].strip() if "User:" in user_msg else user_msg
                    item["question"] = question
                except Exception as e:
                    print(f"Extract question from user message failed: {e}")
            if not question:
                print(f"Warning: Skipping item with empty question: {item}")
                continue

            if question not in processed_queries:
                # Ensure sticky and balanced port assignment per unique question
                if question not in question_to_ports:
                    planning_port = planning_ports[planning_rr_idx % len(planning_ports)]
                    question_to_ports[question] = planning_port
                    planning_rr_idx += 1
                planning_port = question_to_ports[question]
                global_idx = start_idx + local_idx
                task_id = make_task_id(
                    item=item,
                    global_idx=global_idx,
                    rollout_idx=rollout_idx,
                    output_tag=output_tag,
                )
                task_info = {
                    "item": item.copy(),
                    "rollout_idx": rollout_idx,
                    "planning_port": planning_port,
                    "task_id": task_id,
                    "task_index": global_idx,
                    "output_tag": output_tag,
                    "running_path": os.path.join(running_dir, f"{task_id}.json"),
                    "finished_path": os.path.join(finished_dir, f"{task_id}.json"),
                    "to_eval_path": os.path.join(to_eval_dir, f"{task_id}.json"),
                    "errored_path": os.path.join(errored_dir, f"{task_id}.json"),
                    "interrupted_path": os.path.join(interrupted_dir, f"{task_id}.json"),
                }
                task_info["resume_state"] = load_running_resume_state(
                    task_info["running_path"],
                    question,
                )
                task_info["progress_callback"] = build_progress_writer(
                    task_info, abort_event=abort_event
                )
                tasks_to_run_all.append({
                    **task_info,
                })
                per_rollout_task_counts[rollout_idx] += 1

    print(f"Total questions in current split: {len(items)}")
    for rollout_idx in range(1, roll_out_count + 1):
        print(f"Rollout {rollout_idx}: already successfully processed: {len(processed_queries_per_rollout[rollout_idx])}, to run: {per_rollout_task_counts[rollout_idx]}")

    if not tasks_to_run_all:
        print("All rollouts have been completed and no execution is required.")
    else:
        if is_deepseek_model(model):
            default_max_input_tokens = "524288"
        elif is_qwen_model(model):
            default_max_input_tokens = "262144"
        else:
            default_max_input_tokens = "196608"
        llm_cfg = {
            'model': model,
            'generate_cfg': {
                'max_input_tokens': int(os.getenv("MAX_INPUT_TOKENS", default_max_input_tokens)),
                'max_retries': 10,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'top_k': int(os.getenv("TOP_K", "40")),
                'min_p': float(os.getenv("MIN_P", "0.0")),
                'presence_penalty': float(os.getenv("PRESENCE_PENALTY", "0.0")),
                'repetition_penalty': float(os.getenv("REPETITION_PENALTY", "1.0")),
            },
            'model_type': 'qwen_dashscope'
        }

        if is_deepseek_model(model):
            test_agent = DeepSeekVllmReactAgent(
                llm=llm_cfg,
                function_list=["code_interpreter", "web_search", "web_extractor"],
            )
        elif is_qwen_model(model):
            test_agent = QwenVllmReactAgent(
                llm=llm_cfg,
                function_list=["code_interpreter", "web_search", "web_extractor"],
            )
        else:
            test_agent = MultiTurnReactAgent(
                llm=llm_cfg,
                function_list=["search", "browse"],
            )
        tool_schemas = copy.deepcopy(getattr(test_agent, "tool_schemas", []))

        write_locks = {i: threading.Lock() for i in range(1, roll_out_count + 1)}
        persisted_task_ids = set()

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    test_agent._run,
                    task,
                    model,
                    auto_judge=args.auto_judge,
                    judge_engine=args.judge_engine,
                    progress_callback=task["progress_callback"],
                    resume_state=task.get("resume_state"),
                    task_metadata={
                        "task_id": task["task_id"],
                        "rollout_idx": task["rollout_idx"],
                        "task_index": task["task_index"],
                        "output_tag": task["output_tag"],
                        "dataset": args.dataset,
                    },
                ): task for task in tasks_to_run_all
            }

            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run_all), desc="Processing All Rollouts"):
                task_info = future_to_task[future]
                rollout_idx = task_info["rollout_idx"]
                output_file = output_files[rollout_idx]
                try:
                    result = future.result()
                    result["task_id"] = task_info["task_id"]
                    result["task_index"] = task_info["task_index"]
                    result.setdefault("tools", tool_schemas)
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    persisted_task_ids.add(task_info["task_id"])
                except VllmServerError as exc:
                    abort_event.set()
                    for pending_task in future_to_task.values():
                        if pending_task["task_id"] in persisted_task_ids:
                            continue
                        cleanup_task_artifacts(pending_task)
                    question = task_info["item"].get("question", "")
                    hard_exit_due_to_vllm_error(
                        "Fatal: detected vLLM server failure while processing "
                        f"task_id={task_info['task_id']} "
                        f"(question={question!r}, rollout={rollout_idx}): {exc}"
                    )
                except concurrent.futures.TimeoutError:
                    question = task_info["item"].get("question", "")
                    print(f'Timeout (>1800s): "{question}" (Rollout {rollout_idx})')
                    future.cancel()
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "tools": tool_schemas,
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "task_id": task_info["task_id"],
                        "task_index": task_info["task_index"],
                        "error": "Timeout (>1800s)",
                        "messages": [],
                        "log": [],
                        "prediction": "[Failed]",
                        "termination": "timeout",
                        "error_type": "TimeoutError",
                        "interrupted": True,
                        "retryable": True,
                    }
                    task_info["progress_callback"](error_result, final=True)
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                    persisted_task_ids.add(task_info["task_id"])
                except Exception as exc:
                    question = task_info["item"].get("question", "")
                    print(f'Task for question "{question}" (Rollout {rollout_idx}) generated an exception: {exc}')
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "tools": tool_schemas,
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "task_id": task_info["task_id"],
                        "task_index": task_info["task_index"],
                        "error": f"Future resolution failed: {exc}",
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                        "messages": [],
                        "log": [],
                        "prediction": "[Failed]",
                        "termination": "task_exception",
                        "retryable": False,
                    }
                    print("===============================")
                    print(error_result)
                    print("===============================")
                    task_info["progress_callback"](error_result, final=True)
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                    persisted_task_ids.add(task_info["task_id"])

        print("\nAll tasks completed!")

    print(f"\nAll {roll_out_count} rollouts completed!")
