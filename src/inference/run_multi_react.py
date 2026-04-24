import argparse
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from tqdm import tqdm
import threading
from datetime import datetime, timezone
from vllm_react_agent import MultiTurnReactAgent
import time
import math
import re


def sanitize_tag_value(value: str) -> str:
    return str(value).strip().replace(".", "p").replace("/", "-").replace(" ", "_")


def sanitize_file_stem(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    sanitized = sanitized.strip("._")
    return sanitized or "task"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def make_task_id(item: dict, global_idx: int, rollout_idx: int, output_tag: str) -> str:
    raw_task_id = item.get("id") or f"task_{global_idx:04d}"
    base_task_id = sanitize_file_stem(raw_task_id)
    return f"{base_task_id}_{sanitize_file_stem(output_tag)}_iter{rollout_idx}"


def build_progress_writer(task_info: dict):
    lifecycle = {"started_at": None}

    def _write(snapshot: dict, final: bool = False):
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
            payload["status"] = "finished"
            payload["finished_at"] = now
            write_json_atomic(task_info["finished_path"], payload)
            try:
                os.remove(task_info["running_path"])
            except FileNotFoundError:
                pass
            return

        payload.setdefault("status", "running")
        write_json_atomic(task_info["running_path"], payload)

    return _write


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dataset", type=str, default="gaia")
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

    context_strategy = os.getenv("CONTEXT_MANAGEMENT_STRATEGY", "none")
    context_reset_threshold = os.getenv("CONTEXT_RESET_THRESHOLD", "0.3")
    context_summary_trigger_tokens = os.getenv(
        "CONTEXT_SUMMARY_TRIGGER_TOKENS", "32768"
    )
    context_total_token_limit = os.getenv("CONTEXT_TOTAL_TOKEN_LIMIT", "1000000")
    max_llm_call_per_run = os.getenv("MAX_LLM_CALL_PER_RUN", "100")
    output_tag = f"ctx-{sanitize_tag_value(context_strategy)}"
    if context_strategy == "summary":
        output_tag += (
            f"_sumctx-{sanitize_tag_value(context_summary_trigger_tokens)}"
            f"_tot-{sanitize_tag_value(context_total_token_limit)}"
        )
    else:
        output_tag += f"_thr-{sanitize_tag_value(context_reset_threshold)}"
    output_tag += f"_turns-{sanitize_tag_value(max_llm_call_per_run)}"

    model_dir = os.path.join(output_base, f"{model_name}")
    dataset_base_dir = os.path.join(model_dir, args.dataset)
    dataset_dir = os.path.join(dataset_base_dir, output_tag)
    running_dir = os.path.join(dataset_dir, "running")
    finished_dir = os.path.join(dataset_dir, "finished")

    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(running_dir, exist_ok=True)
    os.makedirs(finished_dir, exist_ok=True)

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

    for rollout_idx in range(1, roll_out_count + 1):
        output_file = output_files[rollout_idx]
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
                }
                task_info["progress_callback"] = build_progress_writer(task_info)
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
        llm_cfg = {
            'model': model,
            'generate_cfg': {
                'max_input_tokens': int(os.getenv("MAX_INPUT_TOKENS", "196608")),
                'max_retries': 10,
                'temperature': args.temperature,
                'top_p': args.top_p,
                'top_k': int(os.getenv("TOP_K", "40")),
            },
            'model_type': 'qwen_dashscope'
        }

        test_agent = MultiTurnReactAgent(
            llm=llm_cfg,
            function_list=["search", "browse"]
        )

        write_locks = {i: threading.Lock() for i in range(1, roll_out_count + 1)}

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    test_agent._run,
                    task,
                    model,
                    auto_judge=args.auto_judge,
                    judge_engine=args.judge_engine,
                    progress_callback=task["progress_callback"],
                    task_metadata={
                        "task_id": task["task_id"],
                        "rollout_idx": task["rollout_idx"],
                        "task_index": task["task_index"],
                        "output_tag": task["output_tag"],
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
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                except concurrent.futures.TimeoutError:
                    question = task_info["item"].get("question", "")
                    print(f'Timeout (>1800s): "{question}" (Rollout {rollout_idx})')
                    future.cancel()
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "task_id": task_info["task_id"],
                        "task_index": task_info["task_index"],
                        "error": "Timeout (>1800s)",
                        "messages": [],
                        "log": [],
                        "prediction": "[Failed]",
                        "termination": "timeout",
                    }
                    task_info["progress_callback"](error_result, final=True)
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")
                except Exception as exc:
                    question = task_info["item"].get("question", "")
                    print(f'Task for question "{question}" (Rollout {rollout_idx}) generated an exception: {exc}')
                    error_result = {
                        "question": question,
                        "answer": task_info["item"].get("answer", ""),
                        "rollout_idx": rollout_idx,
                        "rollout_id": rollout_idx,
                        "task_id": task_info["task_id"],
                        "task_index": task_info["task_index"],
                        "error": f"Future resolution failed: {exc}",
                        "messages": [],
                        "log": [],
                        "prediction": "[Failed]",
                        "termination": "future_resolution_failed",
                    }
                    print("===============================")
                    print(error_result)
                    print("===============================")
                    task_info["progress_callback"](error_result, final=True)
                    with write_locks[rollout_idx]:
                        with open(output_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(error_result, ensure_ascii=False) + "\n")

        print("\nAll tasks completed!")

    print(f"\nAll {roll_out_count} rollouts completed!")
