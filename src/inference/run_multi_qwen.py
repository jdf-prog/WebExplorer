import argparse
import copy
import json
import math
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from qwen_runtime.qwen_agent import QwenFunctionCallAgent


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
    with open(temp_path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
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


def load_items(dataset_name: str):
    data_filepath = os.path.join(CURRENT_DIR, "eval_data", f"{dataset_name}.jsonl")
    with open(data_filepath, "r", encoding="utf-8") as file_obj:
        return [json.loads(line) for line in file_obj if line.strip()]


def extract_question(item: dict) -> str:
    question = item.get("question", "").strip()
    if question:
        return question
    raw_msg = item.get("messages", [{}, {}])[1].get("content", "")
    return raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.getenv("QWEN_MODEL_NAME", "qwen3.5-plus"))
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--dataset", type=str, default="browsecomp_validation_100")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--roll_out_count", type=int, default=1)
    parser.add_argument("--total_splits", type=int, default=1)
    parser.add_argument("--worker_split", type=int, default=1)
    parser.add_argument("--auto_judge", action="store_true")
    parser.add_argument(
        "--judge_engine",
        type=str,
        default="openai",
        choices=["deepseekchat", "geminiflash", "openai"],
    )
    args = parser.parse_args()

    model_name = os.path.basename(args.model.rstrip("/"))
    output_base = args.output or os.getenv(
        "QWEN_OUTPUT_PATH",
        os.path.join(CURRENT_DIR, "..", "..", "..", "outputs", "qwen_function_call"),
    )
    model_dir = os.path.join(output_base, model_name)
    dataset_dir = os.path.join(model_dir, args.dataset)
    running_dir = os.path.join(dataset_dir, "running")
    finished_dir = os.path.join(dataset_dir, "finished")
    os.makedirs(running_dir, exist_ok=True)
    os.makedirs(finished_dir, exist_ok=True)

    items = load_items(args.dataset)
    total_items = len(items)
    items_per_split = math.ceil(total_items / args.total_splits)
    start_idx = (args.worker_split - 1) * items_per_split
    end_idx = min(args.worker_split * items_per_split, total_items)
    items = items[start_idx:end_idx]

    output_tag = (
        f"qwenctx-fold"
        f"_toolmax-{sanitize_tag_value(os.getenv('QWEN_TOOL_CONTEXT_MAX', '32000'))}"
        f"_tooltarget-{sanitize_tag_value(os.getenv('QWEN_TOOL_CONTEXT_TARGET', '5000'))}"
        f"_turns-{sanitize_tag_value(os.getenv('QWEN_MAX_LLM_CALL_PER_RUN', '200'))}"
    )

    if args.total_splits > 1:
        output_files = {
            idx: os.path.join(
                dataset_dir,
                f"iter{idx}_{output_tag}_split{args.worker_split}of{args.total_splits}.jsonl",
            )
            for idx in range(1, args.roll_out_count + 1)
        }
    else:
        output_files = {
            idx: os.path.join(dataset_dir, f"iter{idx}_{output_tag}.jsonl")
            for idx in range(1, args.roll_out_count + 1)
        }

    processed_queries_per_rollout = {}
    for rollout_idx in range(1, args.roll_out_count + 1):
        processed = set()
        output_file = output_files[rollout_idx]
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as file_obj:
                for line in file_obj:
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "question" in data and "error" not in data:
                        processed.add(data["question"].strip())
        processed_queries_per_rollout[rollout_idx] = processed

    tasks_to_run_all = []
    for rollout_idx in range(1, args.roll_out_count + 1):
        processed_queries = processed_queries_per_rollout[rollout_idx]
        for local_idx, item in enumerate(items):
            question = extract_question(item)
            if not question or question in processed_queries:
                continue
            item = item.copy()
            item["question"] = question
            global_idx = start_idx + local_idx
            task_id = make_task_id(item, global_idx, rollout_idx, output_tag)
            task_info = {
                "item": item,
                "rollout_idx": rollout_idx,
                "task_id": task_id,
                "task_index": global_idx,
                "output_tag": output_tag,
                "running_path": os.path.join(running_dir, f"{task_id}.json"),
                "finished_path": os.path.join(finished_dir, f"{task_id}.json"),
            }
            task_info["progress_callback"] = build_progress_writer(task_info)
            tasks_to_run_all.append(task_info)

    if not tasks_to_run_all:
        print("All rollouts have been completed and no execution is required.")
        raise SystemExit(0)

    llm_cfg = {
        "model": args.model,
        "generate_cfg": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": int(os.getenv("QWEN_TOP_K", "20")),
            "min_p": float(os.getenv("QWEN_MIN_P", "0.0")),
            "presence_penalty": float(os.getenv("QWEN_PRESENCE_PENALTY", "1.5")),
            "repetition_penalty": float(os.getenv("QWEN_REPETITION_PENALTY", "1.0")),
        },
        "api_key": os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
        "base_url": os.getenv(
            "QWEN_API_BASE",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "tokenizer_path": os.getenv("QWEN_TOKENIZER_PATH", ""),
    }

    agent = QwenFunctionCallAgent(llm=llm_cfg)
    write_locks = {idx: threading.Lock() for idx in range(1, args.roll_out_count + 1)}

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(
                agent._run,
                task,
                args.model,
                auto_judge=args.auto_judge,
                judge_engine=args.judge_engine,
                progress_callback=task["progress_callback"],
                task_metadata={
                    "task_id": task["task_id"],
                    "rollout_idx": task["rollout_idx"],
                    "task_index": task["task_index"],
                    "output_tag": task["output_tag"],
                },
            ): task
            for task in tasks_to_run_all
        }

        for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run_all), desc="Processing Qwen Rollouts"):
            task_info = future_to_task[future]
            rollout_idx = task_info["rollout_idx"]
            output_file = output_files[rollout_idx]
            try:
                result = future.result()
                result["task_id"] = task_info["task_id"]
                result["task_index"] = task_info["task_index"]
                with write_locks[rollout_idx]:
                    with open(output_file, "a", encoding="utf-8") as file_obj:
                        file_obj.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as exc:
                question = task_info["item"].get("question", "")
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
                task_info["progress_callback"](error_result, final=True)
                with write_locks[rollout_idx]:
                    with open(output_file, "a", encoding="utf-8") as file_obj:
                        file_obj.write(json.dumps(error_result, ensure_ascii=False) + "\n")

    print("\nAll Qwen tasks completed!")
