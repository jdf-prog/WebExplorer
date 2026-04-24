import copy
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
import tiktoken
from transformers import AutoTokenizer

try:
    import json5
except ImportError:  # pragma: no cover - fallback for lean runtime envs
    json5 = None

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool

from auto_judge import compute_score_genrm
from tool_webexplorer_browse import *
from tool_webexplorer_search import *

MAX_LLM_CALL_PER_RUN = int(os.getenv("MAX_LLM_CALL_PER_RUN", 100))

TRUNCATED_MESSAGE = """
--- Maximum Length Limit Reached ---
You have reached the maximum length limit.
The response is truncated."""
FINAL_MESSAGE = """
--- Final Step Reached ---
Now you reach the final step.
You are forbidden to call any tools.
You must offer your final answer now."""

SYSTEM_PROMPT = "You are a helpful assistant."
MINIMAX_21_SYSTEM_PROMPT = "You are a helpful assistant. Your name is MiniMax-M2.1 and is built by MiniMax."
MINIMAX_25_SYSTEM_PROMPT = "You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax."

TOOL_CLASS = [
    WebExplorerBrowse(),
    WebExplorerSearch(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MultiTurnReactAgent(FnCallAgent):
    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
        llm: Optional[Union[Dict, BaseChatModel]] = None,
        **kwargs,
    ):
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = self._resolve_model_path(llm["model"])
        self.context_management_strategy = os.getenv(
            "CONTEXT_MANAGEMENT_STRATEGY", "none"
        ).strip().lower()
        self.context_reset_threshold = float(os.getenv("CONTEXT_RESET_THRESHOLD", "0.3"))
        self.context_summary_trigger_tokens = int(
            os.getenv("CONTEXT_SUMMARY_TRIGGER_TOKENS", "32768")
        )
        self.context_total_token_limit = int(
            os.getenv("CONTEXT_TOTAL_TOKEN_LIMIT", "1000000")
        )
        self.tokenizer = None
        self._tokenizer_initialized = False
        self._get_tokenizer()
        self.tool_schemas = self._build_tool_schemas()

    def _resolve_model_path(self, model_name_or_path: str) -> str:
        candidate = Path(model_name_or_path)
        if candidate.exists():
            return str(candidate)

        repo_root = Path(__file__).resolve().parents[3]
        local_candidate = repo_root / "models" / model_name_or_path
        if local_candidate.exists():
            return str(local_candidate)

        return model_name_or_path

    def _build_tool_schemas(self) -> List[Dict]:
        tool_schemas: List[Dict] = []
        for tool in TOOL_CLASS:
            tool_schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        return tool_schemas

    def _get_tokenizer(self):
        if self._tokenizer_initialized:
            return self.tokenizer

        self._tokenizer_initialized = True

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path)
            if not getattr(tokenizer, "chat_template", None):
                chat_template_path = Path(self.llm_local_path) / "chat_template.jinja"
                if chat_template_path.exists():
                    tokenizer.chat_template = chat_template_path.read_text(
                        encoding="utf-8"
                    )
            self.tokenizer = tokenizer
        except Exception:
            self.tokenizer = None

        return self.tokenizer

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content

    def _normalize_tool_call(self, tool_call) -> Dict:
        if hasattr(tool_call, "model_dump"):
            tool_call = tool_call.model_dump(exclude_none=True)
        return copy.deepcopy(tool_call)

    def _normalize_assistant_message(self, message) -> Dict:
        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
        }

        reasoning_content = getattr(message, "reasoning_content", None)
        if reasoning_content is not None:
            assistant_message["reasoning_content"] = reasoning_content

        if getattr(message, "tool_calls", None):
            assistant_message["tool_calls"] = []
            for tool_call in message.tool_calls:
                normalized_tool_call = self._normalize_tool_call(tool_call)
                assistant_message["tool_calls"].append(normalized_tool_call)

        return assistant_message

    def _strip_internal_message_fields(self, messages: List[Dict]) -> List[Dict]:
        return [
            {key: value for key, value in message.items() if not key.startswith("_")}
            for message in copy.deepcopy(messages)
        ]

    def _normalize_usage(self, usage) -> Optional[Dict]:
        if usage is None:
            return None

        if isinstance(usage, dict):
            usage_payload = copy.deepcopy(usage)
        elif hasattr(usage, "model_dump"):
            usage_payload = usage.model_dump(exclude_none=True)
        else:
            usage_payload = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, key, None)
                if value is not None:
                    usage_payload[key] = value

        return usage_payload or None

    def _token_count_from_usage(self, usage: Optional[Dict]):
        if not usage:
            return None, None

        for key in ("prompt_tokens", "total_tokens"):
            value = usage.get(key)
            if value is None:
                continue
            try:
                return int(value), f"server_usage.{key}"
            except (TypeError, ValueError):
                continue

        return None, None

    def _get_context_token_count(self, messages, usage: Optional[Dict] = None):
        token_count, token_count_source = self._token_count_from_usage(usage)
        if token_count is not None:
            return token_count, token_count_source, usage

        token_count = self.count_tokens(messages)
        return token_count, "local_count_tokens", usage

    def _count_text_tokens(self, text: str, model: str = "gpt-4o") -> int:
        if not text:
            return 0

        tokenizer = self._get_tokenizer()
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                pass

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))

    def _estimate_completion_tokens(self, assistant_message: Dict) -> int:
        total = 0
        total += self._count_text_tokens(assistant_message.get("content") or "")
        total += self._count_text_tokens(
            assistant_message.get("reasoning_content") or ""
        )

        tool_calls = assistant_message.get("tool_calls") or []
        if tool_calls:
            total += self._count_text_tokens(
                json.dumps(tool_calls, ensure_ascii=False)
            )

        return total

    def _get_call_usage(
        self, request_messages: List[Dict], assistant_message: Dict
    ) -> Dict:
        usage = self._normalize_usage(assistant_message.get("_usage"))
        if usage:
            return usage

        prompt_tokens = self.count_tokens(request_messages)
        completion_tokens = self._estimate_completion_tokens(assistant_message)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "estimated": True,
        }

    def _accumulate_usage(self, cumulative_usage: Dict, usage: Optional[Dict]) -> None:
        if not usage:
            return

        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if value is None:
                continue
            try:
                cumulative_usage[key] += int(value)
            except (TypeError, ValueError):
                continue

        if usage.get("estimated"):
            cumulative_usage["estimated_calls"] += 1

    def _make_context_management_stats(
        self, context_events: List[Dict], cumulative_usage: Dict
    ) -> Dict:
        discard_all_count = sum(
            1 for event in context_events if event.get("strategy") == "discard_all"
        )
        summary_count = sum(
            1 for event in context_events if event.get("strategy") == "summary"
        )
        return {
            "context_management_count": len(context_events),
            "discard_all_count": discard_all_count,
            "summary_count": summary_count,
            "context_reset_events": context_events,
            "cumulative_token_usage": copy.deepcopy(cumulative_usage),
            "context_total_token_limit": self.context_total_token_limit,
            "context_summary_trigger_tokens": self.context_summary_trigger_tokens,
        }

    def _latest_assistant_content(self, messages: List[Dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("content"):
                return message["content"]
        return ""

    def _build_summary_request_messages(
        self, messages: List[Dict], question: str
    ) -> List[Dict]:
        stripped_messages = self._strip_internal_message_fields(messages)
        transcript = json.dumps(stripped_messages, ensure_ascii=False)
        return [
            {
                "role": "system",
                "content": (
                    "You summarize a web research agent's prior interaction so it can "
                    "continue later with less context. Preserve established facts, "
                    "important evidence, URLs or source names when available, tool "
                    "results worth remembering, failed paths, remaining uncertainties, "
                    "and the current best candidate answer. Do not invent facts."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question:\n{question}\n\n"
                    "Conversation history to summarize:\n"
                    f"{transcript}\n\n"
                    "Write a compact continuation summary with these sections:\n"
                    "1. Facts established\n"
                    "2. Evidence and sources\n"
                    "3. Open questions\n"
                    "4. Current plan\n"
                    "5. Best current answer candidate"
                ),
            },
        ]

    def _build_messages_from_summary(
        self, system_prompt: str, question: str, summary_text: str
    ) -> List[Dict]:
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Context summary from previous turns. Treat this as authoritative "
                    "memory of prior work and continue from here without restarting:\n\n"
                    f"{summary_text}"
                ),
            },
        ]

    def _prepare_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        api_messages = self._strip_internal_message_fields(messages)

        for message in api_messages:
            if message.get("role") != "assistant":
                continue

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                continue

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                arguments = function.get("arguments")
                if isinstance(arguments, dict):
                    function["arguments"] = json.dumps(arguments, ensure_ascii=False)

        return api_messages

    def _prepare_messages_for_template(self, messages: List[Dict]) -> List[Dict]:
        template_messages = self._strip_internal_message_fields(messages)

        for message in template_messages:
            if message.get("role") != "assistant":
                continue

            tool_calls = message.get("tool_calls")
            if not tool_calls:
                continue

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    function["arguments"] = self._parse_tool_arguments(arguments)

        return template_messages

    def _parse_tool_arguments(self, arguments):
        if isinstance(arguments, dict):
            return copy.deepcopy(arguments)
        if arguments in (None, ""):
            return {}
        if not isinstance(arguments, str):
            raise ValueError("Tool arguments must be a string or dict")

        try:
            return json.loads(arguments)
        except Exception:
            if json5 is None:
                raise
            return json5.loads(arguments)

    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        tool_messages: List[Dict] = []

        for idx, tool_call in enumerate(tool_calls):
            function = tool_call.get("function", {})
            tool_name = function.get("name", "")
            raw_arguments = function.get("arguments", "")
            tool_call_id = tool_call.get("id") or f"call_{int(time.time() * 1000)}_{idx}"

            try:
                tool_args = self._parse_tool_arguments(raw_arguments)
                result = self.custom_call_tool(tool_name, tool_args)
            except Exception:
                result = (
                    'Error: Tool call arguments are not valid JSON. Tool call must '
                    'contain a valid "name" and "arguments" field.'
                )

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                }
            )

        return tool_messages

    def _emit_progress(
        self,
        progress_callback,
        *,
        question: str,
        answer: str,
        messages: List[Dict],
        round_idx: int,
        planning_port: int,
        status: str,
        prediction: Optional[str] = None,
        termination: Optional[str] = None,
        extra_payload: Optional[Dict] = None,
        final: bool = False,
    ) -> None:
        if progress_callback is None:
            return

        payload = {
            "status": status,
            "question": question,
            "answer": answer,
            "messages": copy.deepcopy(messages),
            "log": copy.deepcopy(messages),
            "prediction": prediction,
            "termination": termination,
            "round": round_idx,
            "planning_port": planning_port,
            "updated_at": utc_now_iso(),
        }
        if extra_payload:
            payload.update(copy.deepcopy(extra_payload))

        progress_callback(payload, final=final)

    def call_server(
        self, msgs, planning_port, use_tools=True, max_tries=10, max_tokens=10000
    ):
        openai_api_key = "EMPTY"
        openai_api_base = f"http://127.0.0.1:{planning_port}/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1

        for attempt in range(max_tries):
            try:
                print(
                    f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---"
                )
                request_kwargs = {
                    "model": self.model,
                    "messages": self._prepare_messages_for_api(msgs),
                    "temperature": self.llm_generate_cfg.get("temperature", 0.6),
                    "top_p": self.llm_generate_cfg.get("top_p", 0.95),
                    "logprobs": True,
                    "max_tokens": max_tokens,
                }
                top_k = self.llm_generate_cfg.get("top_k")
                if top_k is not None:
                    request_kwargs["extra_body"] = {"top_k": top_k}
                if use_tools:
                    request_kwargs["tools"] = self.tool_schemas

                chat_response = client.chat.completions.create(**request_kwargs)
                message = chat_response.choices[0].message
                content = message.content
                has_tool_calls = bool(getattr(message, "tool_calls", None))
                usage = self._normalize_usage(getattr(chat_response, "usage", None))

                if (content and content.strip()) or has_tool_calls:
                    print(
                        "--- Service call successful, received a valid response ---"
                    )
                    assistant_message = self._normalize_assistant_message(message)
                    if usage:
                        assistant_message["_usage"] = usage
                    return assistant_message

                print(f"Warning: Attempt {attempt + 1} received an empty response.")
            except (APIError, APIConnectionError, APITimeoutError) as e:
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")

        return {"role": "assistant", "content": "vllm server error!!!"}

    def add_auto_judge(self, result, auto_judge, judge_engine, messages, question, answer):
        if auto_judge and answer:
            try:
                prediction = result.get("prediction", "")
                if not prediction:
                    print("Warning: No prediction found for auto judge")
                    result["auto_judge"] = {"error": "No prediction found", "score": 0}
                    return result

                judge_result = compute_score_genrm(
                    prediction=prediction,
                    ground_truth=answer,
                    question=question,
                    engine=judge_engine,
                )
                result["auto_judge"] = judge_result
                print(
                    f"Auto Judge Score: {judge_result['score']}, Prediction: "
                    f"'{prediction[:100]}...', Ground Truth: '{answer}'"
                )
            except Exception as e:
                print(f"Auto judge failed: {e}")
                result["auto_judge"] = {"error": str(e), "score": 0}
        return result

    def count_tokens(self, messages, model="gpt-4o"):
        tokenizer = self._get_tokenizer()
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                tokenized = tokenizer.apply_chat_template(
                    self._prepare_messages_for_template(messages),
                    tools=self.tool_schemas,
                    tokenize=True,
                    add_generation_prompt=False,
                )
                if hasattr(tokenized, "keys") and "input_ids" in tokenized:
                    input_ids = tokenized["input_ids"]
                    if hasattr(input_ids, "shape"):
                        return int(input_ids.shape[-1])
                    try:
                        first_item = input_ids[0]
                    except (IndexError, TypeError):
                        return len(input_ids)
                    if isinstance(first_item, (list, tuple)):
                        return len(first_item)
                    if hasattr(first_item, "shape"):
                        return int(first_item.shape[-1])
                    return len(input_ids)
                return len(tokenized)
            except Exception as e:
                print(f"Warning: tokenizer.apply_chat_template failed, fallback to tiktoken: {e}")

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(json.dumps(messages, ensure_ascii=False)))

    def maybe_reset_context(self, messages, question, usage: Optional[Dict] = None):
        if self.context_management_strategy not in {"discard_all", "summary"}:
            return messages, None, None

        if self.context_management_strategy == "discard_all":
            max_input_tokens = self.llm_generate_cfg.get("max_input_tokens", 320000)
            reset_threshold_tokens = int(max_input_tokens * self.context_reset_threshold)
        else:
            max_input_tokens = None
            reset_threshold_tokens = self.context_summary_trigger_tokens

        token_count, token_count_source, token_usage = self._get_context_token_count(
            messages,
            usage=usage,
        )
        reset_info = {
            "strategy": self.context_management_strategy,
            "token_count": token_count,
            "token_count_source": token_count_source,
            "threshold": reset_threshold_tokens,
            "max_input_tokens": max_input_tokens,
        }
        if self.context_management_strategy == "discard_all":
            reset_info["threshold_ratio"] = self.context_reset_threshold
        if token_usage:
            reset_info["usage"] = copy.deepcopy(token_usage)

        print(
            f"context management: strategy={self.context_management_strategy}, "
            f"token_count={token_count}, source={token_count_source}, "
            f"reset_threshold={reset_threshold_tokens}",
            flush=True,
        )

        if token_count > reset_threshold_tokens:
            action = self.context_management_strategy
            print(
                f"context management: action={action} because "
                f"{token_count} > {reset_threshold_tokens}",
                flush=True,
            )
            if action == "discard_all":
                return [{"role": "user", "content": question}], action, reset_info
            return messages, action, reset_info

        return messages, None, reset_info

    def _run(
        self,
        data: str,
        model: str,
        auto_judge: bool = False,
        judge_engine: str = "deepseekchat",
        **kwargs,
    ) -> List[List[Message]]:
        self.model = model
        try:
            question = data["item"]["question"]
        except Exception:
            raw_msg = data["item"]["messages"][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        start_time = time.time()
        planning_port = data["planning_port"]
        answer = data["item"]["answer"]
        self.user_prompt = question
        if "minimax-m2.5" in model.lower():
            system_prompt = MINIMAX_25_SYSTEM_PROMPT
        elif "minimax-m2.1" in model.lower():
            system_prompt = MINIMAX_21_SYSTEM_PROMPT
        else:
            system_prompt = SYSTEM_PROMPT
        progress_callback = kwargs.get("progress_callback")
        task_metadata = kwargs.get("task_metadata", {})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        termination = "unknown"
        context_reset_events = []
        cumulative_token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_calls": 0,
        }

        def finalize_result(prediction: str, termination_reason: str):
            result = {
                "question": question,
                "answer": answer,
                "round": round,
                "messages": messages,
                "log": messages,
                "prediction": prediction,
                "termination": termination_reason,
            }
            result.update(
                self._make_context_management_stats(
                    context_reset_events, cumulative_token_usage
                )
            )
            result.update(task_metadata)

            if auto_judge and answer:
                self._emit_progress(
                    progress_callback,
                    question=question,
                    answer=answer,
                    messages=messages,
                    round_idx=round,
                    planning_port=planning_port,
                    status="judging",
                    prediction=prediction,
                    termination=termination_reason,
                    extra_payload=task_metadata,
                    final=False,
                )

            result = self.add_auto_judge(
                result, auto_judge, judge_engine, messages, question, answer
            )
            self._emit_progress(
                progress_callback,
                question=question,
                answer=answer,
                messages=messages,
                round_idx=round,
                planning_port=planning_port,
                status="finished",
                prediction=result.get("prediction"),
                termination=result.get("termination"),
                extra_payload={
                    **task_metadata,
                    "auto_judge": result.get("auto_judge"),
                    **self._make_context_management_stats(
                        context_reset_events, cumulative_token_usage
                    ),
                },
                final=True,
            )
            return result

        self._emit_progress(
            progress_callback,
            question=question,
            answer=answer,
            messages=messages,
            round_idx=round,
            planning_port=planning_port,
            status="running",
            extra_payload=task_metadata,
            final=False,
        )

        while num_llm_calls_available > 0:
            if time.time() - start_time > 150 * 60:
                prediction = "No answer found after 2h30mins"
                termination = "No answer found after 2h30mins"
                return finalize_result(prediction, termination)

            round += 1
            num_llm_calls_available -= 1
            request_messages = copy.deepcopy(messages)
            assistant_message = self.call_server(messages, planning_port, use_tools=True)
            print(f"Round {round}: {assistant_message}")
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls", [])
            has_tool_call = bool(tool_calls)
            if has_tool_call:
                tool_messages = self._execute_tool_calls(tool_calls)
                messages.extend(tool_messages)

            request_usage = self._get_call_usage(request_messages, assistant_message)
            self._accumulate_usage(cumulative_token_usage, request_usage)
            latest_usage = request_usage

            self._emit_progress(
                progress_callback,
                question=question,
                answer=answer,
                messages=messages,
                round_idx=round,
                planning_port=planning_port,
                status="running",
                prediction=assistant_message.get("content"),
                termination=None,
                extra_payload={
                    **task_metadata,
                    "num_llm_calls_available": num_llm_calls_available,
                    "has_tool_call": has_tool_call,
                    "cumulative_token_usage": cumulative_token_usage,
                },
                final=False,
            )

            if not has_tool_call:
                termination = "no_tool_call"
                break

            if (
                self.context_management_strategy == "summary"
                and cumulative_token_usage["total_tokens"]
                >= self.context_total_token_limit
            ):
                termination = "total_token_limit_reached"
                prediction = self._latest_assistant_content(messages)
                return finalize_result(prediction, termination)

            if num_llm_calls_available <= 0:
                messages.append({"role": "user", "content": FINAL_MESSAGE})
                final_request_messages = copy.deepcopy(messages)
                assistant_message = self.call_server(
                    messages,
                    planning_port,
                    use_tools=False,
                )
                messages.append(assistant_message)
                self._accumulate_usage(
                    cumulative_token_usage,
                    self._get_call_usage(final_request_messages, assistant_message),
                )
                prediction = messages[-1]["content"]
                termination = "exceed_llm_calls"
                self._emit_progress(
                    progress_callback,
                    question=question,
                    answer=answer,
                    messages=messages,
                    round_idx=round,
                    planning_port=planning_port,
                    status="running",
                    prediction=prediction,
                    termination=termination,
                    extra_payload={
                        **task_metadata,
                        "cumulative_token_usage": cumulative_token_usage,
                    },
                    final=False,
                )
                return finalize_result(prediction, termination)

            messages_before_reset = len(messages)
            messages, context_action, reset_info = self.maybe_reset_context(
                messages,
                question,
                usage=latest_usage,
            )
            if context_action == "discard_all":
                reset_event = {
                    "round": round,
                    "messages_before_reset": messages_before_reset,
                    "messages_after_reset": len(messages),
                    "num_llm_calls_available": num_llm_calls_available,
                    **(reset_info or {}),
                }
                context_reset_events.append(reset_event)
                self._emit_progress(
                    progress_callback,
                    question=question,
                    answer=answer,
                    messages=messages,
                    round_idx=round,
                    planning_port=planning_port,
                    status="running",
                    prediction=assistant_message.get("content"),
                    termination="context_reset",
                    extra_payload={
                        **task_metadata,
                        "context_reset_events": context_reset_events,
                        "context_reset_event": reset_event,
                        "cumulative_token_usage": cumulative_token_usage,
                    },
                    final=False,
                )
                continue
            if context_action == "summary":
                if num_llm_calls_available <= 0:
                    print(
                        "context management: skipping summary because there is not "
                        "enough remaining LLM-call budget",
                        flush=True,
                    )
                else:
                    summary_request_messages = self._build_summary_request_messages(
                        messages,
                        question,
                    )
                    summary_message = self.call_server(
                        summary_request_messages,
                        planning_port,
                        use_tools=False,
                        max_tokens=2048,
                    )
                    summary_usage = self._get_call_usage(
                        summary_request_messages, summary_message
                    )
                    self._accumulate_usage(cumulative_token_usage, summary_usage)
                    summary_text = (summary_message.get("content") or "").strip()
                    messages = self._build_messages_from_summary(
                        system_prompt,
                        question,
                        summary_text,
                    )
                    reset_event = {
                        "round": round,
                        "strategy": "summary",
                        "messages_before_reset": messages_before_reset,
                        "messages_after_reset": len(messages),
                        "num_llm_calls_available": num_llm_calls_available,
                        "summary_text": summary_text,
                        "summary_usage": summary_usage,
                        **(reset_info or {}),
                    }
                    context_reset_events.append(reset_event)
                    self._emit_progress(
                        progress_callback,
                        question=question,
                        answer=answer,
                        messages=messages,
                        round_idx=round,
                        planning_port=planning_port,
                        status="running",
                        prediction=assistant_message.get("content"),
                        termination="context_summary",
                        extra_payload={
                            **task_metadata,
                            "context_reset_events": context_reset_events,
                            "context_reset_event": reset_event,
                            "cumulative_token_usage": cumulative_token_usage,
                        },
                        final=False,
                    )
                    if (
                        cumulative_token_usage["total_tokens"]
                        >= self.context_total_token_limit
                    ):
                        termination = "total_token_limit_reached"
                        prediction = summary_text or self._latest_assistant_content(
                            request_messages + [assistant_message]
                        )
                        return finalize_result(prediction, termination)
                    continue

            max_tokens = 108 * 1024
            if reset_info and reset_info.get("token_count") is not None:
                token_count = reset_info["token_count"]
                token_count_source = reset_info.get("token_count_source")
            else:
                token_count, token_count_source, _ = self._get_context_token_count(
                    messages,
                    usage=latest_usage,
                )
            print(
                f"round: {round}, token count: {token_count}, "
                f"source: {token_count_source}"
            )

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")

                messages.append({"role": "user", "content": TRUNCATED_MESSAGE + FINAL_MESSAGE})
                truncated_request_messages = copy.deepcopy(messages)
                assistant_message = self.call_server(
                    messages,
                    planning_port,
                    use_tools=False,
                )
                messages.append(assistant_message)
                self._accumulate_usage(
                    cumulative_token_usage,
                    self._get_call_usage(truncated_request_messages, assistant_message),
                )
                prediction = messages[-1]["content"]
                termination = "token_limit_reached"
                self._emit_progress(
                    progress_callback,
                    question=question,
                    answer=answer,
                    messages=messages,
                    round_idx=round,
                    planning_port=planning_port,
                    status="running",
                    prediction=prediction,
                    termination=termination,
                    extra_payload={
                        **task_metadata,
                        "cumulative_token_usage": cumulative_token_usage,
                    },
                    final=False,
                )
                return finalize_result(prediction, termination)

        prediction = self._latest_assistant_content(messages) or messages[-1].get("content", "")
        if termination != "no_tool_call":
            if num_llm_calls_available <= 0:
                termination = "exceed_llm_calls"
            else:
                termination = "unknown"
        return finalize_result(prediction, termination)

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in TOOL_MAP:
            tool_payload = copy.deepcopy(tool_args)
            tool_payload["params"] = copy.deepcopy(tool_args)
            raw_result = TOOL_MAP[tool_name].call(tool_payload, **kwargs)
            result = raw_result
            return result
        return f"Error: Tool {tool_name} not found"
