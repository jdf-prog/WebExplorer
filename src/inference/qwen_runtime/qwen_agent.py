import copy
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError
from transformers import AutoTokenizer

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency in some envs
    tiktoken = None

from auto_judge import compute_score_genrm

from .qwen_tool_code_interpreter import QwenCodeInterpreterTool
from .qwen_tool_web_extractor import QwenWebExtractorTool
from .qwen_tool_web_search import QwenWebSearchTool


QWEN_DEFAULT_SYSTEM_PROMPT = (
    "Search intensity is set to high. Please conduct thorough, multi-source "
    "research and provide comprehensive, well-cited results."
)
FINAL_MESSAGE = (
    "You must absolutely not perform any FunctionCall. "
    "You can only answer based on information retrieved. "
    "Put final answer inside <answer>...</answer> tags."
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RuntimeTokenContextRewriter:
    def __init__(self, tokenizer, max_context_length: int, target_context_length: int):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.target_context_length = target_context_length
        self.fold_text = "Content folded due to space limitation"
        self.mask_token_length = self._encode_len(self.fold_text)

    def _encode_len(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        if tiktoken is not None:
            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        return max(1, len(text) // 4)

    def _get_msg_length(self, msg: dict) -> int:
        if msg.get("role") == "assistant":
            content = msg.get("reasoning_content") or msg.get("content") or ""
        else:
            content = msg.get("content") or ""
        return self._encode_len(content)

    def process(self, messages: list) -> list:
        total_tool_tokens = 0
        msg_lengths = {}
        tool_indices = []

        for idx, msg in enumerate(messages):
            if msg.get("role") == "tool":
                length = self._get_msg_length(msg)
                msg_lengths[idx] = length
                total_tool_tokens += length
                tool_indices.append(idx)

        if total_tool_tokens <= self.max_context_length:
            return copy.deepcopy(messages)

        processed_msgs = copy.deepcopy(messages)
        current_tool_tokens = total_tool_tokens
        masked_count = 0
        total_tools = len(tool_indices)

        for idx in tool_indices:
            remaining_tools = total_tools - masked_count
            if remaining_tools <= 2:
                break

            if processed_msgs[idx].get("content") == self.fold_text:
                masked_count += 1
                continue

            original_len = msg_lengths[idx]
            saved = original_len - self.mask_token_length
            if saved > 0:
                processed_msgs[idx]["content"] = self.fold_text
                current_tool_tokens -= saved
                masked_count += 1

            if current_tool_tokens <= self.target_context_length:
                break

        return processed_msgs


class QwenFunctionCallAgent:
    def __init__(self, llm: Optional[Dict] = None, **kwargs):
        llm = llm or {}
        self.llm_generate_cfg = llm.get("generate_cfg", {})
        self.model = llm.get("model", os.getenv("QWEN_MODEL_NAME", "qwen3.5-plus"))
        self.api_key = llm.get("api_key") or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = llm.get("base_url") or os.getenv(
            "QWEN_API_BASE",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.is_dashscope_backend = "dashscope.aliyuncs.com" in (self.base_url or "")
        self.model_timeout = int(os.getenv("QWEN_MODEL_TIMEOUT", "3600"))
        self.max_context_tokens = int(os.getenv("QWEN_MAX_CONTEXT_TOKENS", str(255 * 1024)))
        self.max_output_tokens_cap = int(os.getenv("QWEN_MAX_OUTPUT_TOKENS_CAP", "65500"))
        self.max_llm_calls_per_run = int(os.getenv("QWEN_MAX_LLM_CALL_PER_RUN", "200"))
        self.enable_thinking = os.getenv("QWEN_ENABLE_THINKING", "1") != "0"
        self.system_prompt = os.getenv("QWEN_SYSTEM_PROMPT", QWEN_DEFAULT_SYSTEM_PROMPT)
        self.temperature = float(
            self.llm_generate_cfg.get("temperature", os.getenv("QWEN_TEMPERATURE", "1.0"))
        )
        self.top_p = float(
            self.llm_generate_cfg.get("top_p", os.getenv("QWEN_TOP_P", "0.95"))
        )
        self.top_k = int(
            self.llm_generate_cfg.get("top_k", os.getenv("QWEN_TOP_K", "20"))
        )
        self.min_p = float(
            self.llm_generate_cfg.get("min_p", os.getenv("QWEN_MIN_P", "0.0"))
        )
        self.presence_penalty = float(
            self.llm_generate_cfg.get(
                "presence_penalty",
                os.getenv("QWEN_PRESENCE_PENALTY", "1.5"),
            )
        )
        self.repetition_penalty = float(
            self.llm_generate_cfg.get(
                "repetition_penalty",
                os.getenv("QWEN_REPETITION_PENALTY", "1.0"),
            )
        )

        self.tool_instances = [
            QwenCodeInterpreterTool(),
            QwenWebSearchTool(),
            QwenWebExtractorTool(),
        ]
        self.tool_map = {tool.name: tool for tool in self.tool_instances}
        self.tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tool_instances
        ]

        self.tokenizer = self._load_tokenizer(
            llm.get("tokenizer_path")
            or os.getenv("QWEN_TOKENIZER_PATH")
            or self.model
        )
        self.context_rewriter = RuntimeTokenContextRewriter(
            tokenizer=self.tokenizer,
            max_context_length=int(os.getenv("QWEN_TOOL_CONTEXT_MAX", "32000")),
            target_context_length=int(os.getenv("QWEN_TOOL_CONTEXT_TARGET", "5000")),
        )

    def _load_tokenizer(self, tokenizer_name_or_path: str):
        if not tokenizer_name_or_path:
            return None
        try:
            candidate = Path(tokenizer_name_or_path)
            if candidate.exists() or "/" in tokenizer_name_or_path:
                return AutoTokenizer.from_pretrained(tokenizer_name_or_path)
            local_candidate = Path(__file__).resolve().parents[4] / "models" / tokenizer_name_or_path
            if local_candidate.exists():
                return AutoTokenizer.from_pretrained(str(local_candidate))
            return AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        except Exception:
            return None

    def _normalize_usage(self, usage) -> Optional[Dict]:
        if usage is None:
            return None
        if isinstance(usage, dict):
            return copy.deepcopy(usage)
        if hasattr(usage, "model_dump"):
            return usage.model_dump(exclude_none=True)
        payload = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, key, None)
            if value is not None:
                payload[key] = value
        return payload or None

    def _token_len(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        if tiktoken is not None:
            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        return max(1, len(text) // 4)

    def count_messages_tokens(self, messages: List[Dict]) -> int:
        total = 0
        for message in messages:
            total += self._token_len(message.get("content") or "")
        return total

    def _strip_internal_message_fields(self, messages: List[Dict]) -> List[Dict]:
        return [
            {key: value for key, value in message.items() if not key.startswith("_")}
            for message in copy.deepcopy(messages)
        ]

    def _prepare_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        api_messages = self._strip_internal_message_fields(messages)
        for message in api_messages:
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls", []) or []:
                function = tool_call.get("function", {})
                arguments = function.get("arguments")
                if isinstance(arguments, dict):
                    function["arguments"] = json.dumps(arguments, ensure_ascii=False)
        return api_messages

    def _normalize_assistant_message(self, message) -> Dict:
        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
        }
        # try reasoning_content or reasoning
        reasoning_content = getattr(message, "reasoning_content", getattr(message, "reasoning", None))
        if reasoning_content not in (None, ""):
            assistant_message["reasoning_content"] = reasoning_content
        if getattr(message, "tool_calls", None):
            assistant_message["tool_calls"] = [
                tool_call.model_dump(exclude_none=True)
                if hasattr(tool_call, "model_dump")
                else copy.deepcopy(tool_call)
                for tool_call in message.tool_calls
            ]
        return assistant_message

    def _parse_tool_arguments(self, tool_name: str, raw_arguments):
        if isinstance(raw_arguments, dict):
            return copy.deepcopy(raw_arguments)
        if raw_arguments in (None, ""):
            return {}
        if not isinstance(raw_arguments, str):
            raise ValueError("Tool arguments must be a string or dict")
        try:
            return json.loads(raw_arguments)
        except json.JSONDecodeError:
            if tool_name == "code_interpreter":
                return {"code": raw_arguments}
            if tool_name == "web_search":
                return {"queries": [raw_arguments]}
            return {}

    def _execute_tool_calls(self, tool_calls: List[Dict]) -> List[Dict]:
        tool_messages = []
        for idx, tool_call in enumerate(tool_calls):
            function = tool_call.get("function", {})
            tool_name = function.get("name", "")
            raw_arguments = function.get("arguments", "")
            tool_call_id = tool_call.get("id") or f"call_{int(time.time() * 1000)}_{idx}"
            tool = self.tool_map.get(tool_name)
            if tool is None:
                result = f"Error: Tool {tool_name} not found"
            else:
                try:
                    tool_args = self._parse_tool_arguments(tool_name, raw_arguments)
                    result = tool.call(tool_args)
                except Exception as exc:
                    result = f"Error executing tool: {exc}"
            tool_messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
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
            "updated_at": utc_now_iso(),
        }
        if extra_payload:
            payload.update(copy.deepcopy(extra_payload))

        progress_callback(payload, final=final)

    def call_model(self, messages: List[Dict], use_tools: bool = True, max_tries: int = 10):
        if not self.api_key:
            raise ValueError(
                "QWEN_API_KEY or DASHSCOPE_API_KEY must be set for Qwen inference."
            )

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.model_timeout,
        )

        for attempt in range(max_tries):
            try:
                prompt_tokens = self.count_messages_tokens(messages)
                dynamic_max_tokens = max(
                    1,
                    min(
                        self.max_context_tokens - prompt_tokens,
                        self.max_output_tokens_cap,
                    ),
                )
                request_kwargs = {
                    "model": self.model,
                    "messages": self._prepare_messages_for_api(messages),
                    "max_tokens": dynamic_max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "presence_penalty": self.presence_penalty,
                }
                extra_body = {
                    "top_k": self.top_k,
                    "min_p": self.min_p,
                    "repetition_penalty": self.repetition_penalty,
                }
                if self.is_dashscope_backend:
                    extra_body["enable_thinking"] = self.enable_thinking
                    request_kwargs["extra_headers"] = {
                        "X-DashScope-DataInspection": '{"input":"disable","output":"disable"}'
                    }
                if extra_body:
                    request_kwargs["extra_body"] = extra_body
                if use_tools:
                    request_kwargs["tools"] = self.tool_schemas

                chat_response = client.chat.completions.create(**request_kwargs)
                if not chat_response.choices:
                    print(
                        f"Warning: Attempt {attempt + 1} received no choices.",
                        flush=True,
                    )
                    continue

                choice = chat_response.choices[0]
                message = choice.message
                finish_reason = choice.finish_reason
                usage = self._normalize_usage(getattr(chat_response, "usage", None))
                assistant_message = self._normalize_assistant_message(message)
                if usage:
                    assistant_message["_usage"] = usage

                has_tool_calls = bool(assistant_message.get("tool_calls"))
                has_content = bool((assistant_message.get("content") or "").strip())
                has_reasoning = bool((assistant_message.get("reasoning_content") or "").strip())

                if has_content or has_tool_calls:
                    return assistant_message, finish_reason

                print(
                    (
                        f"Warning: Attempt {attempt + 1} received an empty assistant "
                        f"response (finish_reason={finish_reason}, reasoning={has_reasoning})."
                    ),
                    flush=True,
                )
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                print(f"Qwen model retry {attempt} after API error: {exc}", flush=True)
            except Exception as exc:
                print(f"Qwen model retry {attempt} after unexpected error: {exc}", flush=True)

            if attempt < max_tries - 1:
                time.sleep(min(30, (2 ** attempt) + random.uniform(0, 1)))

        return {"role": "assistant", "content": "qwen model error"}, "error"

    def add_auto_judge(self, result, auto_judge, judge_engine, messages, question, answer):
        if auto_judge and answer:
            try:
                prediction = result.get("prediction", "")
                if not prediction:
                    result["auto_judge"] = {"error": "No prediction found", "score": 0}
                    return result
                result["auto_judge"] = compute_score_genrm(
                    prediction=prediction,
                    ground_truth=answer,
                    question=question,
                    engine=judge_engine,
                )
            except Exception as exc:
                result["auto_judge"] = {"error": str(exc), "score": 0}
        return result

    def _run(
        self,
        data: Dict,
        model: str,
        auto_judge: bool = False,
        judge_engine: str = "openai",
        **kwargs,
    ) -> Dict:
        self.model = model or self.model
        item = data["item"]
        question = item.get("question", "").strip()
        if not question:
            raw_msg = item.get("messages", [{}, {}])[1].get("content", "")
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg
        answer = item.get("answer", "")
        progress_callback = kwargs.get("progress_callback")
        task_metadata = kwargs.get("task_metadata", {})

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        fold_triggered_step = None
        round_idx = 0
        termination = "unknown"

        def finalize(prediction: str, termination_reason: str):
            result = {
                "question": question,
                "answer": answer,
                "round": round_idx,
                "messages": messages,
                "log": messages,
                "prediction": prediction,
                "termination": termination_reason,
                "context_fold_trigger_step": fold_triggered_step,
            }
            result.update(task_metadata)
            result = self.add_auto_judge(
                result,
                auto_judge,
                judge_engine,
                messages,
                question,
                answer,
            )
            self._emit_progress(
                progress_callback,
                question=question,
                answer=answer,
                messages=messages,
                round_idx=round_idx,
                status="finished",
                prediction=result.get("prediction"),
                termination=result.get("termination"),
                extra_payload={
                    **task_metadata,
                    "auto_judge": result.get("auto_judge"),
                    "context_fold_trigger_step": fold_triggered_step,
                },
                final=True,
            )
            return result

        self._emit_progress(
            progress_callback,
            question=question,
            answer=answer,
            messages=messages,
            round_idx=round_idx,
            status="running",
            extra_payload=task_metadata,
            final=False,
        )

        while round_idx < self.max_llm_calls_per_run:
            if round_idx + 1 == self.max_llm_calls_per_run:
                messages.append({"role": "tool", "name": "hint", "content": FINAL_MESSAGE})

            inference_messages = self.context_rewriter.process(messages)
            if fold_triggered_step is None:
                for message in inference_messages:
                    if message.get("content") == self.context_rewriter.fold_text:
                        fold_triggered_step = round_idx
                        break

            assistant_message, finish_reason = self.call_model(
                inference_messages,
                use_tools=True,
            )
            messages.append(assistant_message)
            round_idx += 1

            tool_calls = assistant_message.get("tool_calls", []) or []
            if finish_reason == "tool_calls" and tool_calls:
                tool_messages = self._execute_tool_calls(tool_calls)
                messages.extend(tool_messages)
                self._emit_progress(
                    progress_callback,
                    question=question,
                    answer=answer,
                    messages=messages,
                    round_idx=round_idx,
                    status="running",
                    prediction=assistant_message.get("content"),
                    termination=None,
                    extra_payload={
                        **task_metadata,
                        "has_tool_call": True,
                        "context_fold_trigger_step": fold_triggered_step,
                    },
                    final=False,
                )
                continue

            termination = "no_tool_call" if finish_reason != "error" else "model_error"
            prediction = assistant_message.get("content", "")
            return finalize(prediction, termination)

        final_prompt_messages = messages + [{"role": "user", "content": FINAL_MESSAGE}]
        assistant_message, _ = self.call_model(final_prompt_messages, use_tools=False)
        messages.append({"role": "user", "content": FINAL_MESSAGE})
        messages.append(assistant_message)
        return finalize(assistant_message.get("content", ""), "exceed_llm_calls")
