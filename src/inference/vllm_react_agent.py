import copy
import json
import os
import random
import re
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
MAX_TOKENS_SAFETY_MARGIN = int(os.getenv("MAX_TOKENS_SAFETY_MARGIN", "1024"))

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


def strip_think_blocks(text: Optional[str]) -> str:
    if text is None:
        return ""

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*\Z", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def normalize_context_management_strategy(strategy: str) -> str:
    normalized = (strategy or "none").strip().lower().replace("-", "_")
    if normalized in {"discard", "discard_all"}:
        return "discard_all"
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


class ToolMessageContextRewriter:
    def __init__(self, tokenizer, max_context_length: int, target_context_length: int):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.target_context_length = target_context_length
        self.fold_text = "Content folded due to space limitation"
        self.mask_token_length = self._encode_len(self.fold_text)

    def _encode_len(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text))
        return len(tiktoken.get_encoding("cl100k_base").encode(text))

    def _get_msg_length(self, msg: Dict) -> int:
        if msg.get("role") == "assistant":
            content = msg.get("reasoning_content") or msg.get("content") or ""
        else:
            content = msg.get("content") or ""
        return self._encode_len(content)

    def process(self, messages: List[Dict]) -> List[Dict]:
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


class MultiTurnReactAgent(FnCallAgent):
    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
        llm: Optional[Union[Dict, BaseChatModel]] = None,
        **kwargs,
    ):
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = self._resolve_model_path(llm["model"])
        self.context_management_strategy = normalize_context_management_strategy(
            os.getenv("CONTEXT_MANAGEMENT_STRATEGY", "none")
        )
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
        self.tool_instances = self._resolve_tool_instances(function_list)
        self.tool_map = {tool.name: tool for tool in self.tool_instances}
        self.tool_schemas = self._build_tool_schemas()
        self.tool_context_rewriter = ToolMessageContextRewriter(
            tokenizer=self.tokenizer,
            max_context_length=int(
                os.getenv(
                    "TOOL_CONTEXT_MAX",
                    os.getenv("QWEN_TOOL_CONTEXT_MAX", "32000"),
                )
            ),
            target_context_length=int(
                os.getenv(
                    "TOOL_CONTEXT_TARGET",
                    os.getenv("QWEN_TOOL_CONTEXT_TARGET", "5000"),
                )
            ),
        )

    def _available_tool_instances(self) -> List[BaseTool]:
        return [
            WebExplorerBrowse(),
            WebExplorerSearch(),
        ]

    def _resolve_tool_instances(
        self, function_list: Optional[List[Union[str, Dict, BaseTool]]]
    ) -> List[BaseTool]:
        available_tools = self._available_tool_instances()
        if not function_list:
            return available_tools

        tool_map = {tool.name: tool for tool in available_tools}
        selected_tools: List[BaseTool] = []
        for tool_spec in function_list:
            if isinstance(tool_spec, BaseTool):
                selected_tools.append(tool_spec)
                continue
            if isinstance(tool_spec, dict):
                tool_name = tool_spec.get("name")
            else:
                tool_name = str(tool_spec)
            if tool_name in tool_map:
                selected_tools.append(tool_map[tool_name])

        return selected_tools or available_tools

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
        for tool in self.tool_instances:
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

    def _prepare_inference_messages(self, messages: List[Dict]) -> List[Dict]:
        if self.context_management_strategy != "fold_tool":
            return messages
        return self.tool_context_rewriter.process(messages)

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

    def _move_thinking_to_reasoning_content(self, message: Dict) -> None:
        thinking_payload = message.pop("thinking", None)
        if message.get("reasoning_content") or thinking_payload is None:
            return
        if isinstance(thinking_payload, dict):
            message["reasoning_content"] = thinking_payload.get("thinking") or ""
        elif isinstance(thinking_payload, str):
            message["reasoning_content"] = thinking_payload

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

    def _ensure_message_usage(
        self, request_messages: List[Dict], assistant_message: Dict
    ) -> Dict:
        usage = self._get_call_usage(request_messages, assistant_message)
        if usage and not assistant_message.get("_usage"):
            assistant_message["_usage"] = copy.deepcopy(usage)
        return usage

    def _accumulate_context_reset_usage(
        self, cumulative_usage: Dict, usage: Optional[Dict]
    ) -> None:
        if not usage:
            return

        prompt_tokens = usage.get("prompt_tokens")

        try:
            if prompt_tokens is not None:
                cumulative_usage["context_reset_prompt_tokens"] += int(prompt_tokens)
        except (TypeError, ValueError):
            pass

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
            "cumulative_token_usage_metric": "context_reset_prompt_tokens",
            "context_total_token_limit": self.context_total_token_limit,
            "context_summary_trigger_tokens": self.context_summary_trigger_tokens,
        }

    def _latest_assistant_content(self, messages: List[Dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("content"):
                return strip_think_blocks(message["content"])
        return ""

    def _format_message_for_summary(self, message: Dict, step_idx: int) -> str:
        role = message.get("role", "unknown")
        lines = [f"[step {step_idx}] {role}"]

        if role == "assistant":
            reasoning_content = strip_think_blocks(
                message.get("reasoning_content") or ""
            )
            content = strip_think_blocks(message.get("content") or "")
            if reasoning_content:
                lines.extend(["<think>", reasoning_content, "</think>"])
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                lines.append("<tool_calls>")
                for tool_call in tool_calls:
                    lines.append(
                        json.dumps(tool_call, ensure_ascii=False, indent=4)
                    )
                lines.append("</tool_calls>")
            if content:
                lines.append(content)
        else:
            content = strip_think_blocks(message.get("content") or "")
            if content:
                lines.append(content)

        return "\n".join(lines)

    def _format_conversation_history_for_summary(
        self, messages: List[Dict]
    ) -> str:
        stripped_messages = self._strip_internal_message_fields(messages)
        return "\n".join(
            self._format_message_for_summary(message, idx)
            for idx, message in enumerate(stripped_messages)
        )

    def _build_summary_request_messages(
        self, messages: List[Dict], question: str
    ) -> List[Dict]:
        transcript = self._format_conversation_history_for_summary(messages)
        summary_prompt = f"""Your task is to create a detailed summary of the conversation so far,
paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns,
and architectural decisions that would be essential for continuing development
work without losing context.
Before providing your final summary, organize your thoughts in an "## Analysis"
section to ensure you've covered all necessary points. In your analysis process:
1. Chronologically analyze each message and section of the conversation. For
   each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like file names, full code snippets, function signatures,
     file edits, etc
2. Double-check for technical accuracy and completeness, addressing each
   required element thoroughly.
Your summary should include the following sections:
1. Primary Request and Intent: Capture all of the user's explicit requests and
   intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies,
   and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections
   examined, modified, or created. Pay special attention to the most recent
   messages and include full code snippets where applicable and include a
   summary of why this file read or edit is important.
4. Problem Solving: Document problems solved and any ongoing troubleshooting
   efforts.
5. Pending Tasks: Outline any pending tasks that you have explicitly been
   asked to work on.
6. Current Work: Describe in detail precisely what was being worked on
   immediately before this summary request, paying special attention to the
   most recent messages from both user and assistant. Include file names and
   code snippets where applicable.
7. Optional Next Step: List the next step that you will take that is related
   to the most recent work you were doing. IMPORTANT: ensure that this step
   is DIRECTLY in line with the user's explicit requests, and the task you
   were working on immediately before this summary request. If your last task
   was concluded, then only list next steps if they are explicitly in line
   with the users request. Do not start on tangential requests without
   confirming with the user first.
8. If there is a next step, include direct quotes from the most recent
   conversation showing exactly what task you were working on and where you
   left off. This should be verbatim to ensure there's no drift in task
   interpretation.
Here's an example of how your output should be structured:
---
## Analysis
[Your thought process, ensuring all points are covered thoroughly and accurately]
## Summary
### 1. Primary Request and Intent
...
### 7. Optional Next Step
[Optional Next step to take]
---
Please provide your summary based on the conversation so far, following this
structure and ensuring precision and thoroughness in your response.




[USER]
{question}
<conversation_history>
{transcript}
</conversation_history>
Directly output the summary content without any other text."""
        return [{"role": "user", "content": summary_prompt}]

    def _last_user_message(self, messages: List[Dict], question: str) -> Dict:
        for message in reversed(messages):
            if message.get("role") == "user":
                return copy.deepcopy(message)
        return {"role": "user", "content": question}

    def _build_messages_after_summary(
        self, messages: List[Dict], system_prompt: str, question: str
    ) -> List[Dict]:
        system_messages = []
        if messages and messages[0].get("role") == "system":
            system_messages.append(copy.deepcopy(messages[0]))
        elif system_prompt:
            system_messages.append({"role": "system", "content": system_prompt})

        return system_messages + [self._last_user_message(messages, question)]

    def _append_summary_to_thinking(self, thinking_content: str, summary_text: str) -> str:
        formatted_summary = (
            "<minimax:context_summary>\n"
            f"{summary_text}\n"
            "</minimax:context_summary>\n\n"
        )
        return f"{thinking_content}{formatted_summary}"

    def _inject_pending_summary_to_thinking(
        self, messages: List[Dict], pending_summary: Optional[str]
    ) -> bool:
        if not pending_summary:
            return False

        for message in reversed(messages):
            if message.get("role") != "assistant":
                continue

            thinking_payload = message.get("thinking")
            if isinstance(thinking_payload, dict):
                thinking_content = strip_think_blocks(
                    thinking_payload.get("thinking") or ""
                )
                thinking_payload["thinking"] = self._append_summary_to_thinking(
                    thinking_content,
                    pending_summary,
                )
                return True
            if isinstance(thinking_payload, str):
                message["thinking"] = self._append_summary_to_thinking(
                    strip_think_blocks(thinking_payload),
                    pending_summary,
                )
                return True

            if message.get("reasoning_content"):
                reasoning_content = strip_think_blocks(message["reasoning_content"])
                message["reasoning_content"] = self._append_summary_to_thinking(
                    reasoning_content,
                    pending_summary,
                )
                return True

            content = message.get("content") or ""
            think_match = re.search(
                r"<think>(.*?)</think>",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )
            if think_match:
                thinking_content = strip_think_blocks(think_match.group(1))
                new_thinking = (
                    "<think>"
                    + self._append_summary_to_thinking(
                        thinking_content,
                        pending_summary,
                    )
                    + "</think>"
                )
                message["content"] = (
                    content[: think_match.start()]
                    + new_thinking
                    + content[think_match.end() :]
                )
                return True

            message["content"] = (
                "<think>"
                + self._append_summary_to_thinking("", pending_summary)
                + "</think>\n"
                + content
            )
            return True

        return False

    def _prepare_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        api_messages = self._strip_internal_message_fields(messages)

        for message in api_messages:
            self._move_thinking_to_reasoning_content(message)

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
            self._move_thinking_to_reasoning_content(message)

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
            "tools": copy.deepcopy(self.tool_schemas),
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

    def call_server(self, msgs, planning_port, use_tools=True, max_tries=10):
        openai_api_key = "EMPTY"
        openai_api_base = f"http://127.0.0.1:{planning_port}/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1
        dynamic_max_tokens, prompt_tokens, max_input_tokens = (
            self._get_dynamic_max_tokens(
                msgs,
                use_tools=use_tools,
            )
        )
        remaining_tokens = max_input_tokens - prompt_tokens
        print(
            "dynamic max_tokens: "
            f"{dynamic_max_tokens} "
            f"(remaining_context_tokens={remaining_tokens}, "
            f"max_input_tokens={max_input_tokens}, "
            f"prompt_tokens={prompt_tokens}, "
            f"safety_margin={MAX_TOKENS_SAFETY_MARGIN})",
            flush=True,
        )
        if dynamic_max_tokens <= 0:
            return {
                "role": "assistant",
                "content": "",
                "_finish_reason": "input_token_limit_reached",
                "_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 0,
                    "total_tokens": prompt_tokens,
                    "estimated": True,
                },
            }

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
                    "max_tokens": dynamic_max_tokens,
                }
                presence_penalty = self.llm_generate_cfg.get("presence_penalty")
                if presence_penalty is not None:
                    request_kwargs["presence_penalty"] = presence_penalty
                extra_body = {}
                top_k = self.llm_generate_cfg.get("top_k")
                min_p = self.llm_generate_cfg.get("min_p")
                repetition_penalty = self.llm_generate_cfg.get("repetition_penalty")
                if top_k is not None:
                    extra_body["top_k"] = top_k
                if min_p is not None:
                    extra_body["min_p"] = min_p
                if repetition_penalty is not None:
                    extra_body["repetition_penalty"] = repetition_penalty
                if extra_body:
                    request_kwargs["extra_body"] = extra_body
                if use_tools:
                    request_kwargs["tools"] = self.tool_schemas

                chat_response = client.chat.completions.create(**request_kwargs)
                choice = chat_response.choices[0]
                message = choice.message
                finish_reason = getattr(choice, "finish_reason", None)
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
                    if finish_reason:
                        assistant_message["_finish_reason"] = finish_reason
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

    def count_tokens(self, messages, model="gpt-4o", include_tools=True):
        tokenizer = self._get_tokenizer()
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                tokenized = tokenizer.apply_chat_template(
                    self._prepare_messages_for_template(messages),
                    tools=self.tool_schemas if include_tools else None,
                    tokenize=True,
                    add_generation_prompt=True,
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

        token_payload = {
            "messages": self._prepare_messages_for_api(messages),
        }
        if include_tools:
            token_payload["tools"] = self.tool_schemas

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(json.dumps(token_payload, ensure_ascii=False)))

    def _get_dynamic_max_tokens(
        self,
        messages: List[Dict],
        use_tools: bool,
    ):
        prompt_tokens = self.count_tokens(messages, include_tools=use_tools)
        max_input_tokens = int(self.llm_generate_cfg.get("max_input_tokens", 196608))
        remaining_tokens = (
            max_input_tokens - prompt_tokens - MAX_TOKENS_SAFETY_MARGIN
        )
        return remaining_tokens, prompt_tokens, max_input_tokens

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
            # system_prompt = MINIMAX_25_SYSTEM_PROMPT
            system_prompt = SYSTEM_PROMPT
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
        context_fold_trigger_step = None
        context_reset_events = []
        cumulative_token_usage = {
            "context_reset_prompt_tokens": 0,
            "estimated_calls": 0,
        }
        pending_summary_for_thinking = None
        history_context = []

        def finalize_result(prediction: str, termination_reason: str):
            result = {
                "question": question,
                "answer": answer,
                "tools": copy.deepcopy(self.tool_schemas),
                "round": round,
                "messages": messages,
                "log": messages,
                "prediction": prediction,
                "termination": termination_reason,
                "context_fold_trigger_step": context_fold_trigger_step,
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
                    "context_fold_trigger_step": context_fold_trigger_step,
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

            if pending_summary_for_thinking:
                injected = self._inject_pending_summary_to_thinking(
                    messages,
                    pending_summary_for_thinking,
                )
                if injected:
                    pending_summary_for_thinking = None

            round += 1
            num_llm_calls_available -= 1
            inference_messages = self._prepare_inference_messages(messages)
            if context_fold_trigger_step is None and inference_messages is not messages:
                for message in inference_messages:
                    if message.get("content") == self.tool_context_rewriter.fold_text:
                        context_fold_trigger_step = round
                        break
            request_messages = copy.deepcopy(inference_messages)
            assistant_message = self.call_server(
                inference_messages,
                planning_port,
                use_tools=True,
            )
            print(f"Round {round}: {assistant_message}")
            messages.append(assistant_message)

            tool_calls = assistant_message.get("tool_calls", [])
            has_tool_call = bool(tool_calls)
            if has_tool_call:
                tool_messages = self._execute_tool_calls(tool_calls)
                messages.extend(tool_messages)

            request_usage = self._ensure_message_usage(
                request_messages, assistant_message
            )
            latest_usage = request_usage
            finish_reason = assistant_message.get("_finish_reason")

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
                    "context_fold_trigger_step": context_fold_trigger_step,
                },
                final=False,
            )

            if not has_tool_call:
                termination = (
                    "max_tokens_reached"
                    if finish_reason == "length"
                    else (
                        "input_token_limit_reached"
                        if finish_reason == "input_token_limit_reached"
                        else "no_tool_call"
                    )
                )
                break

            if num_llm_calls_available <= 0:
                messages.append({"role": "user", "content": FINAL_MESSAGE})
                final_inference_messages = self._prepare_inference_messages(messages)
                final_request_messages = copy.deepcopy(final_inference_messages)
                assistant_message = self.call_server(
                    final_inference_messages,
                    planning_port,
                    use_tools=False,
                )
                messages.append(assistant_message)
                self._ensure_message_usage(
                    final_request_messages, assistant_message
                )
                prediction = strip_think_blocks(messages[-1].get("content") or "")
                finish_reason = assistant_message.get("_finish_reason")
                termination = (
                    "max_tokens_reached"
                    if finish_reason == "length"
                    else (
                        "input_token_limit_reached"
                        if finish_reason == "input_token_limit_reached"
                        else "exceed_llm_calls"
                    )
                )
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
                        "context_fold_trigger_step": context_fold_trigger_step,
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
                self._accumulate_context_reset_usage(
                    cumulative_token_usage,
                    (reset_info or {}).get("usage"),
                )
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
                        "context_fold_trigger_step": context_fold_trigger_step,
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
                    )
                    summary_usage = self._get_call_usage(
                        summary_request_messages, summary_message
                    )
                    if summary_usage and not summary_message.get("_usage"):
                        summary_message["_usage"] = copy.deepcopy(summary_usage)
                    self._accumulate_context_reset_usage(
                        cumulative_token_usage,
                        (reset_info or {}).get("usage"),
                    )
                    summary_text = strip_think_blocks(
                        summary_message.get("content") or ""
                    )
                    pending_summary_for_thinking = summary_text
                    history_context.append(copy.deepcopy(messages))
                    messages = self._build_messages_after_summary(
                        messages,
                        system_prompt,
                        question,
                    )
                    reset_event = {
                        "round": round,
                        "strategy": "summary",
                        "messages_before_reset": messages_before_reset,
                        "messages_after_reset": len(messages),
                        "num_llm_calls_available": num_llm_calls_available,
                        "summary_text": summary_text,
                        "summary_usage": summary_usage,
                        "summary_injection": "pending_assistant_thinking",
                        "history_context_count": len(history_context),
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
                            "context_fold_trigger_step": context_fold_trigger_step,
                        },
                        final=False,
                    )
                    if (
                        cumulative_token_usage["context_reset_prompt_tokens"]
                        >= self.context_total_token_limit
                    ):
                        termination = "total_token_limit_reached"
                        pre_summary_messages = (
                            history_context[-1]
                            if history_context
                            else request_messages + [assistant_message]
                        )
                        prediction = self._latest_assistant_content(
                            pre_summary_messages
                        )
                        return finalize_result(prediction, termination)
                    continue

            max_tokens = 108 * 1024
            if reset_info and reset_info.get("token_count") is not None:
                token_count = reset_info["token_count"]
                token_count_source = reset_info.get("token_count_source")
            else:
                token_count_messages = self._prepare_inference_messages(messages)
                token_count, token_count_source, _ = self._get_context_token_count(
                    token_count_messages,
                    usage=None if token_count_messages is not messages else latest_usage,
                )
            print(
                f"round: {round}, token count: {token_count}, "
                f"source: {token_count_source}"
            )

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")

                messages.append({"role": "user", "content": TRUNCATED_MESSAGE + FINAL_MESSAGE})
                truncated_inference_messages = self._prepare_inference_messages(messages)
                truncated_request_messages = copy.deepcopy(truncated_inference_messages)
                assistant_message = self.call_server(
                    truncated_inference_messages,
                    planning_port,
                    use_tools=False,
                )
                messages.append(assistant_message)
                self._ensure_message_usage(
                    truncated_request_messages, assistant_message
                )
                prediction = strip_think_blocks(messages[-1].get("content") or "")
                finish_reason = assistant_message.get("_finish_reason")
                termination = (
                    "max_tokens_reached"
                    if finish_reason == "length"
                    else (
                        "input_token_limit_reached"
                        if finish_reason == "input_token_limit_reached"
                        else "token_limit_reached"
                    )
                )
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
                        "context_fold_trigger_step": context_fold_trigger_step,
                    },
                    final=False,
                )
                return finalize_result(prediction, termination)

        prediction = self._latest_assistant_content(messages) or strip_think_blocks(
            messages[-1].get("content", "")
        )
        if termination != "no_tool_call":
            if num_llm_calls_available <= 0:
                termination = "exceed_llm_calls"
            else:
                termination = "unknown"
        return finalize_result(prediction, termination)

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in self.tool_map:
            tool_payload = copy.deepcopy(tool_args)
            tool_payload["params"] = copy.deepcopy(tool_args)
            raw_result = self.tool_map[tool_name].call(tool_payload, **kwargs)
            result = raw_result
            return result
        return f"Error: Tool {tool_name} not found"
