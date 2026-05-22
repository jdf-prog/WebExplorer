import copy
import importlib.util
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

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
CONTEXT_SUMMARY_MAX_TOKENS_CAP = int(
    os.getenv("CONTEXT_SUMMARY_MAX_TOKENS_CAP", "32768")
)
TASK_TIME_LIMIT_MINUTES = float(os.getenv("WEBEXPLORER_TASK_TIME_LIMIT_MINUTES", "150"))
DEFAULT_NAM_MAX_MEMORY_SIZE = 32000
DEFAULT_NAM_TRIGGER_LOW_FRAC = 0.25
DEFAULT_NAM_TRIGGER_HIGH_FRAC = 0.75
VLLM_SERVER_ERROR_MESSAGE = "vllm server error!!!"
TOOL_CALL_REPAIR_MAX_RESAMPLES = int(
    os.getenv("WEBEXPLORER_TOOL_CALL_REPAIR_MAX_RESAMPLES", "3")
)
RAW_TOOL_CALL_RESAMPLE_MAX_ATTEMPTS = int(
    os.getenv("WEBEXPLORER_RAW_TOOL_CALL_RESAMPLES", "1")
)


class VllmServerError(RuntimeError):
    """Raised when the local vLLM server is unavailable after all retries."""


class ToolCallParseError(RuntimeError):
    """Raised when a model emits an unparseable raw tool call."""


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
QWEN_REPRO_SYSTEM_PROMPT = (
    "Search intensity is set to high. Please conduct thorough, multi-source "
    "research and provide comprehensive, well-cited results."
)
MINIMAX_21_SYSTEM_PROMPT = "You are a helpful assistant. Your name is MiniMax-M2.1 and is built by MiniMax."
MINIMAX_25_SYSTEM_PROMPT = "You are a helpful assistant. Your name is MiniMax-M2.5 and is built by MiniMax."
NAM_STAGE1_PROMPT = """You are the context memory controller for an agent.

Your job is to decide whether the current conversation history should be summarized before the next assistant response.

The agent has a bounded active context memory. If the context is getting large, contains many tool results, repeated search/browse outputs, or older intermediate reasoning that can be compressed while preserving task-critical facts, answer YES. If the context is still compact and preserving the exact raw history is more useful than summarizing, answer NO.

Consider:
- Current token count
- Low and high trigger thresholds
- Number of messages
- Whether the recent history contains large tool observations
- Whether important facts, constraints, URLs, search results, and pending subtasks can be safely captured in a summary
- Whether continuing without summarization risks exceeding context memory soon

Output exactly one word after any thinking: YES or NO."""

TOOL_CLASS = [
    WebExplorerBrowse(),
    WebExplorerSearch(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


def task_time_limit_seconds() -> Optional[float]:
    if TASK_TIME_LIMIT_MINUTES <= 0:
        return None
    return TASK_TIME_LIMIT_MINUTES * 60


def task_time_limit_termination() -> str:
    return f"No answer found after {TASK_TIME_LIMIT_MINUTES:g}mins"


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


def normalize_model_server_backend(backend: str) -> str:
    normalized = (backend or "auto").strip().lower().replace("-", "_")
    if normalized in {"vllm", "sglang"}:
        return normalized
    return "auto"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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

    def _count_tool_tokens(self, messages: List[Dict]) -> tuple[int, Dict[int, int], List[int]]:
        total_tool_tokens = 0
        msg_lengths = {}
        tool_indices = []

        for idx, msg in enumerate(messages):
            if msg.get("role") == "tool":
                length = self._get_msg_length(msg)
                msg_lengths[idx] = length
                total_tool_tokens += length
                tool_indices.append(idx)

        return total_tool_tokens, msg_lengths, tool_indices

    def tool_token_count(self, messages: List[Dict]) -> int:
        total_tool_tokens, _, _ = self._count_tool_tokens(messages)
        return total_tool_tokens

    def process_with_stats(self, messages: List[Dict]) -> tuple[List[Dict], Dict]:
        total_tool_tokens, msg_lengths, tool_indices = self._count_tool_tokens(messages)
        stats = {
            "tool_tokens_before": total_tool_tokens,
            "tool_tokens_after": total_tool_tokens,
            "tool_messages_before": len(tool_indices),
            "folded_tool_messages": 0,
            "fold_applied": False,
        }

        if total_tool_tokens <= self.max_context_length:
            return copy.deepcopy(messages), stats

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
                stats["fold_applied"] = True

            if current_tool_tokens <= self.target_context_length:
                break

        stats["tool_tokens_after"] = current_tool_tokens
        stats["folded_tool_messages"] = masked_count
        return processed_msgs, stats

    def process(self, messages: List[Dict]) -> List[Dict]:
        processed_msgs, _ = self.process_with_stats(messages)
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
        model_basename = os.path.basename(str(llm["model"]).rstrip("/")).lower()
        self.is_qwen_model = "qwen" in model_basename
        self.is_deepseek_model = "deepseek" in model_basename
        self.deepseek_max_output_tokens_cap = int(
            os.getenv("DEEPSEEK_MAX_OUTPUT_TOKENS_CAP", "65536")
        )
        self.context_management_strategy = normalize_context_management_strategy(
            os.getenv("CONTEXT_MANAGEMENT_STRATEGY", "none")
        )
        self.model_server_backend = normalize_model_server_backend(
            os.getenv("MODEL_SERVER_BACKEND", "auto")
        )
        self.context_reset_threshold = float(os.getenv("CONTEXT_RESET_THRESHOLD", "0.3"))
        keep_system_default = "1" if self.is_deepseek_model else "0"
        self.discard_all_keep_system_prompt = os.getenv(
            "DISCARD_ALL_KEEP_SYSTEM_PROMPT", keep_system_default
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.discard_prompt_threshold_ratio = float(
            os.getenv("DISCARD_PROMPT_THRESHOLD_RATIO", "0.85")
        )
        self.discard_history_tool_tokens = int(
            os.getenv("DISCARD_HISTORY_TOOL_TOKENS", "0")
        )
        self.discard_history_min_rounds = int(
            os.getenv("DISCARD_HISTORY_MIN_ROUNDS", "0")
        )
        self.discard_history_max_rounds = int(
            os.getenv("DISCARD_HISTORY_MAX_ROUNDS", "0")
        )
        self.nam_max_memory_size = int(
            os.getenv("NAM_MAX_MEMORY_SIZE", str(DEFAULT_NAM_MAX_MEMORY_SIZE))
        )
        self.nam_trigger_low_frac = float(
            os.getenv("NAM_TRIGGER_LOW_FRAC", str(DEFAULT_NAM_TRIGGER_LOW_FRAC))
        )
        self.nam_trigger_high_frac = float(
            os.getenv("NAM_TRIGGER_HIGH_FRAC", str(DEFAULT_NAM_TRIGGER_HIGH_FRAC))
        )
        self.nam_stage1_enabled = env_flag("NAM_STAGE1_ENABLED", False)
        default_summary_tag = (
            "context_summary"
            if "qwen" in model_basename
            else "minimax:context_summary"
        )
        self.context_summary_tag = os.getenv(
            "CONTEXT_SUMMARY_TAG",
            default_summary_tag,
        )
        self.context_summary_trigger_tokens = self.nam_max_memory_size
        self.context_total_token_limit = int(
            os.getenv("CONTEXT_TOTAL_TOKEN_LIMIT", "1000000")
        )
        self.tokenizer = None
        self._tokenizer_initialized = False
        self._vllm_dsv4_encoding = None
        self._vllm_dsv4_encoding_initialized = False
        self._sglang_dsv4_encoding = None
        self._sglang_dsv4_encoding_initialized = False
        self._token_count_warning_keys = set()
        self._last_token_count_method = "unknown"
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
        self.tool_context_max = self.tool_context_rewriter.max_context_length
        self.tool_context_target = self.tool_context_rewriter.target_context_length
        self._last_context_fold_stats: Optional[Dict] = None

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
        if self.context_management_strategy not in {"fold_tool", "fold_then_discard"}:
            self._last_context_fold_stats = None
            return messages
        processed_messages, fold_stats = self.tool_context_rewriter.process_with_stats(messages)
        self._last_context_fold_stats = fold_stats
        return processed_messages

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

    def _warn_token_count_once(self, key: str, message: str) -> None:
        if key in self._token_count_warning_keys:
            return
        self._token_count_warning_keys.add(key)
        print(f"Warning: {message}", flush=True)

    def _is_deepseek_v4_model(self) -> bool:
        model_id = " ".join(
            str(value)
            for value in (
                self.llm_local_path,
                getattr(self, "model", ""),
            )
            if value
        ).lower()
        return self.is_deepseek_model and "v4" in model_id

    def _load_module_from_path(self, module_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _get_vllm_dsv4_encoding(self):
        if self._vllm_dsv4_encoding_initialized:
            return self._vllm_dsv4_encoding

        self._vllm_dsv4_encoding_initialized = True
        candidates = []
        for path_item in sys.path:
            if not path_item:
                continue
            candidates.append(
                Path(path_item) / "vllm/tokenizers/deepseek_v4_encoding.py"
            )

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                self._vllm_dsv4_encoding = self._load_module_from_path(
                    "_webexplorer_vllm_deepseek_v4_encoding",
                    candidate,
                )
                if self._vllm_dsv4_encoding is not None:
                    return self._vllm_dsv4_encoding
            except Exception as exc:
                self._warn_token_count_once(
                    "vllm_dsv4_encoding_load",
                    f"failed to load vLLM DeepSeek-V4 encoder from {candidate}: {exc}",
                )

        return None

    def _get_sglang_dsv4_encoding(self):
        if self._sglang_dsv4_encoding_initialized:
            return self._sglang_dsv4_encoding

        self._sglang_dsv4_encoding_initialized = True
        repo_root = Path(__file__).resolve().parents[3]
        candidates = []
        for path_item in sys.path:
            if not path_item:
                continue
            candidates.append(
                Path(path_item) / "sglang/srt/entrypoints/openai/encoding_dsv4.py"
            )
        candidates.append(
            repo_root
            / "sglang/python/sglang/srt/entrypoints/openai/encoding_dsv4.py"
        )

        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                self._sglang_dsv4_encoding = self._load_module_from_path(
                    "_webexplorer_sglang_deepseek_v4_encoding",
                    candidate,
                )
                if self._sglang_dsv4_encoding is not None:
                    return self._sglang_dsv4_encoding
            except Exception as exc:
                self._warn_token_count_once(
                    "sglang_dsv4_encoding_load",
                    f"failed to load SGLang DeepSeek-V4 encoder from {candidate}: {exc}",
                )

        return None

    def _deepseek_thinking_mode_for_template(self) -> str:
        return "thinking" if self._deepseek_reasoning_effort() is not None else "chat"

    def _prepare_messages_for_dsv4_count(
        self,
        backend: str,
        messages: List[Dict],
        include_tools: bool,
    ) -> List[Dict]:
        prepared_messages = self._prepare_messages_for_api(messages)

        if backend == "vllm":
            if include_tools and self.tool_schemas:
                prepared_messages.insert(
                    0,
                    {
                        "role": "system",
                        "tools": copy.deepcopy(self.tool_schemas),
                    },
                )
            return prepared_messages

        if not prepared_messages or prepared_messages[0].get("role") != "system":
            prepared_messages.insert(0, {"role": "system", "content": ""})
        if include_tools and self.tool_schemas:
            prepared_messages[0]["tools"] = copy.deepcopy(self.tool_schemas)
        return prepared_messages

    def _count_tokens_with_backend_template(
        self,
        messages: List[Dict],
        include_tools: bool,
    ) -> Optional[tuple[int, str]]:
        if not self._is_deepseek_v4_model():
            return None
        if self.model_server_backend not in {"vllm", "sglang"}:
            return None

        encoding_module = (
            self._get_vllm_dsv4_encoding()
            if self.model_server_backend == "vllm"
            else self._get_sglang_dsv4_encoding()
        )
        tokenizer = self._get_tokenizer()
        if encoding_module is None or tokenizer is None:
            return None

        try:
            prompt = encoding_module.encode_messages(
                self._prepare_messages_for_dsv4_count(
                    self.model_server_backend,
                    messages,
                    include_tools,
                ),
                thinking_mode=self._deepseek_thinking_mode_for_template(),
                reasoning_effort=self._deepseek_reasoning_effort(),
            )
            if self.model_server_backend == "vllm":
                token_ids = tokenizer.encode(prompt, add_special_tokens=False)
            else:
                token_ids = tokenizer.encode(prompt)
            return len(token_ids), f"{self.model_server_backend}_dsv4_encoding"
        except Exception as exc:
            self._warn_token_count_once(
                f"{self.model_server_backend}_dsv4_count",
                (
                    f"{self.model_server_backend} DeepSeek-V4 token count failed; "
                    f"falling back to generic tokenizer path: {exc}"
                ),
            )
            return None

    def sanity_check_output(self, content):
        return "<think>" in content and "</think>" in content

    def _normalize_tool_call(self, tool_call) -> Dict:
        if hasattr(tool_call, "model_dump"):
            tool_call = tool_call.model_dump(exclude_none=True)
        return copy.deepcopy(tool_call)

    def _normalize_reasoning_payload(self, reasoning_payload) -> str:
        if reasoning_payload in (None, ""):
            return ""
        if isinstance(reasoning_payload, str):
            return reasoning_payload
        if isinstance(reasoning_payload, dict):
            for key in (
                "reasoning_content",
                "reasoning",
                "thinking",
                "content",
                "text",
                "summary",
            ):
                value = reasoning_payload.get(key)
                if value not in (None, ""):
                    return self._normalize_reasoning_payload(value)
            return json.dumps(reasoning_payload, ensure_ascii=False)
        if isinstance(reasoning_payload, list):
            parts = [
                self._normalize_reasoning_payload(item)
                for item in reasoning_payload
            ]
            return "\n".join(part for part in parts if part)
        return str(reasoning_payload)

    def _get_message_payload_field(self, message, field: str):
        value = getattr(message, field, None)
        if value not in (None, ""):
            return value
        if isinstance(message, dict):
            return message.get(field)
        if hasattr(message, "model_dump"):
            payload = message.model_dump(exclude_none=True)
            return payload.get(field)
        return None

    def _extract_reasoning_content(self, message) -> str:
        reasoning_payload = self._get_message_payload_field(message, "reasoning_content")
        if reasoning_payload in (None, ""):
            reasoning_payload = self._get_message_payload_field(message, "reasoning")
        return self._normalize_reasoning_payload(reasoning_payload)

    def _normalize_assistant_message(self, message) -> Dict:
        assistant_message = {
            "role": "assistant",
            "content": message.content or "",
        }

        reasoning_content = self._extract_reasoning_content(message)
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content

        if getattr(message, "tool_calls", None):
            assistant_message["tool_calls"] = []
            for tool_call in message.tool_calls:
                normalized_tool_call = self._normalize_tool_call(tool_call)
                assistant_message["tool_calls"].append(normalized_tool_call)

        return assistant_message

    def _looks_like_raw_tool_call(self, content: Optional[str]) -> bool:
        if not content:
            return False
        stripped = strip_think_blocks(content).lstrip()
        if not stripped:
            return False
        raw_markers = (
            "<｜DSML｜tool_calls",
            "<tool_calls",
            "<｜DSML｜invoke",
            "<invoke",
        )
        if stripped.startswith(raw_markers) or any(
            marker in stripped for marker in raw_markers[:2]
        ):
            return True

        for tool_name in self.tool_map:
            if re.search(
                rf"<{re.escape(tool_name)}\b[^>]*>.*?</{re.escape(tool_name)}>",
                stripped,
                flags=re.DOTALL,
            ):
                return True
        return bool(
            re.search(
                r"<function>\s*([A-Za-z_][\w]*)\s*,\s*.*?</function>",
                stripped,
                flags=re.DOTALL,
            )
        )

    def _schema_required_params(self, tool_name: str) -> set:
        tool = self.tool_map.get(tool_name)
        if tool is None:
            return set()
        parameters = getattr(tool, "parameters", {}) or {}
        required = parameters.get("required") or []
        return {str(param) for param in required}

    def _decode_dsml_parameter_value(self, raw_value: str, is_string: str):
        if is_string == "true":
            return raw_value
        value = raw_value.strip()
        try:
            return json.loads(value)
        except Exception:
            if json5 is None:
                raise
            return json5.loads(value)

    def _parse_raw_dsml_tool_calls(self, content: str) -> List[Dict]:
        raw_content = strip_think_blocks(content).strip()
        raw_markers = (
            "<｜DSML｜tool_calls",
            "<tool_calls",
            "<｜DSML｜invoke",
            "<invoke",
        )
        if not (
            raw_content.lstrip().startswith(raw_markers)
            or any(marker in raw_content for marker in raw_markers[:2])
        ):
            raise ValueError("content does not look like a DSML tool-call block")

        invoke_pattern = re.compile(
            r"<(?:｜DSML｜)?invoke\s+name=\"([^\"]+)\"\s*>\s*"
            r"(.*?)"
            r"\s*</(?:｜DSML｜)?invoke\s*>",
            re.DOTALL,
        )
        parameter_pattern = re.compile(
            r"<(?:｜DSML｜)?parameter\s+name=\"([^\"]+)\"\s+"
            r"string=\"(true|false)\"\s*>"
            r"(.*?)"
            r"</(?:｜DSML｜)?parameter\s*>",
            re.DOTALL,
        )

        tool_calls: List[Dict] = []
        for idx, invoke_match in enumerate(invoke_pattern.finditer(raw_content)):
            tool_name = invoke_match.group(1).strip()
            if tool_name not in self.tool_map:
                raise ValueError(f"unknown tool name in DSML block: {tool_name}")

            body = invoke_match.group(2)
            param_matches = list(parameter_pattern.finditer(body))
            if not param_matches:
                raise ValueError(f"no DSML parameters found for tool {tool_name}")

            arguments = {}
            for param_match in param_matches:
                param_name = param_match.group(1).strip()
                is_string = param_match.group(2).strip().lower()
                raw_value = param_match.group(3)
                arguments[param_name] = self._decode_dsml_parameter_value(
                    raw_value, is_string
                )

            missing = self._schema_required_params(tool_name) - set(arguments)
            if missing:
                raise ValueError(
                    f"missing required parameters for tool {tool_name}: "
                    f"{sorted(missing)}"
                )

            tool_calls.append(
                {
                    "id": f"call_repaired_{int(time.time() * 1000)}_{idx}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )

        if not tool_calls:
            raise ValueError("no DSML invoke blocks found")

        return tool_calls

    def _decode_legacy_parameter_value(self, raw_value: str, schema: Dict):
        value = raw_value.strip()
        if not value:
            return value

        try:
            return json.loads(value)
        except Exception:
            if json5 is not None:
                try:
                    return json5.loads(value)
                except Exception:
                    pass

        if schema.get("type") == "array":
            return [value]
        return value

    def _legacy_parameter_tag_names(self, param_name: str) -> List[str]:
        tag_names = [param_name]
        if param_name.endswith("ies") and len(param_name) > 3:
            tag_names.append(param_name[:-3] + "y")
        elif param_name.endswith("s") and len(param_name) > 1:
            tag_names.append(param_name[:-1])
        return tag_names

    def _parse_raw_legacy_xml_tool_calls(self, content: str) -> List[Dict]:
        raw_content = strip_think_blocks(content).strip()
        tool_calls: List[Dict] = []

        for tool_idx, tool_name in enumerate(self.tool_map):
            tool_pattern = re.compile(
                rf"<{re.escape(tool_name)}\b[^>]*>\s*"
                rf"(.*?)"
                rf"\s*</{re.escape(tool_name)}>",
                re.DOTALL,
            )
            for match_idx, tool_match in enumerate(tool_pattern.finditer(raw_content)):
                body = tool_match.group(1).strip()
                tool = self.tool_map[tool_name]
                parameters = getattr(tool, "parameters", {}) or {}
                properties = parameters.get("properties") or {}
                arguments = {}

                if body.startswith("{"):
                    try:
                        parsed_body = json.loads(body)
                    except Exception:
                        if json5 is None:
                            raise
                        parsed_body = json5.loads(body)
                    if not isinstance(parsed_body, dict):
                        raise ValueError(
                            f"legacy XML body for tool {tool_name} is not an object"
                        )
                    arguments.update(parsed_body)

                for param_name, param_schema in properties.items():
                    schema = param_schema if isinstance(param_schema, dict) else {}
                    for tag_name in self._legacy_parameter_tag_names(param_name):
                        param_pattern = re.compile(
                            rf"<{re.escape(tag_name)}\b[^>]*>\s*"
                            rf"(.*?)"
                            rf"\s*</{re.escape(tag_name)}>",
                            re.DOTALL,
                        )
                        param_matches = list(param_pattern.finditer(body))
                        if not param_matches:
                            continue

                        decoded_values = [
                            self._decode_legacy_parameter_value(
                                param_match.group(1),
                                schema,
                            )
                            for param_match in param_matches
                        ]
                        if schema.get("type") == "array":
                            values = []
                            for decoded_value in decoded_values:
                                if isinstance(decoded_value, list):
                                    values.extend(decoded_value)
                                else:
                                    values.append(decoded_value)
                            arguments[param_name] = values
                        else:
                            arguments[param_name] = decoded_values[0]
                        break

                for param_name, param_schema in properties.items():
                    if param_name in arguments:
                        continue
                    schema = param_schema if isinstance(param_schema, dict) else {}
                    for alias in self._legacy_parameter_tag_names(param_name)[1:]:
                        if alias not in arguments:
                            continue
                        value = arguments[alias]
                        if schema.get("type") == "array" and not isinstance(value, list):
                            value = [value]
                        arguments[param_name] = value
                        break

                required = self._schema_required_params(tool_name)
                if not arguments and len(required) == 1 and body:
                    only_param = next(iter(required))
                    param_schema = properties.get(only_param, {})
                    arguments[only_param] = self._decode_legacy_parameter_value(
                        body,
                        param_schema if isinstance(param_schema, dict) else {},
                    )

                missing = required - set(arguments)
                if missing:
                    raise ValueError(
                        f"missing required parameters for legacy XML tool {tool_name}: "
                        f"{sorted(missing)}"
                    )

                tool_calls.append(
                    {
                        "id": (
                            f"call_repaired_{int(time.time() * 1000)}_"
                            f"{tool_idx}_{match_idx}"
                        ),
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                )

        if not tool_calls:
            raise ValueError("no legacy XML tool-call blocks found")

        return tool_calls

    def _parse_raw_legacy_function_tool_calls(self, content: str) -> List[Dict]:
        raw_content = strip_think_blocks(content).strip()
        function_pattern = re.compile(
            r"<function>\s*(.*?)\s*</function>",
            re.DOTALL,
        )
        tool_calls: List[Dict] = []

        for idx, function_match in enumerate(function_pattern.finditer(raw_content)):
            body = function_match.group(1).strip()
            call_match = re.match(
                r"([A-Za-z_][\w]*)\s*,\s*(.*)\Z",
                body,
                flags=re.DOTALL,
            )
            if not call_match:
                continue

            tool_name = call_match.group(1).strip()
            if tool_name not in self.tool_map:
                continue

            raw_arguments = call_match.group(2).strip()
            try:
                arguments = json.loads(raw_arguments)
            except Exception:
                if json5 is None:
                    raise
                arguments = json5.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"legacy function arguments for tool {tool_name} are not an object"
                )

            missing = self._schema_required_params(tool_name) - set(arguments)
            if missing:
                raise ValueError(
                    f"missing required parameters for legacy function tool {tool_name}: "
                    f"{sorted(missing)}"
                )

            tool_calls.append(
                {
                    "id": f"call_repaired_{int(time.time() * 1000)}_function_{idx}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )

        if not tool_calls:
            raise ValueError("no legacy function tool-call blocks found")

        return tool_calls

    def _parse_raw_tool_calls(self, content: str) -> tuple[List[Dict], str]:
        try:
            return self._parse_raw_dsml_tool_calls(content), "raw_dsml_content"
        except ValueError as dsml_exc:
            try:
                return (
                    self._parse_raw_legacy_xml_tool_calls(content),
                    "raw_legacy_xml_content",
                )
            except ValueError as legacy_xml_exc:
                try:
                    return (
                        self._parse_raw_legacy_function_tool_calls(content),
                        "raw_legacy_function_content",
                    )
                except ValueError as legacy_function_exc:
                    raise ValueError(
                        f"{dsml_exc}; legacy XML parse failed: {legacy_xml_exc}; "
                        f"legacy function parse failed: {legacy_function_exc}"
                    ) from legacy_function_exc

    def _repair_raw_tool_call_message(
        self,
        assistant_message: Dict,
        *,
        raw_content: Optional[str],
        original_finish_reason: Optional[str],
    ) -> bool:
        if not raw_content:
            raise ValueError("tool-call finish_reason without content or tool_calls")

        tool_calls, repair_source = self._parse_raw_tool_calls(raw_content)
        assistant_message["content"] = ""
        assistant_message["tool_calls"] = tool_calls
        assistant_message["_finish_reason"] = "tool_calls"
        assistant_message["_tool_call_repair"] = {
            "repaired": True,
            "source": repair_source,
            "original_finish_reason": original_finish_reason,
            "tool_call_count": len(tool_calls),
        }
        return True

    def _strip_internal_message_fields(self, messages: List[Dict]) -> List[Dict]:
        return [
            {key: value for key, value in message.items() if not key.startswith("_")}
            for message in copy.deepcopy(messages)
        ]

    def _move_thinking_to_reasoning_content(self, message: Dict) -> None:
        reasoning_payload = message.pop("reasoning", None)
        if not message.get("reasoning_content") and reasoning_payload not in (None, ""):
            message["reasoning_content"] = self._normalize_reasoning_payload(
                reasoning_payload
            )

        thinking_payload = message.pop("thinking", None)
        if message.get("reasoning_content") or thinking_payload is None:
            return
        if isinstance(thinking_payload, dict):
            message["reasoning_content"] = thinking_payload.get("thinking") or ""
        elif isinstance(thinking_payload, str):
            message["reasoning_content"] = thinking_payload

    def _adapt_reasoning_fields_for_backend(self, message: Dict) -> None:
        if message.get("role") != "assistant":
            return
        if self.model_server_backend != "vllm":
            return

        reasoning_content = message.get("reasoning_content")
        if reasoning_content not in (None, "") and not message.get("reasoning"):
            message["reasoning"] = reasoning_content

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

    def _get_context_token_count(
        self,
        messages,
        usage: Optional[Dict] = None,
        prefer_usage: bool = True,
    ):
        if prefer_usage:
            token_count, token_count_source = self._token_count_from_usage(usage)
            if token_count is not None:
                return token_count, token_count_source, usage

        token_count = self.count_tokens(messages)
        return (
            token_count,
            f"local_count_tokens.{self._last_token_count_method}",
            {
                "prompt_tokens": token_count,
                "completion_tokens": 0,
                "total_tokens": token_count,
                "estimated": True,
            },
        )

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
            1
            for event in context_events
            if event.get("action") == "discard_all" or event.get("strategy") == "discard_all"
        )
        summary_count = sum(
            1 for event in context_events if event.get("strategy") == "summary"
        )
        return {
            "context_management_strategy": self.context_management_strategy,
            "model_server_backend": self.model_server_backend,
            "last_local_token_count_method": self._last_token_count_method,
            "context_management_count": len(context_events),
            "discard_all_count": discard_all_count,
            "summary_count": summary_count,
            "context_reset_events": context_events,
            "cumulative_token_usage": copy.deepcopy(cumulative_usage),
            "cumulative_token_usage_metric": "context_reset_prompt_tokens",
            "context_total_token_limit": self.context_total_token_limit,
            "context_summary_trigger_tokens": self.context_summary_trigger_tokens,
            "nam_max_memory_size": self.nam_max_memory_size,
            "nam_trigger_low_frac": self.nam_trigger_low_frac,
            "nam_trigger_high_frac": self.nam_trigger_high_frac,
            "nam_stage1_enabled": self.nam_stage1_enabled,
            "context_summary_tag": self.context_summary_tag,
            "tool_context_max": self.tool_context_max,
            "tool_context_target": self.tool_context_target,
            "discard_prompt_threshold_ratio": self.discard_prompt_threshold_ratio,
            "discard_history_tool_tokens": self.discard_history_tool_tokens,
            "discard_history_min_rounds": self.discard_history_min_rounds,
            "discard_history_max_rounds": self.discard_history_max_rounds,
            "max_input_tokens": int(self.llm_generate_cfg.get("max_input_tokens", 196608)),
            "forced_finalize_context_tokens": self._forced_finalize_context_tokens(),
        }

    def _get_summary_thresholds(self) -> Dict[str, int]:
        low = int(self.nam_trigger_low_frac * self.nam_max_memory_size)
        high = int(self.nam_trigger_high_frac * self.nam_max_memory_size)
        return {
            "low": low,
            "high": high,
            "max_memory_size": self.nam_max_memory_size,
        }

    def _forced_finalize_context_tokens(self) -> int:
        configured = os.getenv("FORCED_FINALIZE_CONTEXT_TOKENS")
        if configured is not None and configured.strip():
            return int(configured)
        max_input_tokens = int(self.llm_generate_cfg.get("max_input_tokens", 196608))
        return int(max_input_tokens * 0.8)

    def _max_llm_calls_per_run(self) -> int:
        configured = os.getenv("MAX_LLM_CALL_PER_RUN")
        if configured is not None:
            return int(configured)
        if self.context_management_strategy == "summary":
            return 800
        if self.is_qwen_model:
            return 200
        return MAX_LLM_CALL_PER_RUN

    def _latest_assistant_content(self, messages: List[Dict]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("content"):
                return strip_think_blocks(message["content"])
        return ""

    def _last_user_index(self, messages: List[Dict]) -> Optional[int]:
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                return idx
        return None

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

    def _build_stage1_request_messages(
        self,
        messages: List[Dict],
        token_count: int,
        thresholds: Dict[str, int],
    ) -> List[Dict]:
        transcript = self._format_conversation_history_for_summary(messages)
        user_prompt = (
            f"Current token count: {token_count}\n"
            f"Low trigger threshold: {thresholds['low']}\n"
            f"High pressure threshold: {thresholds['high']}\n"
            f"Maximum active memory size: {thresholds['max_memory_size']}\n"
            f"Message count: {len(messages)}\n\n"
            "<conversation_history>\n"
            f"{transcript}\n"
            "</conversation_history>\n\n"
            "Decide whether to summarize this context before the next assistant "
            "response. Output exactly YES or NO after any thinking."
        )
        return [
            {"role": "system", "content": NAM_STAGE1_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _parse_stage1_summary_decision(self, content: Optional[str]) -> bool:
        decision = strip_think_blocks(content or "").strip().upper()
        return bool(re.search(r"\bYES\b", decision))

    def _should_summarize_with_stage1(
        self,
        messages: List[Dict],
        token_count: int,
        thresholds: Dict[str, int],
        planning_port: int,
        request_log_callback: Optional[Callable[[Dict], None]] = None,
    ) -> tuple[bool, Dict]:
        stage1_messages = self._build_stage1_request_messages(
            messages,
            token_count,
            thresholds,
        )
        stage1_response = self.call_server(
            stage1_messages,
            planning_port,
            use_tools=False,
            request_log_callback=request_log_callback,
        )
        decision = self._parse_stage1_summary_decision(
            stage1_response.get("content")
        )
        stage1_usage = self._get_call_usage(stage1_messages, stage1_response)
        if stage1_usage and not stage1_response.get("_usage"):
            stage1_response["_usage"] = copy.deepcopy(stage1_usage)
        return decision, {
            "stage1_decision": "YES" if decision else "NO",
            "stage1_response": strip_think_blocks(stage1_response.get("content") or ""),
            "stage1_usage": stage1_usage,
        }

    def _generate_summary_message(
        self,
        summary_request_messages: List[Dict],
        planning_port: int,
        max_attempts: int = 3,
        request_log_callback: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        summary_message = {}
        for attempt in range(max_attempts):
            summary_message = self.call_server(
                summary_request_messages,
                planning_port,
                use_tools=False,
                max_tokens_cap=CONTEXT_SUMMARY_MAX_TOKENS_CAP,
                request_log_callback=request_log_callback,
            )
            if strip_think_blocks(summary_message.get("content") or ""):
                if attempt:
                    summary_message["_summary_retry_count"] = attempt
                return summary_message

        raise ValueError("summary failed")

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

    def _build_messages_after_discard(
        self, messages: List[Dict], system_prompt: str, question: str
    ) -> List[Dict]:
        system_messages = []
        if messages and messages[0].get("role") == "system":
            system_messages.append(copy.deepcopy(messages[0]))
        elif system_prompt:
            system_messages.append({"role": "system", "content": system_prompt})

        return system_messages + [self._last_user_message(messages, question)]

    def _build_discard_all_messages(
        self, messages: List[Dict], system_prompt: str, question: str
    ) -> List[Dict]:
        if self.discard_all_keep_system_prompt:
            return self._build_messages_after_discard(messages, system_prompt, question)
        return [{"role": "user", "content": question}]

    def _raw_tool_tokens_in_messages(self, messages: List[Dict]) -> int:
        total = 0
        for message in messages:
            if message.get("role") != "tool":
                continue
            total += self._count_text_tokens(message.get("content") or "")
        return total

    def _assistant_rounds_in_messages(self, messages: List[Dict]) -> int:
        return sum(1 for message in messages if message.get("role") == "assistant")

    def _format_context_summary(self, summary_text: str) -> str:
        formatted_summary = (
            f"<{self.context_summary_tag}>\n"
            f"{summary_text}\n"
            f"</{self.context_summary_tag}>\n\n"
        )
        return formatted_summary

    def _prepend_summary_to_thinking(self, thinking_content: str, summary_text: str) -> str:
        return f"{self._format_context_summary(summary_text)}{thinking_content}"

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
                thinking_payload["thinking"] = self._prepend_summary_to_thinking(
                    thinking_content,
                    pending_summary,
                )
                return True
            if isinstance(thinking_payload, str):
                message["thinking"] = self._prepend_summary_to_thinking(
                    strip_think_blocks(thinking_payload),
                    pending_summary,
                )
                return True

            if message.get("reasoning_content"):
                reasoning_content = strip_think_blocks(message["reasoning_content"])
                message["reasoning_content"] = self._prepend_summary_to_thinking(
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
                    + self._prepend_summary_to_thinking(
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
                + self._prepend_summary_to_thinking("", pending_summary)
                + "</think>\n"
                + content
            )
            return True

        return False

    def _prepare_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        api_messages = self._strip_internal_message_fields(messages)

        for message in api_messages:
            self._move_thinking_to_reasoning_content(message)
            self._adapt_reasoning_fields_for_backend(message)

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

    def _with_context_awareness(
        self,
        messages: List[Dict],
        cumulative_usage: Dict,
    ) -> List[Dict]:
        request_messages = copy.deepcopy(messages)
        if self.context_management_strategy != "summary":
            return request_messages

        consumed = int(cumulative_usage.get("context_reset_prompt_tokens", 0))
        remaining = max(self.context_total_token_limit - consumed, 0)
        awareness = (
            "\n\n<context_awareness>\n"
            f"Remaining total context budget for this attempt: {remaining} tokens.\n"
            f"Consumed context budget from summarized/reset history: {consumed} tokens.\n"
            f"Total context budget for this attempt: {self.context_total_token_limit} tokens.\n"
            "Use available tools efficiently. Older context may be summarized "
            "when memory pressure is high.\n"
            "</context_awareness>"
        )

        if request_messages and request_messages[0].get("role") == "system":
            request_messages[0]["content"] = (
                (request_messages[0].get("content") or "") + awareness
            )
            return request_messages

        return [{"role": "system", "content": awareness.strip()}] + request_messages

    def _prepare_messages_for_template(self, messages: List[Dict]) -> List[Dict]:
        template_messages = self._strip_internal_message_fields(messages)

        for message in template_messages:
            self._move_thinking_to_reasoning_content(message)
            self._adapt_reasoning_fields_for_backend(message)

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
            started_at = time.time()

            try:
                tool_args = self._parse_tool_arguments(raw_arguments)
                result = self.custom_call_tool(tool_name, tool_args)
                status = "success"
                error_type = None
            except Exception:
                result = (
                    'Error: Tool call arguments are not valid JSON. Tool call must '
                    'contain a valid "name" and "arguments" field.'
                )
                status = "error"
                error_type = "invalid_tool_arguments"

            elapsed_s = time.time() - started_at

            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                    "_timing": {
                        "tool_name": tool_name,
                        "status": status,
                        "error_type": error_type,
                        "elapsed_s": round(elapsed_s, 4),
                        "started_at": started_at,
                        "finished_at": time.time(),
                    },
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

    def _configured_sampling_params(self) -> Dict:
        sampling_params = {
            "model": self.model,
            "model_server_backend": self.model_server_backend,
            "max_input_tokens": int(
                self.llm_generate_cfg.get("max_input_tokens", 196608)
            ),
            "max_retries": int(self.llm_generate_cfg.get("max_retries", 10)),
            "temperature": self.llm_generate_cfg.get("temperature", 0.6),
            "top_p": self.llm_generate_cfg.get("top_p", 0.95),
            "logprobs": False,
        }
        presence_penalty = self.llm_generate_cfg.get("presence_penalty")
        if presence_penalty is not None:
            sampling_params["presence_penalty"] = presence_penalty

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
        chat_template_kwargs = self._chat_template_kwargs()
        if chat_template_kwargs:
            extra_body["chat_template_kwargs"] = chat_template_kwargs
        if extra_body:
            sampling_params["extra_body"] = extra_body

        return sampling_params

    def _build_sampling_request_info(
        self,
        request_kwargs: Dict,
        *,
        attempt: int,
        max_tries: int,
        use_tools: bool,
        prompt_tokens: int,
        max_input_tokens: int,
        remaining_tokens: int,
        max_tokens_cap: Optional[int],
    ) -> Dict:
        request_info = {
            "attempt": attempt,
            "max_tries": max_tries,
            "model": request_kwargs.get("model", self.model),
            "model_server_backend": self.model_server_backend,
            "temperature": request_kwargs.get("temperature"),
            "top_p": request_kwargs.get("top_p"),
            "logprobs": request_kwargs.get("logprobs"),
            "max_tokens": request_kwargs.get("max_tokens"),
            "use_tools": use_tools,
            "tool_count": len(self.tool_schemas) if use_tools else 0,
            "prompt_tokens": prompt_tokens,
            "max_input_tokens": max_input_tokens,
            "remaining_context_tokens": remaining_tokens,
            "max_tokens_cap": max_tokens_cap,
            "safety_margin": MAX_TOKENS_SAFETY_MARGIN,
        }
        if "presence_penalty" in request_kwargs:
            request_info["presence_penalty"] = request_kwargs["presence_penalty"]
        if "extra_body" in request_kwargs:
            request_info["extra_body"] = copy.deepcopy(request_kwargs["extra_body"])
        return request_info

    def call_server(
        self,
        msgs,
        planning_port,
        use_tools=True,
        max_tries=10,
        max_tokens_cap=None,
        request_log_callback: Optional[Callable[[Dict], None]] = None,
    ):
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
        if max_tokens_cap is None and self.is_deepseek_model:
            max_tokens_cap = self.deepseek_max_output_tokens_cap
        if max_tokens_cap is not None:
            dynamic_max_tokens = min(dynamic_max_tokens, max_tokens_cap)
        print(
            "dynamic max_tokens: "
            f"{dynamic_max_tokens} "
            f"(remaining_context_tokens={remaining_tokens}, "
            f"max_input_tokens={max_input_tokens}, "
            f"prompt_tokens={prompt_tokens}, "
            f"safety_margin={MAX_TOKENS_SAFETY_MARGIN}, "
            f"max_tokens_cap={max_tokens_cap})",
            flush=True,
        )
        if dynamic_max_tokens <= 0:
            if request_log_callback is not None:
                request_log_callback(
                    {
                        **self._configured_sampling_params(),
                        "attempt": 0,
                        "max_tries": max_tries,
                        "use_tools": use_tools,
                        "tool_count": len(self.tool_schemas) if use_tools else 0,
                        "status": "input_token_limit_reached",
                        "prompt_tokens": prompt_tokens,
                        "remaining_context_tokens": remaining_tokens,
                        "max_tokens": dynamic_max_tokens,
                        "max_tokens_cap": max_tokens_cap,
                        "safety_margin": MAX_TOKENS_SAFETY_MARGIN,
                    }
                )
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

        repair_retry_limit = max(0, TOOL_CALL_REPAIR_MAX_RESAMPLES)
        repair_failures = 0
        raw_tool_call_resample_limit = max(0, RAW_TOOL_CALL_RESAMPLE_MAX_ATTEMPTS)
        raw_tool_call_resamples = 0

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
                    "logprobs": False,
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
                chat_template_kwargs = self._chat_template_kwargs()
                if chat_template_kwargs:
                    extra_body["chat_template_kwargs"] = chat_template_kwargs
                if extra_body:
                    request_kwargs["extra_body"] = extra_body
                if use_tools:
                    request_kwargs["tools"] = self.tool_schemas

                request_info = self._build_sampling_request_info(
                    request_kwargs,
                    attempt=attempt + 1,
                    max_tries=max_tries,
                    use_tools=use_tools,
                    prompt_tokens=prompt_tokens,
                    max_input_tokens=max_input_tokens,
                    remaining_tokens=remaining_tokens,
                    max_tokens_cap=max_tokens_cap,
                )
                request_started_at = time.time()
                chat_response = client.chat.completions.create(**request_kwargs)
                request_elapsed_s = time.time() - request_started_at
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
                    needs_tool_call_repair = (
                        use_tools
                        and not has_tool_calls
                        and (
                            finish_reason == "tool_calls"
                            or self._looks_like_raw_tool_call(content)
                        )
                    )
                    if needs_tool_call_repair:
                        if raw_tool_call_resamples < raw_tool_call_resample_limit:
                            raw_tool_call_resamples += 1
                            request_info["status"] = "raw_tool_call_resample"
                            request_info["elapsed_s"] = round(request_elapsed_s, 4)
                            request_info["started_at"] = request_started_at
                            request_info["finished_at"] = time.time()
                            request_info["finish_reason"] = finish_reason
                            request_info["has_tool_calls"] = False
                            request_info["raw_tool_call_resamples"] = (
                                raw_tool_call_resamples
                            )
                            request_info["raw_tool_call_resample_limit"] = (
                                raw_tool_call_resample_limit
                            )
                            if usage:
                                request_info["usage"] = copy.deepcopy(usage)
                            if request_log_callback is not None:
                                request_log_callback(request_info)
                            print(
                                "Warning: raw tool-call text was returned without "
                                "structured tool_calls; resampling "
                                f"{raw_tool_call_resamples}/"
                                f"{raw_tool_call_resample_limit} before repair.",
                                flush=True,
                            )
                            time.sleep(min(0.5 * raw_tool_call_resamples, 2.0))
                            continue

                        try:
                            self._repair_raw_tool_call_message(
                                assistant_message,
                                raw_content=content,
                                original_finish_reason=finish_reason,
                            )
                            has_tool_calls = True
                            request_info["tool_call_repaired"] = True
                            request_info["original_finish_reason"] = finish_reason
                            finish_reason = "tool_calls"
                            print(
                                "Repaired raw tool call from response content.",
                                flush=True,
                            )
                        except Exception as repair_exc:
                            repair_failures += 1
                            request_info["status"] = "tool_call_parse_mismatch"
                            request_info["elapsed_s"] = round(request_elapsed_s, 4)
                            request_info["started_at"] = request_started_at
                            request_info["finished_at"] = time.time()
                            request_info["finish_reason"] = finish_reason
                            request_info["has_tool_calls"] = False
                            request_info["tool_call_repair_error"] = str(repair_exc)
                            request_info["tool_call_repair_failures"] = repair_failures
                            if usage:
                                request_info["usage"] = copy.deepcopy(usage)
                            if request_log_callback is not None:
                                request_log_callback(request_info)

                            if repair_failures < repair_retry_limit:
                                sleep_time = min(1 + repair_failures, 3)
                                print(
                                    "Warning: raw tool-call response could not be "
                                    f"repaired ({repair_exc}); resampling "
                                    f"{repair_failures}/{repair_retry_limit} "
                                    f"after {sleep_time:.2f}s.",
                                    flush=True,
                                )
                                time.sleep(sleep_time)
                                continue

                            raise ToolCallParseError(
                                "raw tool-call response could not be repaired after "
                                f"{repair_failures} attempts: {repair_exc}"
                            ) from repair_exc

                    if usage:
                        assistant_message["_usage"] = usage
                    if finish_reason:
                        assistant_message["_finish_reason"] = finish_reason
                    request_info["status"] = "success"
                    request_info["elapsed_s"] = round(request_elapsed_s, 4)
                    request_info["started_at"] = request_started_at
                    request_info["finished_at"] = time.time()
                    request_info["finish_reason"] = finish_reason
                    request_info["has_tool_calls"] = has_tool_calls
                    if usage:
                        request_info["usage"] = copy.deepcopy(usage)
                    if request_log_callback is not None:
                        request_log_callback(request_info)
                    return assistant_message

                request_info["status"] = "empty_response"
                request_info["elapsed_s"] = round(request_elapsed_s, 4)
                request_info["started_at"] = request_started_at
                request_info["finished_at"] = time.time()
                if request_log_callback is not None:
                    request_log_callback(request_info)
                print(f"Warning: Attempt {attempt + 1} received an empty response.")
            except ToolCallParseError:
                raise
            except (APIError, APIConnectionError, APITimeoutError) as e:
                if request_log_callback is not None:
                    request_log_callback(
                        {
                            **locals().get("request_info", {}),
                            "attempt": attempt + 1,
                            "max_tries": max_tries,
                            "model": self.model,
                            "use_tools": use_tools,
                            "tool_count": len(self.tool_schemas) if use_tools else 0,
                            "status": "api_error",
                            "error_type": type(e).__name__,
                            "error": str(e),
                        }
                    )
                print(f"Error: Attempt {attempt + 1} failed with an API or network error: {e}")
            except Exception as e:
                if request_log_callback is not None:
                    request_log_callback(
                        {
                            **locals().get("request_info", {}),
                            "attempt": attempt + 1,
                            "max_tries": max_tries,
                            "model": self.model,
                            "use_tools": use_tools,
                            "tool_count": len(self.tool_schemas) if use_tools else 0,
                            "status": "error",
                            "error_type": type(e).__name__,
                            "error": str(e),
                        }
                    )
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")

        if env_flag("WEBEXPLORER_FAIL_FAST_ON_VLLM_ERROR", False):
            raise VllmServerError(
                f"vLLM server unavailable after {max_tries} attempts on port {planning_port}"
            )
        return {"role": "assistant", "content": VLLM_SERVER_ERROR_MESSAGE}

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
        backend_count = self._count_tokens_with_backend_template(
            messages,
            include_tools=include_tools,
        )
        if backend_count is not None:
            token_count, method = backend_count
            self._last_token_count_method = method
            return token_count

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
                        self._last_token_count_method = "hf_apply_chat_template"
                        return int(input_ids.shape[-1])
                    try:
                        first_item = input_ids[0]
                    except (IndexError, TypeError):
                        self._last_token_count_method = "hf_apply_chat_template"
                        return len(input_ids)
                    if isinstance(first_item, (list, tuple)):
                        self._last_token_count_method = "hf_apply_chat_template"
                        return len(first_item)
                    if hasattr(first_item, "shape"):
                        self._last_token_count_method = "hf_apply_chat_template"
                        return int(first_item.shape[-1])
                    self._last_token_count_method = "hf_apply_chat_template"
                    return len(input_ids)
                self._last_token_count_method = "hf_apply_chat_template"
                return len(tokenized)
            except Exception as e:
                print(f"Warning: tokenizer.apply_chat_template failed, fallback to tiktoken: {e}")

        token_payload = {
            "messages": self._prepare_messages_for_api(messages),
        }
        if include_tools:
            token_payload["tools"] = self.tool_schemas

        encoding = tiktoken.encoding_for_model(model)
        self._last_token_count_method = "tiktoken_json_payload"
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

    def _context_token_thresholds(self) -> tuple[Optional[int], Optional[int], Optional[Dict]]:
        if self.context_management_strategy == "discard_all":
            max_input_tokens = int(self.llm_generate_cfg.get("max_input_tokens", 320000))
            reset_threshold_tokens = int(max_input_tokens * self.context_reset_threshold)
            return max_input_tokens, reset_threshold_tokens, None

        if self.context_management_strategy == "summary":
            thresholds = self._get_summary_thresholds()
            return None, thresholds["low"], thresholds

        if self.context_management_strategy == "fold_then_discard":
            max_input_tokens = int(self.llm_generate_cfg.get("max_input_tokens", 196608))
            reset_threshold_tokens = int(
                max_input_tokens * self.discard_prompt_threshold_ratio
            )
            return max_input_tokens, reset_threshold_tokens, None

        return None, None, None

    def maybe_reset_context(
        self,
        messages,
        question,
        usage: Optional[Dict] = None,
        planning_port: Optional[int] = None,
        system_prompt: Optional[str] = None,
        request_log_callback: Optional[Callable[[Dict], None]] = None,
    ):
        if self.context_management_strategy not in {
            "discard_all",
            "summary",
            "fold_then_discard",
        }:
            return messages, None, None

        max_input_tokens, reset_threshold_tokens, thresholds = self._context_token_thresholds()
        evaluation_messages = messages
        fold_stats = None
        if self.context_management_strategy == "fold_then_discard":
            evaluation_messages, fold_stats = self.tool_context_rewriter.process_with_stats(
                messages
            )

        token_count, token_count_source, token_usage = self._get_context_token_count(
            evaluation_messages,
            usage=None,
            prefer_usage=False,
        )
        reset_info = {
            "strategy": self.context_management_strategy,
            "token_count": token_count,
            "token_count_source": token_count_source,
            "threshold": reset_threshold_tokens,
            "max_input_tokens": max_input_tokens,
        }
        history_tool_tokens = self._raw_tool_tokens_in_messages(messages)
        rounds_since_reset = self._assistant_rounds_in_messages(messages)
        reset_info.update(
            {
                "history_tool_tokens": history_tool_tokens,
                "rounds_since_reset": rounds_since_reset,
            }
        )
        if self.context_management_strategy == "discard_all":
            reset_info["threshold_ratio"] = self.context_reset_threshold
            reset_info["keep_system_prompt"] = self.discard_all_keep_system_prompt
        elif self.context_management_strategy == "fold_then_discard":
            reset_info["threshold_ratio"] = self.discard_prompt_threshold_ratio
        else:
            reset_info.update(
                {
                    "threshold_low": thresholds["low"],
                    "threshold_high": thresholds["high"],
                    "max_memory_size": thresholds["max_memory_size"],
                    "threshold_low_frac": self.nam_trigger_low_frac,
                    "threshold_high_frac": self.nam_trigger_high_frac,
                    "stage1_enabled": self.nam_stage1_enabled,
                }
            )
        if token_usage:
            reset_info["usage"] = copy.deepcopy(token_usage)
        if fold_stats:
            reset_info["fold_stats"] = copy.deepcopy(fold_stats)

        print(
            f"context management: strategy={self.context_management_strategy}, "
            f"token_count={token_count}, source={token_count_source}, "
            f"reset_threshold={reset_threshold_tokens}",
            flush=True,
        )

        if self.context_management_strategy in {"discard_all", "fold_then_discard"}:
            if (
                self.discard_history_max_rounds > 0
                and rounds_since_reset >= self.discard_history_max_rounds
            ):
                action = "discard_all"
                reset_info["trigger"] = "history_max_rounds"
                print(
                    "context management: action=discard_all because "
                    f"rounds_since_reset={rounds_since_reset} >= "
                    f"discard_history_max_rounds={self.discard_history_max_rounds}",
                    flush=True,
                )
                if self.context_management_strategy == "fold_then_discard":
                    return (
                        self._build_messages_after_discard(
                            messages, system_prompt or "", question
                        ),
                        action,
                        reset_info,
                    )
                return (
                    self._build_discard_all_messages(messages, system_prompt or "", question),
                    action,
                    reset_info,
                )

            if (
                self.discard_history_tool_tokens > 0
                and history_tool_tokens >= self.discard_history_tool_tokens
                and (
                    self.discard_history_min_rounds <= 0
                    or rounds_since_reset >= self.discard_history_min_rounds
                )
            ):
                action = "discard_all"
                reset_info["trigger"] = "history_tool_tokens"
                print(
                    "context management: action=discard_all because "
                    f"history_tool_tokens={history_tool_tokens} >= "
                    f"discard_history_tool_tokens={self.discard_history_tool_tokens} "
                    f"and rounds_since_reset={rounds_since_reset} >= "
                    f"discard_history_min_rounds={self.discard_history_min_rounds}",
                    flush=True,
                )
                if self.context_management_strategy == "fold_then_discard":
                    return (
                        self._build_messages_after_discard(
                            messages, system_prompt or "", question
                        ),
                        action,
                        reset_info,
                    )
                return (
                    self._build_discard_all_messages(messages, system_prompt or "", question),
                    action,
                    reset_info,
                )

        if self.context_management_strategy == "summary":
            if not self.nam_stage1_enabled:
                if token_count >= thresholds["max_memory_size"]:
                    reset_info["summary_trigger"] = "forced"
                    print(
                        "context management: action=summary because "
                        f"{token_count} >= {thresholds['max_memory_size']}",
                        flush=True,
                    )
                    return messages, "summary", reset_info

                print(
                    "context management: action=none because NAM stage1 is "
                    "disabled and active memory is below max",
                    flush=True,
                )
                return messages, None, reset_info

            if token_count >= thresholds["max_memory_size"]:
                reset_info["summary_trigger"] = "forced"
                print(
                    "context management: action=summary because "
                    f"{token_count} >= {thresholds['max_memory_size']}",
                    flush=True,
                )
                return messages, "summary", reset_info

            if planning_port is None:
                raise ValueError("planning_port is required for NAM stage1 summary checks")

            should_summarize, stage1_info = self._should_summarize_with_stage1(
                messages,
                token_count,
                thresholds,
                planning_port,
                request_log_callback=request_log_callback,
            )
            reset_info.update(stage1_info)
            reset_info["summary_trigger"] = "stage1"
            if should_summarize:
                print(
                    "context management: action=summary because NAM stage1 "
                    "returned YES",
                    flush=True,
                )
                return messages, "summary", reset_info

            print(
                "context management: action=none because NAM stage1 returned NO",
                flush=True,
            )
            return messages, None, reset_info

        if token_count > reset_threshold_tokens:
            action = "discard_all"
            reset_info["trigger"] = "prompt_token_threshold"
            print(
                f"context management: action={action} because "
                f"{token_count} > {reset_threshold_tokens}",
                flush=True,
            )
            if self.context_management_strategy == "fold_then_discard":
                return (
                    self._build_messages_after_discard(messages, system_prompt or "", question),
                    action,
                    reset_info,
                )
            return (
                self._build_discard_all_messages(messages, system_prompt or "", question),
                action,
                reset_info,
            )

        return messages, None, reset_info

    def _system_prompt_for_run(self, model: str) -> str:
        if "minimax-m2.5" in model.lower():
            return SYSTEM_PROMPT
        if "minimax-m2.1" in model.lower():
            return MINIMAX_21_SYSTEM_PROMPT
        return SYSTEM_PROMPT

    def _deepseek_reasoning_effort(self) -> Optional[str]:
        thinking_mode = os.getenv("DEEPSEEK_THINKING_MODE", "think").strip().lower()
        if thinking_mode in {"think", "think_high", "think-high", "high"}:
            return "high"
        if thinking_mode in {"think_max", "think-max", "max"}:
            return "max"
        return None

    def _chat_template_kwargs(self) -> Optional[Dict[str, Union[bool, str]]]:
        if "deepseek" not in self.model.lower():
            return None

        reasoning_effort = self._deepseek_reasoning_effort()
        if reasoning_effort is None:
            return None

        return {
            "thinking": True,
            "reasoning_effort": reasoning_effort,
        }

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
        system_prompt = self._system_prompt_for_run(model)
        progress_callback = kwargs.get("progress_callback")
        task_metadata = kwargs.get("task_metadata", {})
        resume_state = kwargs.get("resume_state") or {}
        sampling_params = self._configured_sampling_params()
        sampling_request_history = copy.deepcopy(
            resume_state.get("sampling_request_history") or []
        )

        def record_sampling_request(request_info: Dict) -> None:
            sampling_request_history.append(copy.deepcopy(request_info))

        def make_sampling_state_payload() -> Dict:
            payload = {
                "sampling_params": copy.deepcopy(sampling_params),
                "sampling_request_history": copy.deepcopy(sampling_request_history),
            }
            if sampling_request_history:
                payload["latest_sampling_request"] = copy.deepcopy(
                    sampling_request_history[-1]
                )
            return payload

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        num_llm_calls_available = self._max_llm_calls_per_run()
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

        resumed_from_running = False
        if isinstance(resume_state, dict) and isinstance(resume_state.get("messages"), list):
            messages = copy.deepcopy(resume_state["messages"])
            round = int(resume_state.get("round") or 0)
            remaining_calls = resume_state.get("num_llm_calls_available")
            if isinstance(remaining_calls, int):
                num_llm_calls_available = remaining_calls
            else:
                num_llm_calls_available = max(0, self._max_llm_calls_per_run() - round)
            context_fold_trigger_step = resume_state.get("context_fold_trigger_step")
            context_reset_events = copy.deepcopy(
                resume_state.get("context_reset_events") or []
            )
            cumulative_token_usage = copy.deepcopy(
                resume_state.get("cumulative_token_usage") or cumulative_token_usage
            )
            pending_summary_for_thinking = resume_state.get("pending_summary_for_thinking")
            if (
                not pending_summary_for_thinking
                and resume_state.get("termination") == "context_summary"
                and isinstance(resume_state.get("context_reset_event"), dict)
            ):
                pending_summary_for_thinking = resume_state["context_reset_event"].get(
                    "summary_text"
                )
            resumed_from_running = True

        def make_running_state_payload(extra: Optional[Dict] = None) -> Dict:
            payload = {
                **task_metadata,
                **make_sampling_state_payload(),
                **self._make_context_management_stats(
                    context_reset_events, cumulative_token_usage
                ),
                "num_llm_calls_available": num_llm_calls_available,
                "context_fold_trigger_step": context_fold_trigger_step,
            }
            if resumed_from_running:
                payload["resumed_from_running"] = True
                if resume_state.get("updated_at"):
                    payload["resumed_snapshot_updated_at"] = resume_state.get("updated_at")
            if pending_summary_for_thinking:
                payload["pending_summary_for_thinking"] = pending_summary_for_thinking
            if extra:
                payload.update(extra)
            return payload

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
            result.update(make_sampling_state_payload())
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
                    extra_payload=make_running_state_payload(),
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
                    **make_running_state_payload(),
                    "auto_judge": result.get("auto_judge"),
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
            extra_payload=make_running_state_payload(),
            final=False,
        )

        per_task_time_limit_seconds = task_time_limit_seconds()
        while num_llm_calls_available > 0:
            if (
                per_task_time_limit_seconds is not None
                and time.time() - start_time > per_task_time_limit_seconds
            ):
                prediction = task_time_limit_termination()
                termination = task_time_limit_termination()
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
            request_messages = self._with_context_awareness(
                inference_messages,
                cumulative_token_usage,
            )
            assistant_message = self.call_server(
                request_messages,
                planning_port,
                use_tools=True,
                request_log_callback=record_sampling_request,
            )
            print(f"Round {round}: {assistant_message}")
            messages.append(assistant_message)
            if pending_summary_for_thinking:
                injected = self._inject_pending_summary_to_thinking(
                    messages,
                    pending_summary_for_thinking,
                )
                if injected:
                    pending_summary_for_thinking = None

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
                    **make_running_state_payload(),
                    "num_llm_calls_available": num_llm_calls_available,
                    "has_tool_call": has_tool_call,
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
                final_request_messages = self._with_context_awareness(
                    final_inference_messages,
                    cumulative_token_usage,
                )
                assistant_message = self.call_server(
                    final_request_messages,
                    planning_port,
                    use_tools=False,
                    request_log_callback=record_sampling_request,
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
                    extra_payload=make_running_state_payload(),
                    final=False,
                )
                return finalize_result(prediction, termination)

            messages_before_reset = len(messages)
            messages, context_action, reset_info = self.maybe_reset_context(
                messages,
                question,
                usage=latest_usage,
                planning_port=planning_port,
                system_prompt=system_prompt,
                request_log_callback=record_sampling_request,
            )
            if context_action == "discard_all":
                self._accumulate_context_reset_usage(
                    cumulative_token_usage,
                    (reset_info or {}).get("usage"),
                )
                reset_event = {
                    "round": round,
                    "action": "discard_all",
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
                    extra_payload=make_running_state_payload({
                        "context_reset_events": context_reset_events,
                        "context_reset_event": reset_event,
                    }),
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
                    last_user_idx = self._last_user_index(messages)
                    messages_to_summarize = (
                        messages[last_user_idx:]
                        if last_user_idx is not None
                        else messages
                    )
                    summary_request_messages = self._build_summary_request_messages(
                        messages_to_summarize,
                        question,
                    )
                    summary_message = self._generate_summary_message(
                        summary_request_messages,
                        planning_port,
                        request_log_callback=record_sampling_request,
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
                        "summary_messages_start_index": last_user_idx,
                        "summary_messages_count": len(messages_to_summarize),
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
                        extra_payload=make_running_state_payload({
                            "context_reset_events": context_reset_events,
                            "context_reset_event": reset_event,
                        }),
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

            max_tokens = self._forced_finalize_context_tokens()
            if reset_info and reset_info.get("token_count") is not None:
                token_count = reset_info["token_count"]
                token_count_source = reset_info.get("token_count_source")
            else:
                token_count_messages = self._prepare_inference_messages(messages)
                token_count, token_count_source, _ = self._get_context_token_count(
                    token_count_messages,
                    usage=None,
                    prefer_usage=False,
                )
            print(
                f"round: {round}, token count: {token_count}, "
                f"source: {token_count_source}"
            )

            if token_count > max_tokens:
                print(f"Token quantity exceeds the limit: {token_count} > {max_tokens}")

                messages.append({"role": "user", "content": TRUNCATED_MESSAGE + FINAL_MESSAGE})
                truncated_inference_messages = self._prepare_inference_messages(messages)
                truncated_request_messages = self._with_context_awareness(
                    truncated_inference_messages,
                    cumulative_token_usage,
                )
                assistant_message = self.call_server(
                    truncated_request_messages,
                    planning_port,
                    use_tools=False,
                    request_log_callback=record_sampling_request,
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
                    extra_payload=make_running_state_payload(),
                    final=False,
                )
                return finalize_result(prediction, termination)

        prediction = self._latest_assistant_content(messages) or strip_think_blocks(
            messages[-1].get("content", "")
        )
        if termination == "unknown" and num_llm_calls_available <= 0:
            termination = "exceed_llm_calls"
        return finalize_result(prediction, termination)

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name in self.tool_map:
            tool_payload = copy.deepcopy(tool_args)
            tool_payload["params"] = copy.deepcopy(tool_args)
            raw_result = self.tool_map[tool_name].call(tool_payload, **kwargs)
            result = raw_result
            return result
        return f"Error: Tool {tool_name} not found"
