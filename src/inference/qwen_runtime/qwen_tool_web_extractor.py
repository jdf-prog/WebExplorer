import os
from typing import Optional, Union

from .qwen_tool_utils import (
    call_openai_compatible_model,
    jina_readpage,
    split_text_by_tokens,
)


def summarize_webpage(url: str, goal: str, max_retry: int = 3) -> str:
    source_text = jina_readpage(url, max_retry=max_retry)
    if source_text.strip() == "" or source_text.startswith("[browse] Failed to read page."):
        return "Browse error. Please try again."

    prompt_template = (
        "Please read the source content and answer the goal below.\n"
        "--- begin of source content ---\n"
        "{source_text}\n"
        "--- end of source content ---\n\n"
        "If there is no relevant information, clearly say so.\n"
        "Goal:\n"
        "{goal}"
    )

    chunks = split_text_by_tokens(source_text)
    chunk_outputs = []
    for chunk in chunks:
        prompt = prompt_template.format(source_text=chunk, goal=goal)
        output = call_openai_compatible_model(
            query=prompt,
            api_key_env="QWEN_EXTRACTOR_API_KEY",
            api_base_env="QWEN_EXTRACTOR_API_BASE",
            model_env="QWEN_EXTRACTOR_MODEL_NAME",
            default_api_base=os.getenv(
                "QWEN_API_BASE",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            default_model=os.getenv("QWEN_EXTRACTOR_MODEL_FALLBACK", "qwen3.5-plus"),
            max_retry=1,
        )
        if output:
            chunk_outputs.append(output)

    if not chunk_outputs:
        return "Browse error. Please try again."

    if len(chunk_outputs) == 1:
        return chunk_outputs[0]

    merged = []
    for idx, output in enumerate(chunk_outputs, start=1):
        merged.append(
            f"--- begin of result part {idx} ---\n{output}\n--- end of result part {idx} ---"
        )
    return (
        "Since the content is too long, the result is split and answered separately. "
        "Please combine the results to get the complete answer.\n"
        + "\n\n".join(merged)
    )


class QwenWebExtractorTool:
    name = "web_extractor"
    description = (
        "Crawl webpage content, and if given a goal, further summarize the relevant "
        "content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "The webpage urls.",
            },
            "goal": {
                "type": "string",
                "description": "The goal of the visit.",
            },
        },
        "required": ["urls", "goal"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.max_retry = cfg.get("max_retry", 3)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            urls = params["urls"]
            goal = params.get("goal", "")
        except Exception:
            return "[web_extractor] Invalid request format: missing 'urls' or 'goal'"

        if isinstance(urls, str):
            urls = [urls]
        if not isinstance(urls, list) or not urls:
            return "[web_extractor] Error: 'urls' must be a non-empty list of strings"

        goal = goal or "Summarize the most relevant information from the page."
        outputs = []
        for url in urls:
            if not isinstance(url, str) or not url.strip():
                continue
            result = summarize_webpage(url.strip(), goal, max_retry=self.max_retry)
            outputs.append(
                f"--- extractor result for [{url.strip()}] ---\n{result}\n--- end of extractor result ---"
            )

        if not outputs:
            return "[web_extractor] Error: no valid urls were provided"
        return "\n\n".join(outputs)

    def make_tool_message(self, content: str) -> dict:
        return {"role": "tool", "name": self.name, "content": content}
