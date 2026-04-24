import os
import random
import re
import time
from typing import Optional

import requests
from openai import OpenAI

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency in some envs
    tiktoken = None


JINA_API_KEY = os.getenv("JINA_API_KEYS", "")


def strip_think_blocks(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned.strip()


def call_openai_compatible_model(
    *,
    query: str,
    api_key_env: str,
    api_base_env: str,
    model_env: str,
    default_api_base: Optional[str] = None,
    default_model: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    max_tokens: int = 32768,
    max_retry: int = 3,
) -> Optional[str]:
    api_key = (
        os.getenv(api_key_env)
        or os.getenv("QWEN_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("API_KEY")
    )
    api_base = os.getenv(api_base_env) or os.getenv("API_BASE") or default_api_base
    model_name = os.getenv(model_env) or os.getenv("SUMMARY_MODEL_NAME") or default_model

    if not api_key or not api_base or not model_name:
        return None

    client = OpenAI(api_key=api_key, base_url=api_base)

    for attempt in range(max_retry):
        try:
            request_kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": query}],
                "max_tokens": max_tokens,
                "temperature": (
                    float(os.getenv("QWEN_TEMPERATURE", "1.0"))
                    if temperature is None
                    else temperature
                ),
                "top_p": (
                    float(os.getenv("QWEN_TOP_P", "0.95"))
                    if top_p is None
                    else top_p
                ),
                "presence_penalty": (
                    float(os.getenv("QWEN_PRESENCE_PENALTY", "1.5"))
                    if presence_penalty is None
                    else presence_penalty
                ),
            }

            extra_body = {}
            resolved_top_k = (
                int(os.getenv("QWEN_TOP_K", "20")) if top_k is None else top_k
            )
            resolved_min_p = (
                float(os.getenv("QWEN_MIN_P", "0.0")) if min_p is None else min_p
            )
            resolved_repetition_penalty = (
                float(os.getenv("QWEN_REPETITION_PENALTY", "1.0"))
                if repetition_penalty is None
                else repetition_penalty
            )
            if resolved_top_k is not None:
                extra_body["top_k"] = resolved_top_k
            if resolved_min_p is not None:
                extra_body["min_p"] = resolved_min_p
            if resolved_repetition_penalty is not None:
                extra_body["repetition_penalty"] = resolved_repetition_penalty
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            response = client.chat.completions.create(**request_kwargs)
            return strip_think_blocks(response.choices[0].message.content)
        except Exception as exc:
            print(
                f"call_openai_compatible_model attempt {attempt} error: {exc}",
                flush=True,
            )
            if attempt == max_retry - 1:
                return None
            time.sleep(random.uniform(1, 4))

    return None


def jina_readpage(url: str, max_retry: int = 3) -> str:
    if not JINA_API_KEY:
        return "[browse] JINA_API_KEYS environment variable is not set."

    for attempt in range(max_retry):
        headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
        try:
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=50,
            )
            if response.status_code == 200:
                return response.text
            print(f"Jina API error: {response.text}", flush=True)
        except Exception as exc:
            print(f"jina_readpage attempt {attempt} error: {exc}", flush=True)
        time.sleep(0.5)

    return "[browse] Failed to read page."


def split_text_by_tokens(text: str, chunk_limit: int = 95000, overlap: int = 1024):
    if tiktoken is None:
        approx_chunk_chars = max(8000, chunk_limit * 3)
        if len(text) <= approx_chunk_chars:
            return [text]
        chunks = []
        step = max(1, approx_chunk_chars - overlap)
        for start in range(0, len(text), step):
            end = min(start + approx_chunk_chars, len(text))
            chunks.append(text[start:end])
        return chunks

    encoding = tiktoken.get_encoding("cl100k_base")
    tokenized = encoding.encode(text)

    if len(tokenized) <= chunk_limit:
        return [text]

    num_split = max(2, len(tokenized) // chunk_limit + 1)
    chunk_len = max(1, len(tokenized) // num_split)
    chunks = []
    for idx in range(num_split):
        start_idx = idx * chunk_len
        end_idx = min(start_idx + chunk_len + overlap, len(tokenized))
        chunks.append(encoding.decode(tokenized[start_idx:end_idx]))
    return chunks
