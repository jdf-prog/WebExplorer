import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Union

import requests


SERPER_API_KEY = os.getenv("SERPER_KEY_ID", "")
LOCAL_SEARCH_URL = os.environ.get("LOCAL_SEARCH_URL", "http://127.0.0.1:8800").rstrip("/")
SERPER_CACHE_ENABLED = os.environ.get("SERPER_CACHE_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
SERPER_CACHE_PATH = os.environ.get(
    "SERPER_CACHE_PATH",
    "./cache/serper_search_cache.jsonl",
)
_SERPER_CACHE = None
LOCAL_SEARCH_CACHE_ENABLED = os.environ.get("LOCAL_SEARCH_CACHE_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
SEARCH_TIMING_LOG_PATH = os.environ.get("WEBEXPLORER_SEARCH_TIMING_LOG", "").strip()
_LOCAL_SEARCH_CACHE: dict[tuple[str, int, str], str] = {}
_LOCAL_SEARCH_CACHE_LOCK = threading.Lock()
_SEARCH_TIMING_LOCK = threading.Lock()


def _normalize_cache_query(query: str) -> str:
    return query.strip()


def _load_serper_cache() -> dict:
    global _SERPER_CACHE
    if _SERPER_CACHE is not None:
        return _SERPER_CACHE

    cache = {}
    cache_path = Path(SERPER_CACHE_PATH)
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    query = item.get("query")
                    content = item.get("content")
                    if isinstance(query, str) and isinstance(content, str):
                        cache[_normalize_cache_query(query)] = content
        except OSError as exc:
            print(f"serper cache load error: {exc}", flush=True)

    _SERPER_CACHE = cache
    return _SERPER_CACHE


def _append_serper_cache(query: str, content: str, topk: int) -> None:
    if not SERPER_CACHE_ENABLED:
        return

    cache_path = Path(SERPER_CACHE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "query": query,
        "content": content,
        "topk": topk,
        "engine": "serper",
        "cached_at": int(time.time()),
    }
    try:
        with cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"serper cache write error: {exc}", flush=True)


def _get_serper_cache_hit(query: str) -> Optional[str]:
    if not SERPER_CACHE_ENABLED:
        return None
    return _load_serper_cache().get(_normalize_cache_query(query))


def _store_serper_cache(query: str, content: str, topk: int) -> None:
    if not SERPER_CACHE_ENABLED:
        return
    normalized_query = _normalize_cache_query(query)
    cache = _load_serper_cache()
    cache[normalized_query] = content
    _append_serper_cache(normalized_query, content, topk)


def _log_search_timing(record: dict) -> None:
    if not SEARCH_TIMING_LOG_PATH:
        return
    try:
        path = Path(SEARCH_TIMING_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _SEARCH_TIMING_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"search timing log write error: {exc}", flush=True)


def _local_cache_key(query: str, topk: int) -> tuple[str, int, str]:
    retrieval_mode = os.environ.get("LOCAL_SEARCH_RETRIEVAL_MODE", "").strip().lower() or "hybrid"
    return (_normalize_cache_query(query), topk, retrieval_mode)


def _get_local_search_cache_hit(query: str, topk: int) -> Optional[str]:
    if not LOCAL_SEARCH_CACHE_ENABLED:
        return None
    with _LOCAL_SEARCH_CACHE_LOCK:
        return _LOCAL_SEARCH_CACHE.get(_local_cache_key(query, topk))


def _store_local_search_cache(query: str, topk: int, content: str) -> None:
    if not LOCAL_SEARCH_CACHE_ENABLED:
        return
    with _LOCAL_SEARCH_CACHE_LOCK:
        _LOCAL_SEARCH_CACHE[_local_cache_key(query, topk)] = content


def get_search_results(query: str, topk: int = 10, max_retry: int = 3, engine: str = "serper") -> str:
    engine = (engine or "serper").strip().lower()
    if engine == "local":
        return local_search_with_service(query, topk=topk, max_retry=max_retry)
    if engine != "serper":
        raise ValueError(f"Unsupported search engine: {engine}")
    if not SERPER_API_KEY:
        raise ValueError("SERPER_KEY_ID environment variable is not set")

    cached_result = _get_serper_cache_hit(query)
    if cached_result is not None:
        print(f"serper cache hit: {query}", flush=True)
        return cached_result

    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {"q": query, "num": topk}

    for retry_cnt in range(max_retry):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            results = response.json()
            if "organic" not in results:
                raise ValueError(f"No results found for query: '{query}'")

            snippets = []
            for page in results["organic"][:topk]:
                snippet = page.get("snippet", "")
                if page.get("date"):
                    snippet = f"Date published: {page['date']}\n{snippet}"
                snippet = snippet.replace("Your browser can't play this video.", "")
                snippets.append(
                    "\n".join(
                        [
                            f"<title>{page.get('title', '')}</title>",
                            f"<url>{page.get('link', '')}</url>",
                            "<snippet>",
                            snippet,
                            "</snippet>",
                        ]
                    )
                )

            content = "\n\n".join(snippets)
            _store_serper_cache(query, content, topk)
            return content
        except Exception as exc:
            print(f"qwen web_search retry {retry_cnt} error: {exc}", flush=True)
            if retry_cnt == max_retry - 1:
                return (
                    f"No results found for '{query}'. Try a more general query. "
                    f"Error: {exc}"
                )
            time.sleep(random.uniform(1, 4))

    return f"Search failed after {max_retry} retries for query: '{query}'"


def local_search_with_service(query: str, topk: int = 10, max_retry: int = 3) -> str:
    cached_result = _get_local_search_cache_hit(query, topk)
    if cached_result is not None:
        print(f"local search cache hit: {query}", flush=True)
        return cached_result

    payload = {"query": query, "topn": topk}
    retrieval_mode = os.environ.get("LOCAL_SEARCH_RETRIEVAL_MODE", "").strip().lower()
    if retrieval_mode:
        payload["mode"] = retrieval_mode
    for retry_cnt in range(max_retry):
        try:
            response = requests.post(f"{LOCAL_SEARCH_URL}/search", json=payload, timeout=30)
            response.raise_for_status()
            pages = response.json().get("results", [])
            if not pages:
                return f"No local offline results found for '{query}'. Try a more general query."
            snippets = []
            for page in pages[:topk]:
                snippet = page.get("summary", "")
                if page.get("date"):
                    snippet = f"Date published: {page['date']}\n{snippet}"
                snippets.append(
                    "\n".join(
                        [
                            f"<title>{page.get('title', '')}</title>",
                            f"<url>{page.get('url', '')}</url>",
                            "<snippet>",
                            snippet,
                            "</snippet>",
                        ]
                    )
                )
            content = "\n\n".join(snippets)
            _store_local_search_cache(query, topk, content)
            return content
        except Exception as exc:
            print(f"qwen local web_search retry {retry_cnt} error: {exc}", flush=True)
            if retry_cnt == max_retry - 1:
                return f"No local offline results found for '{query}'. Error: {exc}"
            time.sleep(random.uniform(0.5, 2))
    return f"Local offline search failed after {max_retry} retries for query: '{query}'"


def get_searches_results(queries: List[str], topk: int = 10, max_retry: int = 3, engine: str = "serper") -> str:
    max_queries = int(os.environ.get("WEBEXPLORER_SEARCH_MAX_QUERIES_PER_CALL", "8"))
    max_workers = int(os.environ.get("WEBEXPLORER_SEARCH_MAX_WORKERS", "4"))
    queries = [str(query) for query in queries[:max_queries]]
    if not queries:
        return ""

    def run_one(query: str) -> str:
        started = time.time()
        result = get_search_results(query, topk=topk, max_retry=max_retry, engine=engine)
        _log_search_timing(
            {
                "event": "web_search_query",
                "engine": engine,
                "query": query,
                "topk": topk,
                "elapsed_s": round(time.time() - started, 4),
                "ts": time.time(),
            }
        )
        return result

    if len(queries) == 1 or max_workers <= 1:
        search_results = [run_one(query) for query in queries]
    else:
        worker_count = min(max_workers, len(queries))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            search_results = list(executor.map(run_one, queries))

    results = []
    for query, result in zip(queries, search_results):
        results.append(
            f"--- search result for [{query}] ---\n{result}\n--- end of search result ---"
        )
    return "\n\n".join(results)


class QwenWebSearchTool:
    name = "web_search"
    description = "Search for information from the internet."
    parameters = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "The list of search queries.",
            }
        },
        "required": ["queries"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.topk = cfg.get("topk", int(os.getenv("QWEN_SEARCH_TOPK", "10")))
        self.max_retry = cfg.get("max_retry", 3)
        self.search_engine = cfg.get(
            "search_engine",
            os.getenv("QWEN_SEARCH_ENGINE", os.getenv("WEBEXPLORER_SEARCH_ENGINE", "serper")),
        ).strip().lower()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            queries = params["queries"]
        except Exception:
            return "[web_search] Invalid request format: missing 'queries'"

        if isinstance(queries, str):
            try:
                decoded = json.loads(queries)
            except json.JSONDecodeError:
                queries = [queries]
            else:
                queries = decoded if isinstance(decoded, list) else [queries]
        if not isinstance(queries, list):
            return "[web_search] Error: 'queries' must be a list of strings"

        try:
            return get_searches_results(
                queries=queries,
                topk=self.topk,
                max_retry=self.max_retry,
                engine=self.search_engine,
            )
        except Exception as exc:
            return f"[web_search] Error: {exc}"

    def make_tool_message(self, content: str) -> dict:
        return {"role": "tool", "name": self.name, "content": content}
