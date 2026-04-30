import os
import http.client
import time
import random
import json
from pathlib import Path
from typing import List, Union, Optional
from qwen_agent.tools.base import BaseTool, register_tool
import requests

SERPER_API_KEY = os.environ.get('SERPER_KEY_ID', '')
SERPER_CACHE_ENABLED = os.environ.get("SERPER_CACHE_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
SERPER_CACHE_PATH = os.environ.get(
    "SERPER_CACHE_PATH",
    "./cache/serper_search_cache.jsonl",
)
_SERPER_CACHE = None


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
        except OSError as e:
            print(f"serper cache load error: {e}", flush=True)

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
    except OSError as e:
        print(f"serper cache write error: {e}", flush=True)


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


def get_searches_results(queries: List[str], topk: int = 10, engine: str = "serper", max_retry: int = 3) -> str:
    """Get search results for multiple queries using specified search engine."""
    results = []
    for i, query in enumerate(queries):
        result = get_search_results(query, topk=topk, engine=engine, max_retry=max_retry)
        # 使用与deep_research_utils.py相同的格式
        formatted_result = f"--- search result for [{query}] ---\n{result}\n--- end of search result ---"
        results.append(formatted_result)
    return "\n\n".join(results)


def get_search_results(query: str, topk: int = 10, engine: str = "serper", max_retry: int = 3) -> str:
    """Get search results for a single query using specified search engine."""
    if engine == "serper":
        return google_search_with_serp(query, topk=topk, max_retry=max_retry)
    else:
        raise ValueError(f"Unsupported search engine: {engine}")


def contains_chinese_basic(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return any('\u4E00' <= char <= '\u9FFF' for char in text)


def google_search_with_serp(query: str, topk: int = 10, max_retry: int = 3) -> str:
    """Perform Google search using Serper API."""
    if not SERPER_API_KEY:
        raise ValueError("SERPER_KEY_ID environment variable is not set")

    cached_result = _get_serper_cache_hit(query)
    if cached_result is not None:
        print(f"serper cache hit: {query}", flush=True)
        return cached_result
    
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "num": topk
    }
    for retry_cnt in range(max_retry):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            results = response.json()
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = []
            
            for page in results["organic"][:topk]:
                # 构建snippet内容
                snippet = ""
                if "snippet" in page:
                    snippet = page["snippet"]
                
                # 添加日期信息到snippet中（如果有的话）
                if "date" in page:
                    snippet = f"Date published: {page['date']}\n{snippet}"
                
                # 添加来源信息到snippet中（如果有的话）
                if "source" in page:
                    snippet = f"Source: {page['source']}\n{snippet}"
                
                # 清理内容
                snippet = snippet.replace("Your browser can't play this video.", "")
                
                # 使用XML格式构建结果
                redacted_version = f"<title>{page['title']}</title>\n<url>{page['link']}</url>\n<snippet>\n{snippet}\n</snippet>"
                web_snippets.append(redacted_version)

            content = "\n\n".join(web_snippets)
            _store_serper_cache(query, content, topk)
            return content
            
        except Exception as e:
            print(f"google_search_with_serp {retry_cnt} error: {e}", flush=True)
            if retry_cnt == max_retry - 1:
                return f"No results found for '{query}'. Try with a more general query. Error: {str(e)}"
            time.sleep(random.uniform(1, 4))
    
    return f"Search failed after {max_retry} retries for query: '{query}'"


@register_tool("search", allow_overwrite=True)
class WebExplorerSearch(BaseTool):
    name = "search"
    description = "Web search in parallel. The parameter is a list of queries. The queries will be sent to a search engine. You will get the brief search results with (title, url, snippet)s for each query."
    parameters = {
        "properties": {
            "queries": {
                "description": "The queries. Google advanced search operators are supported.",
                "items": {
                    "type": "string"
                },
                "type": "array"
            },
        },
        "required": ["queries"],
        "type": "object",
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        self.search_engine = cfg.get("search_engine", "serper") if cfg else "serper"
        self.topk = cfg.get("topk", 10) if cfg else 10
        self.max_retry = cfg.get("max_retry", 3) if cfg else 3

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            queries = params["queries"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'queries' field"
        
        if isinstance(queries, str):
            try:
                decoded = json.loads(queries)
            except json.JSONDecodeError:
                # Single query (backward compatibility)
                queries = [queries]
            else:
                queries = decoded if isinstance(decoded, list) else [queries]
        
        if not isinstance(queries, list):
            return "[Search] Error: 'queries' must be a list of strings"
        
        try:
            result = get_searches_results(
                queries=queries,
                topk=self.topk,
                engine=self.search_engine,
                max_retry=self.max_retry
            )
            return result
        except Exception as e:
            return f"[Search] Error: {str(e)}"

if __name__ == "__main__":
    result = WebExplorerSearch().call({"queries": ["What is the capital of July?", "What is the capital of China?"]})
    print(result)
