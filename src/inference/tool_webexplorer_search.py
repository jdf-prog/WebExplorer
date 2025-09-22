import json
import os
import http.client
import time
import random
from typing import List, Union, Optional
from qwen_agent.tools.base import BaseTool, register_tool
import requests

SERPER_API_KEY = os.environ.get('SERPER_KEY_ID', '')
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
    description = "Web search tool that performs batched web searches: supply an array 'queries'; the tool retrieves search results for each query."
    parameters = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. The queries will be sent to search engine. You will get the brief search results with (title, url, snippet)s for each query."
            },
        },
        "required": ["queries"],
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
            # Single query (backward compatibility)
            queries = [queries]
        
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