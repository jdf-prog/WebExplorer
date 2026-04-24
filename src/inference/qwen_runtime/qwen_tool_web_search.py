import os
import random
import time
from typing import List, Optional, Union

import requests


SERPER_API_KEY = os.getenv("SERPER_KEY_ID", "")


def get_search_results(query: str, topk: int = 10, max_retry: int = 3) -> str:
    if not SERPER_API_KEY:
        raise ValueError("SERPER_KEY_ID environment variable is not set")

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
                if page.get("source"):
                    snippet = f"Source: {page['source']}\n{snippet}"
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

            return "\n\n".join(snippets)
        except Exception as exc:
            print(f"qwen web_search retry {retry_cnt} error: {exc}", flush=True)
            if retry_cnt == max_retry - 1:
                return (
                    f"No results found for '{query}'. Try a more general query. "
                    f"Error: {exc}"
                )
            time.sleep(random.uniform(1, 4))

    return f"Search failed after {max_retry} retries for query: '{query}'"


def get_searches_results(queries: List[str], topk: int = 10, max_retry: int = 3) -> str:
    results = []
    for query in queries:
        result = get_search_results(query, topk=topk, max_retry=max_retry)
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

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            queries = params["queries"]
        except Exception:
            return "[web_search] Invalid request format: missing 'queries'"

        if isinstance(queries, str):
            queries = [queries]
        if not isinstance(queries, list):
            return "[web_search] Error: 'queries' must be a list of strings"

        try:
            return get_searches_results(
                queries=queries,
                topk=self.topk,
                max_retry=self.max_retry,
            )
        except Exception as exc:
            return f"[web_search] Error: {exc}"

    def make_tool_message(self, content: str) -> dict:
        return {"role": "tool", "name": self.name, "content": content}
