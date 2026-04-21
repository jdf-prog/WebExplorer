import os
import time
import random
import requests
from typing import Union, Optional
from qwen_agent.tools.base import BaseTool, register_tool
from openai import OpenAI
import tiktoken


JINA_API_KEY = os.getenv("JINA_API_KEYS", "")

def get_geminiflash_response(query: str, temperature: float = 0.0, max_retry: int = 5) -> str:
    """Get response from Gemini Flash model using standard OpenAI-compatible API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    api_base = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
    
    if not api_key:
        print("Warning: GEMINI_API_KEY not set, skipping Gemini response", flush=True)
        return None
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        for retry_cnt in range(max_retry):
            try:
                response = client.chat.completions.create(
                    model="gemini-2.0-flash-exp",
                    messages=[{"role": "user", "content": query}],
                    temperature=temperature,
                    max_tokens=32768
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"get_geminiflash_response {retry_cnt} error: {e}", flush=True)
                if retry_cnt == max_retry - 1:
                    return None
                time.sleep(random.uniform(4, 32))
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}", flush=True)
    
    return None


def get_deepseekchat_response(query: str, temperature: float = 0.0, max_retry: int = 3) -> str:
    """Get response from DeepSeek Chat model using standard OpenAI-compatible API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    
    if not api_key:
        print("Warning: DEEPSEEK_API_KEY not set, skipping DeepSeek response", flush=True)
        return None
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        for retry_cnt in range(max_retry):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": query}],
                    temperature=temperature,
                    max_tokens=32768
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"get_deepseekchat_response {retry_cnt} error: {e}", flush=True)
                if retry_cnt == max_retry - 1:
                    return None
                time.sleep(random.uniform(4, 32))
    except Exception as e:
        print(f"Failed to initialize DeepSeek client: {e}", flush=True)
    
    return None


def get_openai_response(query: str, temperature: float = 0.0, max_retry: int = 3) -> str:
    """Get response from OpenAI API."""
    api_key = os.environ.get("API_KEY")
    url_llm = os.environ.get("API_BASE")
    model_name = os.environ.get("SUMMARY_MODEL_NAME", "")
    
    if not api_key or not url_llm:
        return None
        
    client = OpenAI(
        api_key=api_key,
        base_url=url_llm,
    )
    
    for attempt in range(max_retry):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": query}],
                temperature=temperature
            )
            content = chat_response.choices[0].message.content
            return content
        except Exception as e:
            print(f"get_openai_response {attempt} error: {e}", flush=True)
            if attempt == max_retry - 1:
                return None
            time.sleep(random.uniform(1, 4))
    return None


def jina_readpage(url: str, max_retry: int = 3) -> str:
    """Read webpage content using Jina service."""
    if not JINA_API_KEY:
        return "[browse] JINA_API_KEYS environment variable is not set."
    
    for attempt in range(max_retry):
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}",
        }
        try:
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=50
            )
            if response.status_code == 200:
                webpage_content = response.text
                return webpage_content
            else:
                print(f"Jina API error: {response.text}")
                raise ValueError("jina readpage error")
        except Exception as e:
            print(f"jina_readpage {attempt} error: {e}", flush=True)
            time.sleep(0.5)
            if attempt == max_retry - 1:
                return "[browse] Failed to read page."
                
    return "[browse] Failed to read page."


def get_browse_results(url: str, browse_query: str, read_engine: str = "jina", generate_engine: str = "deepseekchat", max_retry: int = 3) -> str:
    """Get browse results by reading webpage and extracting relevant information."""
    time.sleep(random.uniform(0, 16))
    
    # Read webpage content
    source_text = ""
    for retry_cnt in range(max_retry):
        try:
            if read_engine == "jina":
                source_text = jina_readpage(url, max_retry=1)
            else:
                raise ValueError(f"Unsupported read engine: {read_engine}")
            break
        except Exception as e:
            print(f"Read {read_engine} {retry_cnt} error: {e}, url: {url}", flush=True)
            if any(word in str(e) for word in ["Client Error"]):
                return "Access to this URL is denied. Please try again."
            time.sleep(random.uniform(16, 64))
    
    if source_text.strip() == "" or source_text.startswith("[browse] Failed to read page."):
        print(f"Browse error with empty source_text.", flush=True)
        return "Browse error. Please try again."
    

    query = f"Please read the source content and answer a following question:\n---begin of source content---\n{source_text}\n---end of source content---\n\nIf there is no relevant information, please clearly refuse to answer. Now answer the question based on the above content:\n{browse_query}"
    
    # 处理长内容分块（仿照deep_research_utils.py的逻辑）
    encoding = tiktoken.get_encoding("cl100k_base")
    tokenized_source_text = encoding.encode(source_text)
    
    if len(tokenized_source_text) > 95000:  # 使用与原代码相同的token限制
        output = "Since the content is too long, the result is split and answered separately. Please combine the results to get the complete answer.\n"
        num_split = max(2, len(tokenized_source_text) // 95000 + 1)
        chunk_len = len(tokenized_source_text) // num_split
        print(f"Browse too long with length {len(tokenized_source_text)}, split into {num_split} parts, with each part length {chunk_len}", flush=True)
        
        outputs = []
        for i in range(num_split):
            start_idx = i * chunk_len
            end_idx = min(start_idx + chunk_len + 1024, len(tokenized_source_text))
            source_text_i = encoding.decode(tokenized_source_text[start_idx:end_idx])
            query_i = f"Please read the source content and answer a following question:\n--- begin of source content ---\n{source_text_i}\n--- end of source content ---\n\nIf there is no relevant information, please clearly refuse to answer. Now answer the question based on the above content:\n{browse_query}"
            
            if generate_engine == "geminiflash":
                output_i = get_geminiflash_response(query_i, temperature=0.0, max_retry=1)
            elif generate_engine == "deepseekchat":
                output_i = get_deepseekchat_response(query_i, temperature=0.0, max_retry=1)
            elif generate_engine == "openai":
                output_i = get_openai_response(query_i, temperature=0.0, max_retry=1)
            else:
                raise ValueError(f"Unsupported generate engine: {generate_engine}")
            
            outputs.append(output_i or "")
        
        for i in range(num_split):
            output += f"--- begin of result part {i+1} ---\n{outputs[i]}\n--- end of result part {i+1} ---\n\n"
    else:
        if generate_engine == "geminiflash":
            output = get_geminiflash_response(query, temperature=0.0, max_retry=1)
        elif generate_engine == "deepseekchat":
            output = get_deepseekchat_response(query, temperature=0.0, max_retry=1)
        elif generate_engine == "openai":
            output = get_openai_response(query, temperature=0.0, max_retry=1)
        else:
            raise ValueError(f"Unsupported generate engine: {generate_engine}")
    
    if output is None or output.strip() == "":
        print(f"Browse error with empty output.", flush=True)
        return "Browse error. Please try again."
    
    return output


@register_tool("browse", allow_overwrite=True)
class WebExplorerBrowse(BaseTool):
    name = "browse"
    description = "Browse a webpage and extract relevant information based on a specific query."
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL of the webpage to browse."
            },
            "query": {
                "type": "string",
                "description": "The specific query to extract relevant information from the webpage."
            }
        },
        "required": ["url", "query"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        default_read_engine = os.environ.get("BROWSE_READ_ENGINE", "jina")
        default_generate_engine = os.environ.get("BROWSE_ENGINE", "deepseekchat")
        self.read_engine = cfg.get("read_engine", default_read_engine) if cfg else default_read_engine
        self.generate_engine = cfg.get("generate_engine", default_generate_engine) if cfg else default_generate_engine
        self.max_retry = cfg.get("max_retry", 3) if cfg else 3

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            query = params["query"]
        except:
            return "[Browse] Invalid request format: Input must be a JSON object containing 'url' and 'query' fields"

        if not url or not isinstance(url, str):
            return "[Browse] Error: 'url' is missing, empty, or not a string"
        
        if not isinstance(query, str):
            return "[Browse] Error: 'query' is missing or not a string"
        
        if query == "":
            query = "Detailed summary of the page."

        try:
            result = get_browse_results(
                url=url,
                browse_query=query,
                read_engine=self.read_engine,
                generate_engine=self.generate_engine,
                max_retry=self.max_retry
            )
            
            print(f'Browse Summary Length {len(result)}; Browse Summary Content {result}')
            return result.strip()
            
        except Exception as e:
            return f"[Browse] Error: {str(e)}"

if __name__ == "__main__":
    result = WebExplorerBrowse().call({"url": "https://www.baidu.com", "query": "Detailed summary of the page."})
    print(result)
