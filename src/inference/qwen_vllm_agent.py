from typing import Dict, List, Optional, Union

from qwen_agent.llm import BaseChatModel
from qwen_agent.tools import BaseTool

from tool_qwen_code_interpreter import QwenCodeInterpreterTool
from tool_qwen_web_extractor import QwenWebExtractorTool
from tool_qwen_web_search import QwenWebSearchTool
from vllm_react_agent import MultiTurnReactAgent


class QwenVllmReactAgent(MultiTurnReactAgent):
    def __init__(
        self,
        function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
        llm: Optional[Union[Dict, BaseChatModel]] = None,
        **kwargs,
    ):
        super().__init__(function_list=function_list, llm=llm, **kwargs)

    def _available_tool_instances(self) -> List[BaseTool]:
        return [
            QwenCodeInterpreterTool(),
            QwenWebSearchTool(),
            QwenWebExtractorTool(),
        ]
