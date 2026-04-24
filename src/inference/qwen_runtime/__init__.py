"""Qwen-specific inference runtime for WebExplorer."""

__all__ = ["QwenFunctionCallAgent"]


def __getattr__(name):
    if name == "QwenFunctionCallAgent":
        from .qwen_agent import QwenFunctionCallAgent

        return QwenFunctionCallAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
