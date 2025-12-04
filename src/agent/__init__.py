"""Agent module - LangChain-based coding assistant."""

from .agent import CodingAgent
from .prompts import get_system_prompt

__all__ = ["CodingAgent", "get_system_prompt"]
