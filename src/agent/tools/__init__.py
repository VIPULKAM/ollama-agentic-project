"""Tool registry for the AI Coding Agent.

This module provides centralized access to all tools available to the agent,
including file operations and RAG-based semantic search.
"""

from typing import List
from langchain_core.tools import BaseTool

from .file_ops import (
    get_read_file_tool,
    get_list_directory_tool,
    get_write_file_tool,
    get_search_code_tool,
)
from .rag_search import get_rag_search_tool
from src.config.settings import Settings


def get_all_tools(settings: Settings = None) -> List[BaseTool]:
    """Get all available tools for the agent.

    This function returns all tools that the agent can use:
    - File Operations (4 tools): read, write, list, search
    - RAG Search (1 tool): semantic codebase search

    Args:
        settings: Application settings (default: Settings())

    Returns:
        List[BaseTool]: List of LangChain tools ready for agent use
    """
    if settings is None:
        settings = Settings()

    tools = []

    # File operation tools (Step 2 - ✅ COMPLETE)
    tools.append(get_read_file_tool())
    tools.append(get_list_directory_tool())
    tools.append(get_write_file_tool())
    tools.append(get_search_code_tool())

    # RAG search tool (Step 3 - ✅ COMPLETE)
    tools.append(get_rag_search_tool(settings))

    return tools


__all__ = ["get_all_tools"]
