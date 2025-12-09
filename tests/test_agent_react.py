"""Lightweight tests for the ReAct agent implementation.

NOTE: These tests are DEPRECATED. The agent now uses LangGraph instead of the old ReAct pattern.
See test_langgraph_agent.py for current tests.
"""

import pytest

# Mark all tests in this file as skipped since the ReAct agent implementation has been replaced
pytestmark = pytest.mark.skip(reason="ReAct agent replaced with LangGraph. See test_langgraph_agent.py")
