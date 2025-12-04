"""Unit tests for CodingAgent with LangGraph implementation."""

import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import sys
import subprocess

print("sys.executable:", sys.executable)
print("sys.path:", sys.path)
print("pip freeze output:")
subprocess.run([sys.executable, "-m", "pip", "freeze"], shell=False)

from src.agent.agent import CodingAgent
from src.config.settings import settings
from langchain_core.messages import AIMessage, HumanMessage



def check_ollama_running():
    """Check if Ollama service is running."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def check_model_available(model_name="qwen2.5-coder:1.5b"):
    """Check if the required model is available in Ollama."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        return model_name in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

@pytest.fixture(scope="module")
def ollama_preflight_check():
    """Fixture to check for Ollama and model availability once per module."""
    if not check_ollama_running():
        pytest.skip("Ollama service is not running. Please run 'ollama serve'.")
    if not check_model_available():
        pytest.skip("qwen2.5-coder:1.5b model not found. Please run 'ollama pull qwen2.5-coder:1.5b'.")

@pytest.fixture
def agent(monkeypatch, ollama_preflight_check):
    """Create a CodingAgent instance configured to use the LangGraph agent with tools."""
    # Ensure tools are enabled for this test session
    monkeypatch.setattr(settings, 'ENABLE_TOOLS', True)
    monkeypatch.setattr(settings, 'ENABLE_FILE_OPS', True)
    monkeypatch.setattr(settings, 'ENABLE_RAG', True)
    
    # Ensure LLM is set up for Ollama
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'ollama')
    monkeypatch.setattr(settings, 'MODEL_NAME', 'qwen2.5-coder:1.5b')
    monkeypatch.setattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')

    return CodingAgent(
        provider="ollama",
        model_name="qwen2.5-coder:1.5b",
        temperature=0.1
    )

@pytest.fixture
def test_file():
    """Fixture to create a temporary file for tool use tests."""
    content = "This is a test file for the agent to read. It contains some unique text."
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    yield Path(tmp_path)
    
    # Cleanup the file
    import os
    os.unlink(tmp_path)


class TestCodingAgentWithLangGraph:
    """Test suite for the LangGraph CodingAgent."""

    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly in tool-using LangGraph mode."""
        assert agent is not None
        assert agent.use_tools is True
        assert agent.provider == "ollama"
        assert len(agent.tools) > 0, "Agent should have tools loaded."
        assert hasattr(agent, 'langgraph_app'), "Agent should have a compiled LangGraph app."

    def test_model_info(self, agent):
        """Test that get_model_info returns correct information for LangGraph tool mode."""
        info = agent.get_model_info()
        assert info["agent_mode"] == "LangGraph (Tools enabled)"
        assert "tools" in info
        assert "read_file" in info["tools"]
        assert "rag_search" in info["tools"]

    def test_simple_query_without_tools(self, agent):
        """Test a simple query that shouldn't require tools."""
        agent.clear_history() # Ensure a clean slate
        query = "What is the capital of France?"
        response = agent.ask(query)
        
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert "paris" in response.content.lower()
        assert "error" not in response.content.lower()

    def test_conversation_history(self, agent):
        """Test that conversation history is maintained with the LangGraph agent."""
        agent.clear_history()
        
        agent.ask("My favorite color is blue.")
        
        query2 = "What is my favorite color?"
        response2 = agent.ask(query2)
        
        assert "blue" in response2.content.lower()
        
        history = agent.get_conversation_history()
        assert "blue" in history.lower()
        assert "What is my favorite color?" in history

    def test_clear_history(self, agent):
        """Test that clearing history works for LangGraph agent."""
        agent.ask("This is a test question for history clear.")
        agent.clear_history()
        history = agent.get_conversation_history()
        assert history == ""

    def test_tool_usage_read_file(self, agent, test_file):
        """Test that the LangGraph agent can use the read_file tool."""
        agent.clear_history()
        query = f"Read the contents of the file named '{test_file.name}' located in '{test_file.parent}'."
        response = agent.ask(query)

        assert response is not None
        assert "This is a test file" in response.content
        assert "unique text" in response.content
        assert "error" not in response.content.lower()
        
    def test_tool_usage_list_directory(self, agent, test_file):
        """Test that the LangGraph agent can use the list_directory tool."""
        agent.clear_history()
        parent_dir = test_file.parent
        query = f"List all files and folders in the directory '{parent_dir}'."
        response = agent.ask(query)

        assert response is not None
        assert test_file.name in response.content
        assert "error" not in response.content.lower()

    # Additional tests for write_file, search_code, rag_search would be added here
    # These would involve setting up appropriate test data and asserting on the outcomes.

    # Example for write_file (would need more sophisticated mock/filesystem setup)
    # def test_tool_usage_write_file(self, agent):
    #     agent.clear_history()
    #     temp_write_file = Path(tempfile.gettempdir()) / "test_write.txt"
    #     query = f"Write 'Hello LangGraph!' into the file '{temp_write_file}'."
    #     response = agent.ask(query)
    #     assert "successfully" in response.content.lower()
    #     assert temp_write_file.read_text() == "Hello LangGraph!"
    #     temp_write_file.unlink()

