"""Unit tests for CodingAgent with ReAct agent and tools."""

import pytest
import subprocess
import tempfile
from pathlib import Path

from src.agent.agent import CodingAgent
from src.config.settings import settings

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
    """Create a CodingAgent instance configured to use the ReAct agent with tools."""
    # Ensure tools are enabled for this test session
    monkeypatch.setattr(settings, 'ENABLE_TOOLS', True)
    monkeypatch.setattr(settings, 'ENABLE_FILE_OPS', True)
    monkeypatch.setattr(settings, 'ENABLE_RAG', True)
    
    return CodingAgent(
        provider="ollama",
        model_name="qwen2.5-coder:1.5b",
        temperature=0.1
    )

@pytest.fixture
def test_file():
    """Fixture to create a temporary file for tool use tests."""
    content = "This is a test file for the agent to read."
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".txt") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    yield Path(tmp_path)
    
    # Cleanup the file
    import os
    os.unlink(tmp_path)


class TestCodingAgentWithTools:
    """Test suite for the ReAct CodingAgent."""

    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly in tool-using mode."""
        assert agent is not None
        assert agent.use_tools is True
        assert agent.provider == "ollama"
        assert len(agent.tools) > 0, "Agent should have tools loaded."

    def test_model_info(self, agent):
        """Test that get_model_info returns correct information for tool mode."""
        info = agent.get_model_info()
        assert info["agent_mode"] == "ReAct (Tools enabled)"
        assert "tools" in info
        assert "read_file" in info["tools"]
        assert "rag_search" in info["tools"]

    def test_simple_query_without_tools(self, agent):
        """Test a simple query that shouldn't require tools."""
        query = "What is Python?"
        response = agent.ask(query)
        
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert "error" not in response.content.lower()

    def test_conversation_history(self, agent):
        """Test that conversation history is maintained with the ReAct agent."""
        agent.clear_history()
        
        agent.ask("My name is Test User.")
        
        query2 = "What is my name?"
        response2 = agent.ask(query2)
        
        assert "test user" in response2.content.lower()
        
        history = agent.get_conversation_history()
        assert "Test User" in history
        assert "What is my name?" in history

    def test_clear_history(self, agent):
        """Test that clearing history works."""
        agent.ask("This is a test question.")
        agent.clear_history()
        history = agent.get_conversation_history()
        assert history == ""

    def test_tool_usage_read_file(self, agent, test_file):
        """Test that the agent can use the read_file tool to answer a question."""
        # This is a simple integration test for tool use.
        # It relies on the model's ability to follow ReAct instructions.
        query = f"What are the contents of the file located at '{test_file}'?"
        response = agent.ask(query)

        assert response is not None
        # Check if the agent's response includes the file's content
        assert "This is a test file" in response.content

    def test_tool_usage_list_directory(self, agent, test_file):
        """Test that the agent can use the list_directory tool."""
        # Ask the agent to list the contents of the parent directory of the test file
        parent_dir = test_file.parent
        query = f"List the files in the directory '{parent_dir}'."
        response = agent.ask(query)

        assert response is not None
        # The agent's output should contain the name of the test file
        assert test_file.name in response.content