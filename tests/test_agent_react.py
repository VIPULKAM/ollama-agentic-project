"""Lightweight tests for the ReAct agent implementation."""

import pytest
from unittest.mock import patch, MagicMock

from langchain_community.llms.fake import FakeListLLM
from langchain_core.messages import AIMessage
from langchain.agents import AgentExecutor, create_react_agent

from src.agent.agent import CodingAgent
from src.config import settings

@pytest.fixture(autouse=True)
def override_settings(monkeypatch):
    """Fixture to automatically enable tools for all tests in this file."""
    monkeypatch.setattr(settings, "ENABLE_TOOLS", True)
    monkeypatch.setattr(settings, "ENABLE_FILE_OPS", True)
    monkeypatch.setattr(settings, "ENABLE_RAG", False) # Keep it simple, no RAG for these tests

@pytest.fixture
def fake_agent() -> CodingAgent:
    """
    Fixture to create a 'blank' CodingAgent by patching its __init__ method.
    This allows for controlled, isolated unit testing of its methods.
    """
    with patch.object(CodingAgent, '__init__', lambda self, **kwargs: None):
        agent = CodingAgent()
        agent.use_tools = True
        agent.store = {}
        agent.session_id = "test_session"
        # Setup a minimal fake llm by default
        agent.llm = FakeListLLM(responses=["Final Answer: default response"])
    return agent

def test_react_agent_setup(fake_agent):
    """Test that the _setup_react_agent method configures the agent correctly."""
    fake_agent._setup_react_agent()

    assert fake_agent.use_tools is True
    assert len(fake_agent.tools) > 0
    assert "read_file" in [tool.name for tool in fake_agent.tools]
    assert "rag_search" not in [tool.name for tool in fake_agent.tools] # As per fixture
    assert fake_agent.prompt is not None
    assert fake_agent.chain_with_history is not None

def test_react_agent_simple_question(fake_agent):
    """Test a simple question that shouldn't require tools."""
    # 1. Configure the agent for this specific test
    fake_llm = FakeListLLM(responses=["Final Answer: I am a fake agent."])
    fake_agent.llm = fake_llm
    fake_agent.claude_llm = None # Not used in this test
    fake_agent.ollama_llm = fake_llm # For _select_llm logic
    fake_agent.provider = "ollama"

    # 2. Mock the chain rebuilding process to inject our fake LLM
    with patch.object(fake_agent, '_rebuild_chain_with_llm') as mock_rebuild:
        # The rebuild should return a runnable that, when invoked, gives our expected output
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = {"output": "I am a fake agent."}
        mock_rebuild.return_value = mock_runnable

        # 3. Call the method to be tested
        response = fake_agent.ask("Hello, who are you?")

    # 4. Assert the results
    mock_rebuild.assert_called_once() # Ensure the chain was rebuilt
    assert isinstance(response, AIMessage)
    assert "I am a fake agent." in response.content

def test_react_agent_with_tool_response(fake_agent):
    """Test the agent's ability to parse and execute a tool action."""
    # 1. Configure the agent with a multi-response LLM
    tool_thought = "Thought: I should list the files.\nAction: list_directory\nAction Input: ."
    final_thought = "Observation: Directory listing is `file.py`.\nThought: I have the answer now.\nFinal Answer: The directory contains `file.py`."
    fake_llm = FakeListLLM(responses=[tool_thought, final_thought])
    
    fake_agent.llm = fake_llm
    fake_agent.claude_llm = None
    fake_agent.ollama_llm = fake_llm
    fake_agent.provider = "ollama"

    # 2. Mock the actual tool to prevent filesystem access and control its output
    with patch("src.agent.tools.file_ops.ListDirectoryTool._run", return_value="Directory listing is `file.py`") as mock_tool_run:
        
        # 3. Setup the real agent executor with our fake LLM.
        # This is what _rebuild_chain_with_llm is supposed to do.
        fake_agent._setup_react_agent() # To get tools and prompt
        
        # We replace the agent's default LLM with our multi-response one
        agent_runnable = create_react_agent(fake_llm, fake_agent.tools, fake_agent.prompt)
        executor = AgentExecutor(
            agent=agent_runnable,
            tools=fake_agent.tools,
            handle_parsing_errors=True
        )

        # 4. Mock the rebuild method to return our fully-formed fake executor
        with patch.object(fake_agent, '_rebuild_chain_with_llm') as mock_rebuild:
            mock_rebuild.return_value.invoke.return_value = executor.invoke(
                {"input": "What files are here?", "chat_history": []}
            )
            
            # 5. Call the method to be tested
            response = fake_agent.ask("What files are here?")

    # 6. Assert the results
    mock_tool_run.assert_called_once_with('.')
    assert isinstance(response, AIMessage)
    assert "The directory contains `file.py`." in response.content
