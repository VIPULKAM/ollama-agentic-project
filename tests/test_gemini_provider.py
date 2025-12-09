"""Tests for the Gemini provider integration."""

import pytest
from unittest.mock import patch

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms.fake import FakeListLLM

from src.agent.agent import CodingAgent
from src.config import settings

@pytest.fixture
def gemini_agent(monkeypatch):
    """Fixture to create a CodingAgent with the Gemini provider."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "dummy-api-key")
    # Disable tools mode for these tests since FakeListLLM doesn't support bind_tools()
    monkeypatch.setattr(settings, "ENABLE_TOOLS", False)

    # Patch the actual ChatGoogleGenerativeAI class to return a fake LLM
    # This avoids real API calls while still testing the initialization logic
    with patch("src.agent.agent.ChatGoogleGenerativeAI") as mock_gemini:
        mock_gemini.return_value = FakeListLLM(responses=["Test response from Gemini"])
        agent = CodingAgent()
        yield agent

def test_gemini_agent_initialization(gemini_agent):
    """Test that the agent initializes correctly with the Gemini provider."""
    assert gemini_agent.provider == "gemini"
    # The mock replaces the real instance, so we check the mock was called
    assert gemini_agent.gemini_llm is not None
    # Check that the main llm is the gemini llm
    assert gemini_agent.llm == gemini_agent.gemini_llm

def test_gemini_agent_ask(gemini_agent):
    """Test a simple question with the Gemini provider."""
    # The FakeListLLM is already set up in the fixture
    response = gemini_agent.ask("Hello, Gemini?")
    
    assert "Test response from Gemini" in response.content

def test_gemini_api_key_error(monkeypatch):
    """Test that a ValueError is raised if the GOOGLE_API_KEY is missing."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GOOGLE_API_KEY", None) # Ensure key is not set
    
    with pytest.raises(ValueError, match="GOOGLE_API_KEY must be set"):
        CodingAgent()
