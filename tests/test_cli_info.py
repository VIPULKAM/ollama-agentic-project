"""Test CLI info command fix for all provider modes."""

import pytest
from unittest.mock import Mock, patch
from src.cli.main import print_model_info
from src.agent.agent import CodingAgent


def test_print_model_info_ollama_mode():
    """Test that print_model_info works with Ollama provider."""
    # Create a mock agent
    mock_agent = Mock(spec=CodingAgent)
    mock_agent.get_model_info.return_value = {
        "provider": "ollama",
        "temperature": 0.1,
        "ollama_model": "qwen2.5-coder:1.5b",
        "ollama_base_url": "http://localhost:11434",
        "ollama_deployment": "local",
    }

    # This should not raise any KeyError
    try:
        with patch('src.cli.main.console'):
            print_model_info(mock_agent)
        assert True, "print_model_info succeeded for ollama mode"
    except KeyError as e:
        pytest.fail(f"KeyError in ollama mode: {e}")


def test_print_model_info_claude_mode():
    """Test that print_model_info works with Claude provider."""
    # Create a mock agent
    mock_agent = Mock(spec=CodingAgent)
    mock_agent.get_model_info.return_value = {
        "provider": "claude",
        "temperature": 0.1,
        "claude_model": "claude-3-haiku-20240307",
        "claude_api_configured": True,
    }

    # This should not raise any KeyError
    try:
        with patch('src.cli.main.console'):
            print_model_info(mock_agent)
        assert True, "print_model_info succeeded for claude mode"
    except KeyError as e:
        pytest.fail(f"KeyError in claude mode: {e}")


def test_print_model_info_hybrid_mode():
    """Test that print_model_info works with Hybrid provider."""
    # Create a mock agent
    mock_agent = Mock(spec=CodingAgent)
    mock_agent.get_model_info.return_value = {
        "provider": "hybrid",
        "temperature": 0.1,
        "ollama_model": "qwen2.5-coder:1.5b",
        "ollama_base_url": "http://localhost:11434",
        "ollama_deployment": "local",
        "claude_model": "claude-3-haiku-20240307",
        "claude_api_configured": True,
        "routing_keywords": ["architecture", "design pattern", "refactor", "optimize"],
    }

    # This should not raise any KeyError
    try:
        with patch('src.cli.main.console'):
            print_model_info(mock_agent)
        assert True, "print_model_info succeeded for hybrid mode"
    except KeyError as e:
        pytest.fail(f"KeyError in hybrid mode: {e}")


def test_print_model_info_missing_keys():
    """Test that print_model_info handles missing keys gracefully."""
    # Create a mock agent with minimal info
    mock_agent = Mock(spec=CodingAgent)
    mock_agent.get_model_info.return_value = {
        "provider": "ollama",
        "temperature": 0.1,
        # Missing other keys
    }

    # This should not raise any KeyError, should use N/A defaults
    try:
        with patch('src.cli.main.console'):
            print_model_info(mock_agent)
        assert True, "print_model_info handled missing keys gracefully"
    except KeyError as e:
        pytest.fail(f"KeyError with missing keys: {e}")


if __name__ == "__main__":
    # Quick local test
    print("Testing Ollama mode...")
    test_print_model_info_ollama_mode()
    print("✓ Ollama mode OK")

    print("Testing Claude mode...")
    test_print_model_info_claude_mode()
    print("✓ Claude mode OK")

    print("Testing Hybrid mode...")
    test_print_model_info_hybrid_mode()
    print("✓ Hybrid mode OK")

    print("Testing missing keys...")
    test_print_model_info_missing_keys()
    print("✓ Missing keys handled OK")

    print("\n✅ All CLI info command tests passed!")
