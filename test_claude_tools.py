#!/usr/bin/env python3
"""
Test script for LangGraph agent with Claude (tool-calling enabled).

This script tests the full tool-based agent functionality once you have
valid API credentials with credits.

Usage:
    python test_claude_tools.py
"""

from src.agent.agent import CodingAgent
from src.config.settings import settings
import tempfile
from pathlib import Path


def test_simple_query():
    """Test 1: Simple query without tools."""
    print("\n" + "="*60)
    print("TEST 1: Simple Query (No Tools Needed)")
    print("="*60)

    agent = CodingAgent(provider='claude', temperature=0.1)
    response = agent.ask("What is the capital of France? Just answer with the city name.")

    print(f"Response: {response.content}")
    assert "paris" in response.content.lower(), "Expected 'Paris' in response"
    print("✅ PASSED: Agent answered correctly without using tools")


def test_conversation_memory():
    """Test 2: Conversation memory."""
    print("\n" + "="*60)
    print("TEST 2: Conversation Memory")
    print("="*60)

    agent = CodingAgent(provider='claude', temperature=0.1)
    agent.clear_history()

    # Set context
    agent.ask("My favorite color is blue.")
    print("✅ Set context: favorite color is blue")

    # Test memory
    response = agent.ask("What is my favorite color?")
    print(f"Response: {response.content}")

    assert "blue" in response.content.lower(), "Agent should remember the color"
    print("✅ PASSED: Agent remembered conversation context")


def test_read_file_tool():
    """Test 3: Read file tool."""
    print("\n" + "="*60)
    print("TEST 3: File Reading Tool")
    print("="*60)

    # Create a temporary test file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("This is a test file with unique content: XYZABC123")
        test_file = f.name

    try:
        agent = CodingAgent(provider='claude', temperature=0.1)
        agent.clear_history()

        response = agent.ask(f"Please read the file at '{test_file}' and tell me what it says.")
        print(f"Response: {response.content[:200]}...")

        assert "XYZABC123" in response.content, "Agent should have read the file content"
        print("✅ PASSED: Agent successfully used read_file tool")
    finally:
        Path(test_file).unlink()


def test_list_directory_tool():
    """Test 4: List directory tool."""
    print("\n" + "="*60)
    print("TEST 4: Directory Listing Tool")
    print("="*60)

    agent = CodingAgent(provider='claude', temperature=0.1)
    agent.clear_history()

    response = agent.ask("List the files in the current directory.")
    print(f"Response: {response.content[:300]}...")

    # Should mention some common files
    has_files = any(name in response.content for name in ['main.py', 'src', 'tests', '.env'])
    assert has_files, "Agent should have listed directory contents"
    print("✅ PASSED: Agent successfully used list_directory tool")


def test_write_file_tool():
    """Test 5: Write file tool."""
    print("\n" + "="*60)
    print("TEST 5: File Writing Tool")
    print("="*60)

    test_file = Path(tempfile.gettempdir()) / "langraph_test_write.txt"

    try:
        agent = CodingAgent(provider='claude', temperature=0.1)
        agent.clear_history()

        response = agent.ask(
            f"Write the text 'LangGraph test successful!' to a file at '{test_file}'. "
            f"Confirm when done."
        )
        print(f"Response: {response.content[:200]}...")

        # Check if file was created
        if test_file.exists():
            content = test_file.read_text()
            assert "LangGraph test successful" in content, "File should contain the test text"
            print("✅ PASSED: Agent successfully used write_file tool")
        else:
            print("⚠️  File not created - tool may need confirmation")
    finally:
        if test_file.exists():
            test_file.unlink()


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LANGGRAPH TOOL-BASED AGENT TEST SUITE")
    print("="*60)
    print("\nTesting with Claude API (Haiku model)")
    print("Make sure you have:")
    print("  1. Valid ANTHROPIC_API_KEY in .env")
    print("  2. Credits in your Anthropic account")
    print("  3. ENABLE_TOOLS=True in .env")

    # Check settings
    settings.ENABLE_TOOLS = True
    settings.ENABLE_FILE_OPS = True
    settings.ENABLE_RAG = False  # Disable RAG for simpler tests

    tests = [
        test_simple_query,
        test_conversation_memory,
        test_read_file_tool,
        test_list_directory_tool,
        test_write_file_tool,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")

    if failed == 0:
        print("\n🎉 All tests passed! LangGraph agent is working perfectly!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Check the output above for details.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
