#!/usr/bin/env python3
"""
Test if qwen2.5-coder:1.5b can follow ReAct format.
This is CRITICAL before implementing the full agent system.
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

def test_react_format():
    """Test if the model can follow Thought/Action/Observation format."""

    print("=" * 60)
    print("Testing qwen2.5-coder:1.5b with ReAct Format")
    print("=" * 60)

    # Initialize model
    llm = OllamaLLM(
        model="qwen2.5-coder:1.5b",
        temperature=0.1
    )

    # ReAct format prompt
    react_prompt = PromptTemplate.from_template("""You are an AI assistant with access to tools.

Available tools:
- read_file: Read contents of a file
- search_code: Search for code patterns

You MUST use this exact format:

Thought: [your reasoning about what to do]
Action: [tool name]
Action Input: [input for the tool]
Observation: [this will be filled by the tool]
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: [your final response to the user]

Question: {question}

Let's begin!
Thought:""")

    # Test questions
    test_cases = [
        {
            "question": "Read the file 'main.py' and tell me what it does",
            "expected_action": "read_file"
        },
        {
            "question": "Search for all functions that contain 'agent' in the codebase",
            "expected_action": "search_code"
        },
        {
            "question": "List files in the src directory",
            "expected_action": "list_directory"  # Should fail if model can't generalize
        }
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_case['question']}")
        print(f"{'='*60}")

        # Generate response
        prompt = react_prompt.format(question=test_case['question'])
        response = llm.invoke(prompt)

        print(f"\nModel Response:\n{response}\n")

        # Check if response follows format
        has_thought = "Thought:" in response
        has_action = "Action:" in response
        has_action_input = "Action Input:" in response

        # Check if correct tool is identified
        correct_tool = test_case["expected_action"].lower() in response.lower()

        result = {
            "question": test_case["question"],
            "follows_format": has_thought and has_action and has_action_input,
            "correct_tool": correct_tool,
            "response": response[:200] + "..." if len(response) > 200 else response
        }
        results.append(result)

        print(f"\n✓ Has 'Thought:': {has_thought}")
        print(f"✓ Has 'Action:': {has_action}")
        print(f"✓ Has 'Action Input:': {has_action_input}")
        print(f"✓ Identified correct tool: {correct_tool}")

    # Summary
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    format_success = sum(1 for r in results if r["follows_format"])
    tool_success = sum(1 for r in results if r["correct_tool"])

    print(f"\nFormat Compliance: {format_success}/{len(results)} tests")
    print(f"Tool Identification: {tool_success}/{len(results)} tests")

    if format_success >= 2 and tool_success >= 2:
        print("\n✅ PASS: qwen2.5-coder:1.5b can follow ReAct format!")
        print("   → Proceed with full implementation")
        return True
    else:
        print("\n❌ FAIL: qwen2.5-coder:1.5b struggles with ReAct format")
        print("   → Consider fallback strategies:")
        print("      1. Use simpler tool routing (keyword matching)")
        print("      2. Use Claude for tool operations in hybrid mode")
        print("      3. Try different prompt format")
        return False


def test_simple_tool_routing():
    """Test simpler approach: keyword-based tool routing."""

    print(f"\n\n{'='*60}")
    print("Testing Fallback: Simple Keyword-Based Routing")
    print(f"{'='*60}")

    llm = OllamaLLM(
        model="qwen2.5-coder:1.5b",
        temperature=0.1
    )

    prompt = PromptTemplate.from_template("""Given this user request, identify what action to take.

User request: {question}

Available actions:
- READ_FILE: Read a file
- WRITE_FILE: Write to a file
- LIST_DIR: List directory contents
- SEARCH_CODE: Search for code patterns
- SEARCH_SEMANTIC: Semantic search across codebase

Respond with ONLY the action name and the target (e.g., "READ_FILE main.py" or "SEARCH_CODE agent").

Action:""")

    tests = [
        ("Read the main.py file", "READ_FILE"),
        ("Search for authentication code", "SEARCH"),
        ("List files in src/", "LIST_DIR"),
    ]

    success = 0
    for question, expected in tests:
        response = llm.invoke(prompt.format(question=question))
        if expected.lower() in response.lower():
            print(f"✓ '{question}' → {response.strip()}")
            success += 1
        else:
            print(f"✗ '{question}' → {response.strip()} (expected {expected})")

    print(f"\nSimple routing: {success}/{len(tests)} tests passed")
    return success >= 2


if __name__ == "__main__":
    print("\n🚀 Starting ReAct Format Validation")
    print("=" * 60)

    # Test ReAct format
    react_works = test_react_format()

    if not react_works:
        # Test fallback
        fallback_works = test_simple_tool_routing()

        if fallback_works:
            print("\n📋 RECOMMENDATION: Use simple keyword-based tool routing")
        else:
            print("\n📋 RECOMMENDATION: Use hybrid mode (Claude for tools, Ollama for chat)")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
