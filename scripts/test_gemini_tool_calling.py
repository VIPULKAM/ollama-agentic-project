#!/usr/bin/env python3
"""
Minimal test to verify Gemini tool calling with bind_tools().
Tests whether ChatGoogleGenerativeAI can successfully call tools.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class TestInput(BaseModel):
    """Input schema for test tool."""
    text: str = Field(description="Input text to process")


class TestTool(BaseTool):
    """A simple test tool for verification."""
    name: str = "test_tool"
    description: str = "A test tool that processes text input"
    args_schema: type[BaseModel] = TestInput

    def _run(self, text: str) -> str:
        """Process the input text."""
        return f"Processed: {text}"


def test_gemini_tool_calling():
    """Test Gemini's ability to call tools using bind_tools()."""

    print("=" * 80)
    print("GEMINI TOOL CALLING TEST")
    print("=" * 80)

    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n❌ ERROR: GOOGLE_API_KEY not found in environment")
        print("Please set GOOGLE_API_KEY in your .env file or environment")
        return False

    print(f"\n✓ API key found: {api_key[:10]}...")

    # Initialize model
    print("\n1. Initializing ChatGoogleGenerativeAI with gemini-1.5-flash...")
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=api_key
        )
        print("   ✓ Model initialized successfully")
    except Exception as e:
        print(f"   ❌ Failed to initialize model: {e}")
        return False

    # Create tool
    print("\n2. Creating test tool...")
    tool = TestTool()
    print(f"   ✓ Tool created: {tool.name}")
    print(f"   ✓ Description: {tool.description}")

    # Bind tool to model
    print("\n3. Binding tool to model using bind_tools()...")
    try:
        model_with_tools = model.bind_tools([tool])
        print("   ✓ Tool bound successfully")
    except Exception as e:
        print(f"   ❌ Failed to bind tools: {e}")
        return False

    # Invoke model with a query that should trigger tool use
    print("\n4. Invoking model with query that should trigger tool use...")
    query = "Please use the test_tool to process the text 'Hello World'"
    print(f"   Query: {query}")

    try:
        response = model_with_tools.invoke(query)
        print("   ✓ Model invoked successfully")
    except Exception as e:
        print(f"   ❌ Failed to invoke model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Analyze response
    print("\n5. Analyzing response...")
    print(f"\n   Response type: {type(response)}")
    print(f"   Response class: {response.__class__.__name__}")

    # Check for tool_calls attribute
    if hasattr(response, 'tool_calls'):
        print(f"\n   ✓ tool_calls attribute present: {response.tool_calls}")
        if response.tool_calls:
            print(f"   ✓ Tool calls found: {len(response.tool_calls)} call(s)")
            for i, tool_call in enumerate(response.tool_calls):
                print(f"\n   Tool call {i+1}:")
                print(f"     - Name: {tool_call.get('name', 'N/A')}")
                print(f"     - Args: {tool_call.get('args', 'N/A')}")
                print(f"     - ID: {tool_call.get('id', 'N/A')}")
        else:
            print(f"   ❌ tool_calls is empty")
    else:
        print(f"   ❌ tool_calls attribute NOT present")

    # Check for additional_kwargs
    if hasattr(response, 'additional_kwargs'):
        print(f"\n   additional_kwargs: {response.additional_kwargs}")

    # Print full response content
    print(f"\n   Response content: {response.content}")

    # Print response structure
    print("\n6. Full response structure:")
    print("-" * 80)
    print(response)
    print("-" * 80)

    # Determine success
    success = (
        hasattr(response, 'tool_calls') and
        response.tool_calls is not None and
        len(response.tool_calls) > 0
    )

    print("\n" + "=" * 80)
    if success:
        print("✅ SUCCESS: Gemini successfully called tools using bind_tools()")
        print("=" * 80)
        return True
    else:
        print("❌ FAILURE: Gemini did NOT call tools")
        print("   This may indicate:")
        print("   - The model doesn't support tool calling")
        print("   - The query wasn't clear enough")
        print("   - The tool binding didn't work as expected")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = test_gemini_tool_calling()
    sys.exit(0 if success else 1)
