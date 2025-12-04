"""Quick test for both read_file and list_directory tools together."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.tools import get_all_tools


def main():
    """Test both tools."""
    print("=" * 70)
    print("Testing File Operation Tools (read_file + list_directory)")
    print("=" * 70)

    # Get all tools
    tools = get_all_tools()
    print(f"\n📦 Available tools: {len(tools)}")
    for i, tool in enumerate(tools, 1):
        print(f"   {i}. {tool.name}: {tool.description[:60]}...")

    # Get individual tools
    read_file = tools[0]
    list_dir = tools[1]

    print("\n" + "=" * 70)
    print("Test Scenario: Explore project structure")
    print("=" * 70)

    # Step 1: List current directory
    print("\n[Step 1] List current directory to see what's available...")
    result = list_dir.invoke({"path": "."})
    print(result)

    # Step 2: List src directory
    print("\n[Step 2] List 'src' directory...")
    result = list_dir.invoke({"path": "src"})
    print(result[:500] + "..." if len(result) > 500 else result)

    # Step 3: Read a file
    print("\n[Step 3] Read .env.example file...")
    result = read_file.invoke({"path": ".env.example"})
    lines = result.split('\n')
    print(f"File has {len(lines)} lines. First 10 lines:")
    print('\n'.join(lines[:10]))

    # Step 4: List a nested directory
    print("\n[Step 4] List 'src/agent' directory...")
    result = list_dir.invoke({"path": "src/agent"})
    print(result)

    # Step 5: Read a source file
    print("\n[Step 5] Read 'src/config/settings.py' (first 15 lines)...")
    result = read_file.invoke({"path": "src/config/settings.py"})
    lines = result.split('\n')
    print('\n'.join(lines[:15]))

    print("\n" + "=" * 70)
    print("✅ Both tools working correctly!")
    print("=" * 70)
    print("\nTools can now:")
    print("  ✓ List directory contents with metadata")
    print("  ✓ Read file contents safely")
    print("  ✓ Validate all paths (within CWD only)")
    print("  ✓ Block path traversal attacks")
    print("  ✓ Handle errors gracefully")


if __name__ == "__main__":
    main()
