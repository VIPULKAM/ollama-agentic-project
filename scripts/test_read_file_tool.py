"""Quick test script for read_file tool."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.tools.file_ops import get_read_file_tool


def test_read_file_tool():
    """Test the read_file tool with various scenarios."""
    tool = get_read_file_tool()

    print("=" * 70)
    print("Testing ReadFile Tool")
    print("=" * 70)

    # Test 1: Read a valid file
    print("\n[Test 1] Reading .env.example file...")
    result = tool._run(".env.example")
    if "Error" in result or "error" in result:
        print(f"❌ FAILED: {result}")
    else:
        lines = result.split('\n')
        print(f"✅ SUCCESS: Read {len(lines)} lines")
        print(f"   First 3 lines: {lines[:3]}")

    # Test 2: Try to read a non-existent file
    print("\n[Test 2] Trying to read non-existent file...")
    result = tool._run("nonexistent_file.txt")
    if "does not exist" in result or "Error" in result:
        print(f"✅ SUCCESS: Correctly rejected - {result[:80]}...")
    else:
        print(f"❌ FAILED: Should have rejected non-existent file")

    # Test 3: Try path traversal attack
    print("\n[Test 3] Testing path traversal attack prevention...")
    result = tool._run("../../etc/passwd")
    if "Access denied" in result or "outside current working directory" in result:
        print(f"✅ SUCCESS: Path traversal blocked - {result[:80]}...")
    else:
        print(f"❌ FAILED: Path traversal should be blocked")

    # Test 4: Try to read a directory
    print("\n[Test 4] Trying to read a directory...")
    result = tool._run("src")
    if "not a file" in result or "Error" in result:
        print(f"✅ SUCCESS: Correctly rejected directory - {result[:80]}...")
    else:
        print(f"❌ FAILED: Should reject directories")

    # Test 5: Read this test script itself
    print("\n[Test 5] Reading this test script...")
    result = tool._run("scripts/test_read_file_tool.py")
    if "Error" in result or "error" in result:
        print(f"❌ FAILED: {result}")
    else:
        lines = result.split('\n')
        print(f"✅ SUCCESS: Read {len(lines)} lines of test script")

    # Test 6: Absolute path within CWD
    print("\n[Test 6] Testing absolute path within CWD...")
    abs_path = str(Path.cwd() / "README.md")
    result = tool._run(abs_path)
    if "Error" in result or "error" in result:
        print(f"❌ FAILED: {result}")
    else:
        lines = result.split('\n')
        print(f"✅ SUCCESS: Read README.md ({len(lines)} lines) using absolute path")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_read_file_tool()
