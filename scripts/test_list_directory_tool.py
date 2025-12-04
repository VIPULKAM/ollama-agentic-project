"""Quick test script for list_directory tool."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.tools.file_ops import get_list_directory_tool


def test_list_directory_tool():
    """Test the list_directory tool with various scenarios."""
    tool = get_list_directory_tool()

    print("=" * 70)
    print("Testing ListDirectory Tool")
    print("=" * 70)

    # Test 1: List current directory (default)
    print("\n[Test 1] Listing current directory (default)...")
    result = tool._run()
    if "Error" in result or "error" in result:
        print(f"❌ FAILED: {result}")
    else:
        lines = result.split('\n')
        print(f"✅ SUCCESS: Listed directory")
        print(result)

    # Test 2: List current directory explicitly
    print("\n[Test 2] Listing current directory (explicit '.')...")
    result = tool._run(".")
    if "Error" in result or "error" in result:
        print(f"❌ FAILED: {result}")
    else:
        print(f"✅ SUCCESS: Listed current directory")

    # Test 3: List a subdirectory
    print("\n[Test 3] Listing 'src' directory...")
    result = tool._run("src")
    if "Error" in result or "error" in result:
        print(f"❌ FAILED: {result}")
    else:
        print(f"✅ SUCCESS: Listed src directory")
        print(result)

    # Test 4: Try to list a non-existent directory
    print("\n[Test 4] Trying to list non-existent directory...")
    result = tool._run("nonexistent_dir_xyz123")
    if "does not exist" in result or "Error" in result:
        print(f"✅ SUCCESS: Correctly rejected - {result[:80]}...")
    else:
        print(f"❌ FAILED: Should have rejected non-existent directory")

    # Test 5: Try path traversal attack
    print("\n[Test 5] Testing path traversal attack prevention...")
    result = tool._run("../../etc")
    if "Access denied" in result or "outside current working directory" in result:
        print(f"✅ SUCCESS: Path traversal blocked - {result[:80]}...")
    else:
        print(f"❌ FAILED: Path traversal should be blocked")

    # Test 6: Try to list a file (not directory)
    print("\n[Test 6] Trying to list a file instead of directory...")
    result = tool._run("README.md")
    if "not a directory" in result or "Error" in result:
        print(f"✅ SUCCESS: Correctly rejected file - {result[:80]}...")
    else:
        print(f"❌ FAILED: Should reject files")

    # Test 7: List nested directory
    print("\n[Test 7] Listing nested directory 'src/agent'...")
    result = tool._run("src/agent")
    if "Error" in result or "error" in result:
        print(f"❌ FAILED: {result}")
    else:
        print(f"✅ SUCCESS: Listed nested directory")
        print(result)

    # Test 8: Absolute path within CWD
    print("\n[Test 8] Testing absolute path within CWD...")
    abs_path = str(Path.cwd() / "src")
    result = tool._run(abs_path)
    if "Error" in result or "error" in result:
        print(f"❌ FAILED: {result}")
    else:
        print(f"✅ SUCCESS: Listed directory using absolute path")

    # Test 9: LangChain interface test
    print("\n[Test 9] Testing with LangChain invoke method...")
    try:
        result = tool.invoke({"path": "."})
        if "Total:" in result:
            print(f"✅ SUCCESS: LangChain invoke works")
        else:
            print(f"❌ FAILED: Unexpected result from invoke")
    except Exception as e:
        print(f"❌ FAILED: LangChain invoke error - {e}")

    # Test 10: Empty directory (create temp)
    print("\n[Test 10] Testing empty directory...")
    import tempfile
    with tempfile.TemporaryDirectory(dir=".") as tmpdir:
        result = tool._run(tmpdir)
        if "empty" in result.lower():
            print(f"✅ SUCCESS: Correctly reports empty directory")
        else:
            print(f"❌ FAILED: Should report empty directory")

    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_list_directory_tool()
