"""Manual testing script for read_file tool - test edge cases interactively."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.tools.file_ops import get_read_file_tool


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def test_case(description, path, tool):
    """Run a single test case."""
    print(f"\n📝 {description}")
    print(f"   Path: {path}")
    result = tool._run(path)

    if len(result) > 200:
        print(f"   Result: {result[:200]}...")
    else:
        print(f"   Result: {result}")

    return result


def main():
    """Run manual tests."""
    tool = get_read_file_tool()

    print_section("ReadFile Tool - Manual Regression Testing")

    # Valid cases
    print_section("VALID CASES (Should Succeed)")

    test_case("1. Read README.md", "README.md", tool)
    test_case("2. Read with relative path ./", "./README.md", tool)
    test_case("3. Read requirements.txt", "requirements.txt", tool)
    test_case("4. Read Python source file", "src/config/settings.py", tool)
    test_case("5. Read hidden file", ".env.example", tool)
    test_case("6. Read nested file", "src/agent/agent.py", tool)

    # Current working directory
    cwd = Path.cwd()
    abs_path = str(cwd / "README.md")
    test_case("7. Read with absolute path in CWD", abs_path, tool)

    # Security cases
    print_section("SECURITY CASES (Should Be Blocked)")

    test_case("8. Path traversal attack ../", "../../etc/passwd", tool)
    test_case("9. Absolute path outside CWD", "/etc/passwd", tool)
    test_case("10. Another traversal variant", "../../../etc/hosts", tool)

    # Error cases
    print_section("ERROR CASES (Should Handle Gracefully)")

    test_case("11. Non-existent file", "nonexistent_file_12345.txt", tool)
    test_case("12. Directory instead of file", "src", tool)
    test_case("13. Empty path", "", tool)
    test_case("14. Whitespace path", "   ", tool)
    test_case("15. Invalid path characters", "file\x00name.txt", tool)

    # Edge cases
    print_section("EDGE CASES")

    # Try to read a large file (if exists)
    large_files = ["venv/lib/python3.13/site-packages/torch/_C.cpython-313-x86_64-linux-gnu.so"]
    for lf in large_files:
        if Path(lf).exists():
            test_case(f"16. Large file test", lf, tool)
            break

    # LangChain interface test
    print_section("LANGCHAIN INTERFACE TEST")

    print("\n📝 Testing with LangChain invoke method")
    try:
        result = tool.invoke({"path": "README.md"})
        print(f"   ✅ LangChain invoke works: {len(result)} chars read")
    except Exception as e:
        print(f"   ❌ LangChain invoke failed: {e}")

    # Tool metadata
    print_section("TOOL METADATA")

    print(f"\n📝 Tool Name: {tool.name}")
    print(f"📝 Tool Description: {tool.description}")
    print(f"📝 Args Schema: {tool.args_schema}")
    print(f"📝 Args Schema Fields: {tool.args_schema.model_fields if hasattr(tool.args_schema, 'model_fields') else 'N/A'}")

    print_section("TESTING COMPLETE")
    print("\n✅ All manual tests completed. Review results above.")
    print("   - Valid cases should return file contents")
    print("   - Security cases should be blocked with 'Access denied'")
    print("   - Error cases should return descriptive error messages")
    print("   - No crashes or unexpected errors should occur\n")


if __name__ == "__main__":
    main()
