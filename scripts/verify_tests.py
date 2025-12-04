"""Quick verification script to check if test fixes resolved all issues."""

import subprocess
import sys


def run_tests():
    """Run the test suite and report results."""
    print("=" * 70)
    print("Running Fixed Test Suite")
    print("=" * 70)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_tools/", "-v", "--tb=short"],
            capture_output=False
        )

        print("\n" + "=" * 70)
        if result.returncode == 0:
            print("✅ All tests passed!")
            print("=" * 70)
            return 0
        else:
            print("❌ Some tests still failing. Please review output above.")
            print("=" * 70)
            return 1

    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
