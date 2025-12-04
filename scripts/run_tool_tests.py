"""Run all file operation tool tests with detailed reporting."""

import subprocess
import sys
from pathlib import Path


def run_test_suite(name, test_file):
    """Run a test suite and return the result."""
    print(f"\n{'='*70}")
    print(f"Test Suite: {name}")
    print('='*70)

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v"],
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {name}: {e}")
        return False


def main():
    """Run all tool tests."""
    print("="*70)
    print("File Operation Tools - Comprehensive Test Suite")
    print("="*70)

    # Test suites to run
    test_suites = [
        ("ReadFile Tool", "tests/test_tools/test_read_file.py"),
        ("ListDirectory Tool", "tests/test_tools/test_list_directory.py"),
        ("Security Tests (Both Tools)", "tests/test_tools/test_file_ops_security.py"),
    ]

    results = {}

    # Run each test suite
    for name, test_file in test_suites:
        results[name] = run_test_suite(name, test_file)

    # Print summary
    print("\n" + "="*70)
    print("Test Results Summary")
    print("="*70)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        color = "\033[0;32m" if passed else "\033[0;31m"
        reset = "\033[0m"
        print(f"{color}{status}{reset} - {name}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
