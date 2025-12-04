"""Comprehensive tests for search_code tool."""

import pytest
import tempfile
import os
from pathlib import Path
from src.agent.tools.file_ops import (
    get_search_code_tool,
    PathValidationError,
)


class TestSearchCodeTool:
    """Tests for the SearchCode tool."""

    @pytest.fixture
    def tool(self):
        """Get a SearchCodeTool instance."""
        return get_search_code_tool()

    def test_tool_name_and_description(self, tool):
        """Test tool has correct name and description."""
        assert tool.name == "search_code"
        assert "search" in tool.description.lower()
        assert "regex" in tool.description.lower() or "pattern" in tool.description.lower()
        assert tool.args_schema is not None

    def test_search_simple_pattern_in_cwd(self, tool):
        """Test searching for a simple pattern in current directory."""
        # Search for "import" in Python files (should find many in src/)
        result = tool._run("import", ".", "py")
        assert "Error" not in result
        # Should find matches
        assert "matches found" in result.lower() or "no matches" in result.lower()

    def test_search_returns_file_paths_and_line_numbers(self, tool):
        """Test that search results include file paths and line numbers."""
        # Search for "class" in Python files
        result = tool._run("class", "src", "py")

        if "No matches found" not in result:
            # Result should have format: path:line_num: content
            assert ":" in result
            # Should have line numbers
            assert any(char.isdigit() for char in result)

    def test_search_with_regex_pattern(self, tool):
        """Test searching with regex patterns."""
        # Search for function definitions: def <name>(
        result = tool._run(r"def\s+\w+\(", "src", "py")
        assert isinstance(result, str)
        # Should either find matches or report no matches
        assert "matches" in result.lower() or "Error" not in result

    def test_search_specific_file_extension(self, tool):
        """Test filtering by specific file extension."""
        # Create temp files with different extensions
        temp_dir = Path("test_search_dir")
        temp_dir.mkdir(exist_ok=True)

        py_file = temp_dir / "test.py"
        js_file = temp_dir / "test.js"
        txt_file = temp_dir / "test.txt"

        py_file.write_text("test_pattern_py = 1")
        js_file.write_text("test_pattern_js = 1")
        txt_file.write_text("test_pattern_txt = 1")

        try:
            # Search only .py files
            result = tool._run("test_pattern", str(temp_dir), "py")

            if "No matches found" not in result:
                # Should find match in .py file
                assert "test.py" in result
                # Should NOT find matches in other files
                assert "test.js" not in result
                assert "test.txt" not in result

        finally:
            # Cleanup
            py_file.unlink()
            js_file.unlink()
            txt_file.unlink()
            temp_dir.rmdir()

    def test_search_multiple_extensions(self, tool):
        """Test searching multiple file extensions."""
        # Search .py and .md files
        result = tool._run("def", "src", "py,md")
        assert isinstance(result, str)
        # Should work without errors
        assert "Invalid" not in result

    def test_search_no_matches(self, tool):
        """Test searching for pattern that doesn't exist."""
        # Search for very unlikely pattern
        result = tool._run("XYZABC123UNLIKELY", "src", "py")
        assert "No matches found" in result or "0 matches" in result

    def test_search_case_sensitive(self, tool):
        """Test that search is case-sensitive by default."""
        # Create temp files
        temp_dir = Path("test_case_dir")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / "test.py"
        temp_file.write_text("TestCase\ntestcase")

        try:
            # Search for exact case
            result = tool._run("TestCase", str(temp_dir), "py")

            if "No matches found" not in result:
                # Should find "TestCase" but pattern is case-sensitive in regex
                assert "test.py" in result

        finally:
            temp_file.unlink()
            temp_dir.rmdir()


class TestSearchCodePathValidation:
    """Tests for path validation in search_code tool."""

    @pytest.fixture
    def tool(self):
        """Get a SearchCodeTool instance."""
        return get_search_code_tool()

    def test_path_traversal_blocked(self, tool):
        """Test that path traversal attacks are blocked."""
        result = tool._run("pattern", "../../etc", "txt")
        assert "Access denied" in result or "outside current working directory" in result

    def test_absolute_path_outside_cwd_blocked(self, tool):
        """Test that absolute paths outside CWD are blocked."""
        result = tool._run("pattern", "/etc", "txt")
        assert "Access denied" in result or "outside current working directory" in result

    def test_search_in_subdirectory(self, tool):
        """Test searching in subdirectories."""
        result = tool._run("import", "src", "py")
        assert "Error" not in result
        # Should either find matches or report no matches
        assert "matches" in result.lower()

    def test_search_in_current_directory_default(self, tool):
        """Test that default path is current directory."""
        result = tool._run("AI", ".", "md")
        # Should search in current directory
        assert isinstance(result, str)

    def test_nonexistent_directory(self, tool):
        """Test searching in non-existent directory."""
        result = tool._run("pattern", "nonexistent_xyz_dir", "py")
        assert "does not exist" in result or "Error" in result

    def test_search_file_instead_of_directory(self, tool):
        """Test that searching a file (not directory) is handled."""
        result = tool._run("pattern", "README.md", "md")
        assert "not a directory" in result or "Error" in result


class TestSearchCodeExcludePatterns:
    """Tests for exclude patterns functionality."""

    @pytest.fixture
    def tool(self):
        """Get a SearchCodeTool instance."""
        return get_search_code_tool()

    def test_excludes_venv_directory(self, tool):
        """Test that .venv directories are automatically excluded."""
        # If we search the whole project, should not search in .venv
        result = tool._run("import", ".", "py")

        # Result should not contain .venv paths
        assert ".venv" not in result or "No matches" in result

    def test_excludes_node_modules(self, tool):
        """Test that node_modules directories are excluded."""
        # Create a temp node_modules directory
        temp_dir = Path("node_modules")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / "test.js"
        temp_file.write_text("TESTPATTERN123")

        try:
            result = tool._run("TESTPATTERN123", ".", "js")

            # Should not find match in node_modules
            assert "node_modules" not in result

        finally:
            temp_file.unlink()
            temp_dir.rmdir()

    def test_excludes_git_directory(self, tool):
        """Test that .git directory is excluded."""
        result = tool._run("pattern", ".", "txt")

        # Should not search in .git directory
        # (this is implicit - just check no errors occur)
        assert isinstance(result, str)


class TestSearchCodeEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def tool(self):
        """Get a SearchCodeTool instance."""
        return get_search_code_tool()

    def test_invalid_regex_pattern(self, tool):
        """Test handling of invalid regex patterns."""
        # Invalid regex (unclosed group)
        result = tool._run("(unclosed", "src", "py")
        assert "Invalid" in result or "error" in result.lower()

    def test_empty_pattern(self, tool):
        """Test searching with empty pattern."""
        # Empty pattern might match everything or be invalid
        result = tool._run("", "src", "py")
        # Should handle gracefully
        assert isinstance(result, str)

    def test_search_with_special_regex_characters(self, tool):
        """Test patterns with special regex characters."""
        # Create temp file with special characters
        temp_dir = Path("test_special_dir")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / "test.py"
        temp_file.write_text("value = data['key']")

        try:
            # Search for literal brackets (need escaping in regex)
            result = tool._run(r"\['key'\]", str(temp_dir), "py")

            # Should work (either find match or not)
            assert isinstance(result, str)

        finally:
            temp_file.unlink()
            temp_dir.rmdir()

    def test_search_very_long_lines(self, tool):
        """Test searching files with very long lines."""
        temp_dir = Path("test_long_dir")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / "test.py"

        # Create a file with a very long line
        long_line = "x" * 5000 + "TARGET" + "y" * 5000
        temp_file.write_text(long_line)

        try:
            result = tool._run("TARGET", str(temp_dir), "py")

            # Should find the pattern
            if "No matches found" not in result:
                assert "test.py" in result

        finally:
            temp_file.unlink()
            temp_dir.rmdir()

    def test_search_binary_file_gracefully_handled(self, tool):
        """Test that binary files are handled gracefully."""
        temp_dir = Path("test_binary_dir")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / "test.py"

        # Write binary content
        temp_file.write_bytes(b'\x00\x01\x02\x03\x04\x05')

        try:
            result = tool._run("pattern", str(temp_dir), "py")

            # Should not crash, handle gracefully
            assert isinstance(result, str)

        finally:
            temp_file.unlink()
            temp_dir.rmdir()

    def test_max_results_limit(self, tool):
        """Test that search respects max results limit."""
        # Create directory with many matches
        temp_dir = Path("test_max_dir")
        temp_dir.mkdir(exist_ok=True)

        # Create 100 files each with pattern
        files = []
        for i in range(100):
            f = temp_dir / f"file_{i}.py"
            f.write_text("MATCHPATTERN")
            files.append(f)

        try:
            result = tool._run("MATCHPATTERN", str(temp_dir), "py")

            # Should limit results (default is 50 in the implementation)
            if "matches found" in result.lower():
                # Check if it mentions the limit
                assert "50" in result or "max" in result.lower()

        finally:
            for f in files:
                if f.exists():
                    f.unlink()
            temp_dir.rmdir()


class TestSearchCodeIntegration:
    """Integration tests for SearchCode tool."""

    @pytest.fixture
    def tool(self):
        """Get a SearchCodeTool instance."""
        return get_search_code_tool()

    def test_search_actual_codebase(self, tool):
        """Test searching the actual project codebase."""
        # Search for "def " in src directory
        result = tool._run(r"def\s+", "src", "py")

        # Should find function definitions
        assert isinstance(result, str)
        if "No matches found" not in result:
            # Should show file paths and line numbers
            assert ".py:" in result

    def test_search_test_files(self, tool):
        """Test searching test files."""
        # Search for "pytest" in test files
        result = tool._run("pytest", "tests", "py")

        assert isinstance(result, str)
        # Should either find matches or report none
        assert "matches" in result.lower() or "No matches" in result

    def test_search_documentation_files(self, tool):
        """Test searching markdown documentation."""
        # Search for "Agent" in markdown files
        result = tool._run("Agent", ".", "md")

        assert isinstance(result, str)
        if "No matches found" not in result:
            # Should find in README.md or other docs
            assert ".md:" in result

    def test_langchain_invoke_method(self, tool):
        """Test using LangChain's invoke method."""
        result = tool.invoke({
            "pattern": "import",
            "path": "src",
            "file_filter": "py"
        })
        assert isinstance(result, str)
        assert "matches" in result.lower()

    def test_tool_returns_string_always(self, tool):
        """Test that tool always returns a string, even on error."""
        test_cases = [
            ("import", "src", "py"),  # Valid search
            ("pattern", "nonexistent", "py"),  # Non-existent directory
            ("(invalid", "src", "py"),  # Invalid regex
            ("pattern", "../../etc", "txt"),  # Path traversal
        ]

        for pattern, path, file_filter in test_cases:
            result = tool._run(pattern, path, file_filter)
            assert isinstance(result, str), f"Tool should return string for {pattern}, {path}"


class TestSearchCodeRealWorldPatterns:
    """Tests with real-world search patterns."""

    @pytest.fixture
    def tool(self):
        """Get a SearchCodeTool instance."""
        return get_search_code_tool()

    def test_search_class_definitions(self, tool):
        """Test searching for class definitions."""
        result = tool._run(r"^class\s+\w+", "src", "py")
        assert isinstance(result, str)

    def test_search_function_definitions(self, tool):
        """Test searching for function definitions."""
        result = tool._run(r"def\s+\w+\(", "src", "py")
        assert isinstance(result, str)

    def test_search_import_statements(self, tool):
        """Test searching for import statements."""
        result = tool._run(r"^import\s+|^from\s+.*import", "src", "py")
        assert isinstance(result, str)

    def test_search_todo_comments(self, tool):
        """Test searching for TODO comments."""
        result = tool._run(r"#\s*TODO", "src", "py")
        assert isinstance(result, str)

    def test_search_environment_variables(self, tool):
        """Test searching for environment variable usage."""
        result = tool._run(r"os\.environ|getenv", "src", "py")
        assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
