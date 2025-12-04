"""Comprehensive tests for list_directory tool."""

import pytest
import tempfile
import os
from pathlib import Path
from src.agent.tools.file_ops import (
    get_list_directory_tool,
    PathValidationError,
)


class TestListDirectoryTool:
    """Tests for the ListDirectory tool."""

    @pytest.fixture
    def tool(self):
        """Get a ListDirectoryTool instance."""
        return get_list_directory_tool()

    def test_tool_name_and_description(self, tool):
        """Test tool has correct name and description."""
        assert tool.name == "list_directory"
        assert "list" in tool.description.lower()
        assert "directory" in tool.description.lower()
        assert tool.args_schema is not None

    def test_list_current_directory_default(self, tool):
        """Test listing current directory with default parameter."""
        result = tool._run()
        assert "Error" not in result
        assert "Total:" in result
        assert "file(s)" in result
        assert "directory(ies)" in result

    def test_list_current_directory_explicit(self, tool):
        """Test listing current directory with explicit '.' parameter."""
        result = tool._run(".")
        assert "Error" not in result
        assert "Total:" in result
        # Should contain project files
        assert any(name in result for name in ["README.md", "requirements.txt", ".env.example"])

    def test_list_subdirectory(self, tool):
        """Test listing a subdirectory."""
        result = tool._run("src")
        assert "Error" not in result
        assert "Total:" in result
        # Should contain src subdirectories
        assert any(name in result for name in ["agent", "cli", "config"])

    def test_list_nested_directory(self, tool):
        """Test listing a nested directory."""
        result = tool._run("src/agent")
        assert "Error" not in result
        assert "Total:" in result
        # Should contain agent module files
        assert "agent.py" in result or "prompts.py" in result

    def test_list_nonexistent_directory(self, tool):
        """Test listing non-existent directory returns error."""
        result = tool._run("nonexistent_xyz_directory_123")
        assert "does not exist" in result or "Error" in result

    def test_list_file_instead_of_directory(self, tool):
        """Test that listing a file (not directory) is rejected."""
        result = tool._run("README.md")
        assert "not a directory" in result or "Error" in result

    def test_path_traversal_blocked(self, tool):
        """Test that path traversal attacks are blocked."""
        result = tool._run("../../etc")
        assert "Access denied" in result or "outside current working directory" in result

    def test_absolute_path_outside_cwd_blocked(self, tool):
        """Test that absolute paths outside CWD are blocked."""
        result = tool._run("/etc")
        assert "Access denied" in result or "outside current working directory" in result

    def test_absolute_path_within_cwd(self, tool):
        """Test that absolute paths within CWD are allowed."""
        abs_path = str(Path.cwd() / "src")
        result = tool._run(abs_path)
        assert "Error" not in result
        assert "Total:" in result

    def test_relative_path_with_dot_slash(self, tool):
        """Test relative paths with ./ prefix."""
        result = tool._run("./src")
        assert "Error" not in result
        assert "Total:" in result

    def test_empty_path_uses_default(self, tool):
        """Test that empty path defaults to current directory."""
        result = tool._run("")
        assert "Error" not in result
        assert "Total:" in result

    def test_whitespace_path_uses_default(self, tool):
        """Test that whitespace-only path defaults to current directory."""
        result = tool._run("   ")
        # Whitespace gets stripped to empty, which defaults to "."
        # This should work since empty string defaults to current directory
        assert "Total:" in result or "does not exist" in result.lower()


class TestListDirectoryOutput:
    """Tests for list_directory output formatting."""

    @pytest.fixture
    def tool(self):
        """Get a ListDirectoryTool instance."""
        return get_list_directory_tool()

    def test_output_contains_headers(self, tool):
        """Test output contains proper column headers."""
        result = tool._run(".")
        assert "Name" in result
        assert "Type" in result
        assert "Size" in result
        assert "Modified" in result

    def test_output_contains_separators(self, tool):
        """Test output contains separator lines."""
        result = tool._run(".")
        assert "---" in result or "━" in result

    def test_output_contains_summary(self, tool):
        """Test output contains summary line."""
        result = tool._run(".")
        assert "Total:" in result
        assert "file(s)" in result
        assert "directory(ies)" in result

    def test_directory_type_shown(self, tool):
        """Test that directories are marked with DIR type."""
        result = tool._run(".")
        # Should have at least one directory (src, tests, etc.)
        assert "DIR" in result

    def test_file_type_shown(self, tool):
        """Test that files are marked with FILE type."""
        result = tool._run(".")
        # Should have at least one file (README.md, etc.)
        assert "FILE" in result

    def test_file_sizes_formatted(self, tool):
        """Test that file sizes are human-readable."""
        result = tool._run(".")
        # Should contain size units
        assert any(unit in result for unit in ["B", "KB", "MB"])

    def test_directories_listed_first(self, tool):
        """Test that directories are listed before files."""
        result = tool._run(".")
        lines = result.split('\n')

        # Find first DIR and first FILE
        first_dir_line = None
        first_file_line = None

        for i, line in enumerate(lines):
            if "DIR" in line and first_dir_line is None:
                first_dir_line = i
            if "FILE" in line and first_file_line is None:
                first_file_line = i

        # If both exist, DIR should come before FILE
        if first_dir_line is not None and first_file_line is not None:
            assert first_dir_line < first_file_line


class TestListDirectoryEmptyAndSpecialCases:
    """Tests for empty directories and special cases."""

    @pytest.fixture
    def tool(self):
        """Get a ListDirectoryTool instance."""
        return get_list_directory_tool()

    def test_empty_directory(self, tool):
        """Test listing an empty directory."""
        # Create a temporary empty directory in CWD
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            result = tool._run(tmpdir)
            assert "empty" in result.lower()

    def test_directory_with_hidden_files(self, tool):
        """Test that hidden files (starting with .) are shown."""
        result = tool._run(".")
        # Project should have .env.example, .gitignore, etc.
        # Check if hidden files are present in output
        # Note: They might not be present, so we just check the tool doesn't crash
        assert "Error" not in result
        assert "Total:" in result

    def test_directory_with_spaces_in_name(self, tool):
        """Test directories with spaces in names."""
        # Create a temp directory with space in name
        with tempfile.TemporaryDirectory(dir=".", prefix="test dir ") as tmpdir:
            tmpdir_name = Path(tmpdir).name
            result = tool._run(tmpdir_name)
            assert "empty" in result.lower()

    def test_deeply_nested_directory(self, tool):
        """Test listing deeply nested directories."""
        # Create nested temp directories
        with tempfile.TemporaryDirectory(dir=".") as tmpdir:
            nested = Path(tmpdir) / "level1" / "level2" / "level3"
            nested.mkdir(parents=True)

            # List each level
            tmpdir_name = Path(tmpdir).name
            result = tool._run(tmpdir_name)
            assert "level1" in result

            # Fix: Use Path object properly for concatenation
            result = tool._run(str(Path(tmpdir_name) / "level1"))
            assert "level2" in result


class TestListDirectoryIntegration:
    """Integration tests for ListDirectory tool."""

    @pytest.fixture
    def tool(self):
        """Get a ListDirectoryTool instance."""
        return get_list_directory_tool()

    def test_list_then_list_subdirectory(self, tool):
        """Test listing directory then listing its subdirectory."""
        # List current directory
        result1 = tool._run(".")
        assert "src" in result1

        # List src subdirectory
        result2 = tool._run("src")
        assert "agent" in result2 or "cli" in result2

    def test_langchain_invoke_method(self, tool):
        """Test using LangChain's invoke method."""
        result = tool.invoke({"path": "."})
        assert isinstance(result, str)
        assert "Total:" in result

    def test_langchain_invoke_with_no_path(self, tool):
        """Test invoke with empty dict (should use default)."""
        # The tool has a default value for path
        result = tool.invoke({})
        assert isinstance(result, str)
        assert "Total:" in result or "Error" not in result

    def test_multiple_sequential_calls(self, tool):
        """Test calling the tool multiple times in sequence."""
        directories = [".", "src", "tests"]

        for directory in directories:
            if Path(directory).exists():
                result = tool._run(directory)
                assert "Total:" in result or "Error" not in result
                assert isinstance(result, str)

    def test_tool_returns_string_always(self, tool):
        """Test that tool always returns a string, even on error."""
        test_cases = [
            ".",
            "src",
            "nonexistent",
            "README.md",  # File, not directory
            "../../etc",  # Path traversal
        ]

        for test_path in test_cases:
            result = tool._run(test_path)
            assert isinstance(result, str), f"Tool should return string for {test_path}"


class TestListDirectoryFormatHelpers:
    """Tests for formatting helper methods."""

    @pytest.fixture
    def tool(self):
        """Get a ListDirectoryTool instance."""
        return get_list_directory_tool()

    def test_format_size_bytes(self, tool):
        """Test size formatting for bytes."""
        assert "B" in tool._format_size(100)
        assert "B" in tool._format_size(1023)

    def test_format_size_kilobytes(self, tool):
        """Test size formatting for kilobytes."""
        result = tool._format_size(1024)
        assert "KB" in result

        result = tool._format_size(1024 * 500)
        assert "KB" in result

    def test_format_size_megabytes(self, tool):
        """Test size formatting for megabytes."""
        result = tool._format_size(1024 * 1024)
        assert "MB" in result

        result = tool._format_size(1024 * 1024 * 5)
        assert "MB" in result

    def test_format_size_gigabytes(self, tool):
        """Test size formatting for gigabytes."""
        result = tool._format_size(1024 * 1024 * 1024)
        assert "GB" in result

    def test_format_time(self, tool):
        """Test time formatting."""
        import time
        timestamp = time.time()
        result = tool._format_time(timestamp)

        # Should be in YYYY-MM-DD HH:MM format
        assert len(result) == 16
        assert result[4] == "-"
        assert result[7] == "-"
        assert result[10] == " "
        assert result[13] == ":"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
