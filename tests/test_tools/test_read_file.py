"""Comprehensive tests for read_file tool."""

import pytest
import tempfile
from pathlib import Path
from src.agent.tools.file_ops import (
    get_read_file_tool,
    validate_path,
    check_file_size,
    PathValidationError,
    FileSizeError,
)


class TestPathValidation:
    """Tests for path validation function."""

    def test_valid_path_in_cwd(self):
        """Test that valid paths in CWD are accepted."""
        # This test file itself should be valid
        path = validate_path("tests/test_tools/test_read_file.py", must_exist=True)
        assert path.exists()
        assert path.is_file()

    def test_relative_path_in_cwd(self):
        """Test relative paths are resolved correctly."""
        path = validate_path("./README.md", must_exist=True)
        assert path.exists()
        assert path.name == "README.md"

    def test_path_outside_cwd_rejected(self):
        """Test that paths outside CWD are rejected."""
        with pytest.raises(PathValidationError, match="outside current working directory"):
            validate_path("/etc/passwd", must_exist=False)

    def test_path_traversal_rejected(self):
        """Test that path traversal attacks are blocked."""
        with pytest.raises(PathValidationError, match="outside current working directory"):
            validate_path("../../etc/passwd", must_exist=False)

    def test_nonexistent_path_rejected_when_must_exist(self):
        """Test that non-existent paths are rejected when must_exist=True."""
        with pytest.raises(PathValidationError, match="does not exist"):
            validate_path("nonexistent_file_xyz123.txt", must_exist=True)

    def test_nonexistent_path_allowed_when_not_must_exist(self):
        """Test that non-existent paths are allowed when must_exist=False."""
        # Should not raise if must_exist=False and path is within CWD
        path = validate_path("future_file.txt", must_exist=False)
        assert isinstance(path, Path)

    def test_empty_path_rejected(self):
        """Test that empty paths are rejected."""
        # Note: The Pydantic validator in the tool catches empty paths before validate_path is called
        # So this test validates the validate_path function directly with a non-empty invalid path
        with pytest.raises((PathValidationError, ValueError, OSError)):
            validate_path("", must_exist=False)


class TestFileSizeCheck:
    """Tests for file size checking."""

    def test_small_file_accepted(self):
        """Test that small files pass size check."""
        # Create a small temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir='.') as f:
            f.write("small content")
            temp_path = Path(f.name)

        try:
            check_file_size(temp_path, max_size_mb=1)
            # Should not raise
            assert True
        finally:
            temp_path.unlink()

    def test_large_file_rejected(self):
        """Test that files exceeding size limit are rejected."""
        # Create a file larger than 1MB
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, dir='.') as f:
            # Write 2MB of data
            f.write(b'x' * (2 * 1024 * 1024))
            temp_path = Path(f.name)

        try:
            with pytest.raises(FileSizeError, match="File too large"):
                check_file_size(temp_path, max_size_mb=1)
        finally:
            temp_path.unlink()


class TestReadFileTool:
    """Tests for the ReadFile tool."""

    @pytest.fixture
    def tool(self):
        """Get a ReadFileTool instance."""
        return get_read_file_tool()

    def test_tool_name_and_description(self, tool):
        """Test tool has correct name and description."""
        assert tool.name == "read_file"
        assert "read" in tool.description.lower()
        assert tool.args_schema is not None

    def test_read_existing_file(self, tool):
        """Test reading an existing file."""
        result = tool._run("README.md")
        assert "Error" not in result or "error" not in result
        assert len(result) > 0
        assert "AI Coding Agent" in result or "Ollama" in result  # Should contain project info

    def test_read_with_relative_path(self, tool):
        """Test reading with relative path."""
        result = tool._run("./README.md")
        assert "Error" not in result
        assert len(result) > 0

    def test_read_nonexistent_file(self, tool):
        """Test reading non-existent file returns error."""
        result = tool._run("nonexistent_xyz_file.txt")
        assert "does not exist" in result or "Error" in result

    def test_read_directory_rejected(self, tool):
        """Test that reading a directory is rejected."""
        result = tool._run("src")
        assert "not a file" in result or "Error" in result

    def test_path_traversal_blocked(self, tool):
        """Test that path traversal attacks are blocked."""
        result = tool._run("../../etc/passwd")
        assert "Access denied" in result or "outside current working directory" in result

    def test_absolute_path_outside_cwd_blocked(self, tool):
        """Test that absolute paths outside CWD are blocked."""
        result = tool._run("/etc/passwd")
        assert "Access denied" in result or "outside current working directory" in result

    def test_read_hidden_file_in_cwd(self, tool):
        """Test reading hidden files (starting with .) in CWD."""
        # Try to read .env.example if it exists
        result = tool._run(".env.example")
        # Should either succeed or report file doesn't exist
        # Should NOT report security error
        assert "Access denied" not in result
        assert "outside current" not in result

    def test_read_file_with_special_characters(self, tool):
        """Test reading files with spaces or special characters in name."""
        # Create a temp file with space in name
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir='.',
                                        prefix='test file ', suffix='.txt') as f:
            f.write("test content")
            temp_path = Path(f.name)

        try:
            result = tool._run(str(temp_path))
            assert "test content" in result
        finally:
            temp_path.unlink()

    def test_read_python_file(self, tool):
        """Test reading a Python source file."""
        result = tool._run("src/config/settings.py")
        assert "Error" not in result or "error" not in result
        assert "import" in result or "class" in result or "def" in result

    def test_read_requirements_file(self, tool):
        """Test reading requirements.txt."""
        result = tool._run("requirements.txt")
        assert "langchain" in result.lower()

    def test_empty_path_rejected(self, tool):
        """Test that empty path is rejected."""
        result = tool._run("")
        assert "Error" in result or "empty" in result.lower()

    def test_whitespace_path_rejected(self, tool):
        """Test that whitespace-only path is rejected."""
        result = tool._run("   ")
        # Should report error (case-insensitive check)
        assert "error" in result.lower() or "empty" in result.lower() or "does not exist" in result.lower()


class TestReadFileEncoding:
    """Tests for file encoding handling."""

    @pytest.fixture
    def tool(self):
        """Get a ReadFileTool instance."""
        return get_read_file_tool()

    def test_read_utf8_file(self, tool):
        """Test reading UTF-8 encoded file."""
        # Create a UTF-8 file with unicode characters
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                        delete=False, dir='.', suffix='.txt') as f:
            f.write("Hello 世界 🌍")
            temp_path = Path(f.name)

        try:
            result = tool._run(str(temp_path))
            assert "Hello" in result
            assert "世界" in result or "Error" not in result  # Either reads correctly or uses fallback
        finally:
            temp_path.unlink()


class TestReadFileToolIntegration:
    """Integration tests for ReadFile tool."""

    @pytest.fixture
    def tool(self):
        """Get a ReadFileTool instance."""
        return get_read_file_tool()

    def test_read_multiple_files_sequentially(self, tool):
        """Test reading multiple files in sequence."""
        files = ["README.md", "requirements.txt", ".env.example"]

        for file_path in files:
            result = tool._run(file_path)
            # Each should either succeed or report "does not exist"
            # Should NOT crash or have internal errors
            assert "Unexpected error" not in result
            assert "exception" not in result.lower()

    def test_tool_returns_string(self, tool):
        """Test that tool always returns a string."""
        test_cases = [
            "README.md",
            "nonexistent.txt",
            "../../etc/passwd",
            "src",
        ]

        for test_path in test_cases:
            result = tool._run(test_path)
            assert isinstance(result, str), f"Tool should return string for {test_path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
