"""Security tests for file operation tools.

These tests ensure that both read_file and list_directory tools
properly prevent various security attacks and unauthorized access.
"""

import pytest
import tempfile
import os
from pathlib import Path
from src.agent.tools.file_ops import (
    get_read_file_tool,
    get_list_directory_tool,
    validate_path,
    PathValidationError,
)


class TestPathTraversalAttacks:
    """Tests to ensure path traversal attacks are blocked."""

    @pytest.fixture
    def read_tool(self):
        """Get ReadFile tool."""
        return get_read_file_tool()

    @pytest.fixture
    def list_tool(self):
        """Get ListDirectory tool."""
        return get_list_directory_tool()

    def test_simple_parent_directory_read(self, read_tool):
        """Test that ../parent access is blocked for read_file."""
        result = read_tool._run("../")
        assert "Access denied" in result or "outside current working directory" in result

    def test_simple_parent_directory_list(self, list_tool):
        """Test that ../parent access is blocked for list_directory."""
        result = list_tool._run("../")
        assert "Access denied" in result or "outside current working directory" in result

    def test_multiple_parent_traversal_read(self, read_tool):
        """Test that ../../ traversal is blocked for read_file."""
        result = read_tool._run("../../etc/passwd")
        assert "Access denied" in result or "outside current working directory" in result

    def test_multiple_parent_traversal_list(self, list_tool):
        """Test that ../../ traversal is blocked for list_directory."""
        result = list_tool._run("../../etc")
        assert "Access denied" in result or "outside current working directory" in result

    def test_mixed_traversal_read(self, read_tool):
        """Test mixed path traversal like src/../../etc for read_file."""
        result = read_tool._run("src/../../etc/passwd")
        assert "Access denied" in result or "outside current working directory" in result

    def test_mixed_traversal_list(self, list_tool):
        """Test mixed path traversal like src/../../etc for list_directory."""
        result = list_tool._run("src/../../etc")
        assert "Access denied" in result or "outside current working directory" in result

    def test_absolute_path_outside_cwd_read(self, read_tool):
        """Test that absolute paths outside CWD are blocked for read_file."""
        result = read_tool._run("/etc/passwd")
        assert "Access denied" in result or "outside current working directory" in result

    def test_absolute_path_outside_cwd_list(self, list_tool):
        """Test that absolute paths outside CWD are blocked for list_directory."""
        result = list_tool._run("/etc")
        assert "Access denied" in result or "outside current working directory" in result

    def test_home_directory_access_read(self, read_tool):
        """Test that ~/home access is blocked for read_file."""
        result = read_tool._run("~/.bashrc")
        # Should be blocked if ~ expands to outside CWD
        # Will succeed if ~ is treated as literal and file exists in CWD
        if Path("~/.bashrc").expanduser() != Path.cwd() / "~/.bashrc":
            assert "Access denied" in result or "does not exist" in result or "outside" in result

    def test_root_directory_access_list(self, list_tool):
        """Test that root directory access is blocked."""
        result = list_tool._run("/")
        assert "Access denied" in result or "outside current working directory" in result


class TestSymlinkAttacks:
    """Tests to ensure symlink attacks are prevented."""

    @pytest.fixture
    def read_tool(self):
        """Get ReadFile tool."""
        return get_read_file_tool()

    @pytest.fixture
    def list_tool(self):
        """Get ListDirectory tool."""
        return get_list_directory_tool()

    def test_symlink_to_outside_cwd_read(self, read_tool):
        """Test that symlinks pointing outside CWD are blocked for read_file."""
        # Create a symlink to /etc/passwd in CWD
        symlink_path = Path.cwd() / "test_symlink_passwd"

        try:
            if not symlink_path.exists():
                symlink_path.symlink_to("/etc/passwd")

            result = read_tool._run(str(symlink_path))
            assert "Access denied" in result or "outside current working directory" in result

        finally:
            # Cleanup
            if symlink_path.is_symlink():
                symlink_path.unlink()

    def test_symlink_to_outside_cwd_list(self, list_tool):
        """Test that symlinks pointing outside CWD are blocked for list_directory."""
        # Create a symlink to /etc in CWD
        symlink_path = Path.cwd() / "test_symlink_etc"

        try:
            if not symlink_path.exists():
                symlink_path.symlink_to("/etc")

            result = list_tool._run(str(symlink_path))
            assert "Access denied" in result or "outside current working directory" in result

        finally:
            # Cleanup
            if symlink_path.is_symlink():
                symlink_path.unlink()

    def test_symlink_within_cwd_allowed_read(self, read_tool):
        """Test that symlinks within CWD are allowed for read_file."""
        # Create a temp file and symlink in CWD
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir='.', suffix='.txt') as f:
            f.write("test content")
            target_path = Path(f.name)

        symlink_path = Path.cwd() / "test_symlink_safe"

        try:
            if not symlink_path.exists():
                symlink_path.symlink_to(target_path)

            result = read_tool._run(str(symlink_path))
            # Should succeed since both symlink and target are in CWD
            assert "test content" in result or "Access denied" not in result

        finally:
            # Cleanup
            if symlink_path.is_symlink():
                symlink_path.unlink()
            if target_path.exists():
                target_path.unlink()


class TestPathValidationFunction:
    """Tests for the validate_path function directly."""

    def test_path_within_cwd_valid(self):
        """Test that paths within CWD are valid."""
        path = validate_path("README.md", must_exist=True)
        assert path.exists()

    def test_path_outside_cwd_invalid(self):
        """Test that paths outside CWD are invalid."""
        with pytest.raises(PathValidationError, match="outside current working directory"):
            validate_path("/etc/passwd", must_exist=False)

    def test_path_traversal_invalid(self):
        """Test that path traversal is invalid."""
        with pytest.raises(PathValidationError, match="outside current working directory"):
            validate_path("../../etc/passwd", must_exist=False)

    def test_nonexistent_path_with_must_exist(self):
        """Test that non-existent paths fail when must_exist=True."""
        with pytest.raises(PathValidationError, match="does not exist"):
            validate_path("nonexistent_file.txt", must_exist=True)

    def test_nonexistent_path_without_must_exist(self):
        """Test that non-existent paths pass when must_exist=False."""
        # Should not raise if within CWD
        path = validate_path("future_file.txt", must_exist=False)
        assert isinstance(path, Path)

    def test_relative_path_resolved(self):
        """Test that relative paths are resolved correctly."""
        path = validate_path("./README.md", must_exist=True)
        assert path.is_absolute()

    def test_symlink_to_outside_cwd_invalid(self):
        """Test that symlinks to outside CWD are invalid."""
        symlink_path = Path.cwd() / "test_symlink_invalid"

        try:
            if not symlink_path.exists():
                symlink_path.symlink_to("/etc/passwd")

            with pytest.raises(PathValidationError, match="outside current working directory"):
                validate_path(str(symlink_path), must_exist=True)

        finally:
            if symlink_path.is_symlink():
                symlink_path.unlink()


class TestInputValidation:
    """Tests for input validation and sanitization."""

    @pytest.fixture
    def read_tool(self):
        """Get ReadFile tool."""
        return get_read_file_tool()

    @pytest.fixture
    def list_tool(self):
        """Get ListDirectory tool."""
        return get_list_directory_tool()

    def test_empty_string_read(self, read_tool):
        """Test that empty string is rejected for read_file."""
        result = read_tool._run("")
        assert "Error" in result or "empty" in result.lower()

    def test_whitespace_string_read(self, read_tool):
        """Test that whitespace-only string is rejected for read_file."""
        result = read_tool._run("   ")
        # Should report error (case-insensitive check)
        assert "error" in result.lower() or "empty" in result.lower() or "does not exist" in result.lower()

    def test_empty_string_list_uses_default(self, list_tool):
        """Test that empty string defaults to current dir for list_directory."""
        result = list_tool._run("")
        # Should list current directory
        assert "Total:" in result

    def test_null_byte_injection_read(self, read_tool):
        """Test that null byte injection is blocked for read_file."""
        result = read_tool._run("README.md\x00/etc/passwd")
        # Should either be blocked or treated as invalid path (case-insensitive)
        assert "error" in result.lower() or "invalid" in result.lower() or "does not exist" in result.lower()

    def test_newline_injection_read(self, read_tool):
        """Test that newline injection is handled for read_file."""
        result = read_tool._run("README.md\n/etc/passwd")
        # Should either be blocked or treated as invalid path
        assert "Error" in result or "Invalid" in result or "does not exist" in result


class TestPermissionHandling:
    """Tests for permission error handling."""

    @pytest.fixture
    def read_tool(self):
        """Get ReadFile tool."""
        return get_read_file_tool()

    @pytest.fixture
    def list_tool(self):
        """Get ListDirectory tool."""
        return get_list_directory_tool()

    def test_unreadable_file(self, read_tool):
        """Test handling of unreadable files."""
        # Create a file with no read permissions
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir='.', suffix='.txt') as f:
            f.write("test")
            temp_path = Path(f.name)

        try:
            # Remove read permission
            os.chmod(temp_path, 0o000)

            result = read_tool._run(str(temp_path))
            # Should report permission error
            assert "Permission denied" in result or "Error" in result

        finally:
            # Restore permissions and cleanup
            os.chmod(temp_path, 0o644)
            temp_path.unlink()


class TestMultipleToolsSecurity:
    """Tests for security when using multiple tools together."""

    @pytest.fixture
    def read_tool(self):
        """Get ReadFile tool."""
        return get_read_file_tool()

    @pytest.fixture
    def list_tool(self):
        """Get ListDirectory tool."""
        return get_list_directory_tool()

    def test_list_then_read_attack(self, read_tool, list_tool):
        """Test that path validation is consistent across tools."""
        # Try to list parent directory
        list_result = list_tool._run("../")
        assert "Access denied" in list_result or "outside" in list_result

        # Try to read file in parent directory
        read_result = read_tool._run("../README.md")
        assert "Access denied" in read_result or "outside" in read_result

    def test_both_tools_enforce_same_security(self, read_tool, list_tool):
        """Test that both tools have consistent security policies."""
        dangerous_paths = [
            "/etc/passwd",
            "../../etc/passwd",
            "/etc",
            "../../etc",
        ]

        for path in dangerous_paths:
            read_result = read_tool._run(path)
            list_result = list_tool._run(path)

            # Both should block
            assert "Access denied" in read_result or "outside" in read_result
            assert "Access denied" in list_result or "outside" in list_result


class TestSecurityWithValidPaths:
    """Tests to ensure valid operations are not blocked."""

    @pytest.fixture
    def read_tool(self):
        """Get ReadFile tool."""
        return get_read_file_tool()

    @pytest.fixture
    def list_tool(self):
        """Get ListDirectory tool."""
        return get_list_directory_tool()

    def test_read_file_in_cwd(self, read_tool):
        """Test that reading files in CWD is allowed."""
        result = read_tool._run("README.md")
        assert "Access denied" not in result
        assert "outside current working directory" not in result

    def test_read_file_in_subdirectory(self, read_tool):
        """Test that reading files in subdirectories is allowed."""
        result = read_tool._run("src/config/settings.py")
        assert "Access denied" not in result
        assert "outside current working directory" not in result

    def test_list_directory_in_cwd(self, list_tool):
        """Test that listing directories in CWD is allowed."""
        result = list_tool._run(".")
        assert "Access denied" not in result
        assert "outside current working directory" not in result

    def test_list_subdirectory(self, list_tool):
        """Test that listing subdirectories is allowed."""
        result = list_tool._run("src")
        assert "Access denied" not in result
        assert "outside current working directory" not in result

    def test_absolute_path_within_cwd_allowed(self, read_tool, list_tool):
        """Test that absolute paths within CWD are allowed."""
        abs_path = str(Path.cwd() / "README.md")
        result = read_tool._run(abs_path)
        assert "Access denied" not in result

        abs_path = str(Path.cwd() / "src")
        result = list_tool._run(abs_path)
        assert "Access denied" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
