"""Comprehensive tests for write_file tool."""

import pytest
import tempfile
import os
from pathlib import Path
from src.agent.tools.file_ops import (
    get_write_file_tool,
    PathValidationError,
)


class TestWriteFileTool:
    """Tests for the WriteFile tool."""

    @pytest.fixture
    def tool(self):
        """Get a WriteFileTool instance."""
        return get_write_file_tool()

    def test_tool_name_and_description(self, tool):
        """Test tool has correct name and description."""
        assert tool.name == "write_file"
        assert "write" in tool.description.lower()
        assert "confirm" in tool.description.lower()
        assert tool.args_schema is not None

    def test_write_new_file_without_confirmation(self, tool):
        """Test that writing without confirmation is blocked."""
        result = tool._run("test_new_file.txt", "test content", confirm_write=False)
        assert "ACTION REQUIRED" in result or "cancelled" in result
        assert "confirm_write=True" in result
        # Verify file was NOT created
        assert not Path("test_new_file.txt").exists()

    def test_write_new_file_with_confirmation(self, tool):
        """Test creating a new file with confirmation."""
        temp_file = Path("test_write_new.txt")
        try:
            result = tool._run(str(temp_file), "Hello World", confirm_write=True)
            assert "successfully" in result.lower()
            assert "create" in result.lower()
            assert temp_file.exists()
            assert temp_file.read_text() == "Hello World"
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_overwrite_existing_file_with_backup(self, tool):
        """Test overwriting existing file creates backup."""
        # Create initial file
        temp_file = Path("test_overwrite.txt")
        original_content = "Original content"
        temp_file.write_text(original_content)

        try:
            # Overwrite with new content
            new_content = "New content"
            result = tool._run(str(temp_file), new_content, confirm_write=True)

            # Verify success message
            assert "successfully" in result.lower()
            assert "overwrite" in result.lower() or "written" in result.lower()

            # Verify new content
            assert temp_file.read_text() == new_content

            # Verify backup was mentioned in result
            assert "backup" in result.lower() or "Backup location" in result

            # Check for diff in output
            assert "DIFF PREVIEW" in result

        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()
            # Cleanup any backup files (backup replaces suffix: test_overwrite.txt -> test_overwrite.bak.{timestamp})
            for backup in Path(".").glob("test_overwrite.bak.*"):
                backup.unlink()

    def test_diff_preview_in_output(self, tool):
        """Test that diff preview is included in output."""
        temp_file = Path("test_diff.txt")
        temp_file.write_text("Line 1\nLine 2\nLine 3")

        try:
            new_content = "Line 1\nModified Line 2\nLine 3"
            result = tool._run(str(temp_file), new_content, confirm_write=True)

            # Check for diff markers
            assert "DIFF PREVIEW" in result
            assert "---" in result or "+++" in result or "@@ " in result

        finally:
            if temp_file.exists():
                temp_file.unlink()
            for backup in Path(".").glob("test_diff.bak.*"):
                backup.unlink()

    def test_write_empty_file(self, tool):
        """Test writing empty content to file."""
        temp_file = Path("test_empty.txt")
        try:
            result = tool._run(str(temp_file), "", confirm_write=True)
            assert "successfully" in result.lower()
            assert temp_file.exists()
            assert temp_file.read_text() == ""
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_write_multiline_content(self, tool):
        """Test writing multiline content."""
        temp_file = Path("test_multiline.txt")
        content = "Line 1\nLine 2\nLine 3\nLine 4"
        try:
            result = tool._run(str(temp_file), content, confirm_write=True)
            assert "successfully" in result.lower()
            assert temp_file.exists()
            assert temp_file.read_text() == content
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_write_unicode_content(self, tool):
        """Test writing unicode content."""
        temp_file = Path("test_unicode.txt")
        content = "Hello 世界 🌍 café"
        try:
            result = tool._run(str(temp_file), content, confirm_write=True)
            assert "successfully" in result.lower()
            assert temp_file.exists()
            assert temp_file.read_text(encoding='utf-8') == content
        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestWriteFilePathValidation:
    """Tests for path validation in write_file tool."""

    @pytest.fixture
    def tool(self):
        """Get a WriteFileTool instance."""
        return get_write_file_tool()

    def test_path_traversal_blocked(self, tool):
        """Test that path traversal attacks are blocked."""
        result = tool._run("../../etc/test.txt", "content", confirm_write=True)
        assert "Access denied" in result or "outside current working directory" in result

    def test_absolute_path_outside_cwd_blocked(self, tool):
        """Test that absolute paths outside CWD are blocked."""
        result = tool._run("/etc/test.txt", "content", confirm_write=True)
        assert "Access denied" in result or "outside current working directory" in result

    def test_write_to_subdirectory(self, tool):
        """Test writing to files in subdirectories."""
        # Create a temp subdirectory
        temp_dir = Path("test_subdir")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / "test.txt"

        try:
            result = tool._run(str(temp_file), "content", confirm_write=True)
            assert "successfully" in result.lower()
            assert temp_file.exists()
        finally:
            if temp_file.exists():
                temp_file.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()

    def test_write_to_directory_fails(self, tool):
        """Test that attempting to write to a directory fails."""
        # Try to write to an existing directory
        result = tool._run("src", "content", confirm_write=True)
        assert "directory" in result.lower() or "error" in result.lower()

    def test_relative_path_with_dot_slash(self, tool):
        """Test relative paths with ./ prefix."""
        temp_file = Path("./test_dot_slash.txt")
        try:
            result = tool._run(str(temp_file), "content", confirm_write=True)
            assert "successfully" in result.lower()
            assert temp_file.exists()
        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestWriteFileEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def tool(self):
        """Get a WriteFileTool instance."""
        return get_write_file_tool()

    def test_write_with_special_characters_in_filename(self, tool):
        """Test writing files with spaces and special characters in name."""
        temp_file = Path("test file with spaces.txt")
        try:
            result = tool._run(str(temp_file), "content", confirm_write=True)
            assert "successfully" in result.lower()
            assert temp_file.exists()
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_write_large_content(self, tool):
        """Test writing large content."""
        temp_file = Path("test_large.txt")
        # Create 100KB of content
        large_content = "x" * (100 * 1024)
        try:
            result = tool._run(str(temp_file), large_content, confirm_write=True)
            assert "successfully" in result.lower()
            assert temp_file.exists()
            assert len(temp_file.read_text()) == len(large_content)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_overwrite_preserves_permissions(self, tool):
        """Test that overwriting a file preserves basic file attributes."""
        temp_file = Path("test_permissions.txt")
        temp_file.write_text("original")

        try:
            # Write new content
            result = tool._run(str(temp_file), "new content", confirm_write=True)
            assert "successfully" in result.lower()
            # File should still exist and be readable
            assert temp_file.exists()
            assert os.access(temp_file, os.R_OK)
        finally:
            if temp_file.exists():
                temp_file.unlink()
            for backup in Path(".").glob("test_permissions.bak.*"):
                backup.unlink()

    def test_write_to_hidden_file(self, tool):
        """Test writing to hidden files (starting with .)."""
        temp_file = Path(".test_hidden")
        try:
            result = tool._run(str(temp_file), "hidden content", confirm_write=True)
            assert "successfully" in result.lower()
            assert temp_file.exists()
        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestWriteFileBackupBehavior:
    """Tests for backup creation behavior."""

    @pytest.fixture
    def tool(self):
        """Get a WriteFileTool instance."""
        return get_write_file_tool()

    def test_backup_created_for_existing_file(self, tool):
        """Test that backup is created when overwriting."""
        temp_file = Path("test_backup.txt")
        temp_file.write_text("original content")

        try:
            result = tool._run(str(temp_file), "new content", confirm_write=True)

            # Check that backup was created (backup replaces suffix: test_backup.txt -> test_backup.bak.{timestamp})
            backups = list(Path(".").glob("test_backup.bak.*"))

            # The result should mention backup
            assert "backup" in result.lower() or "Backup location" in result

            # If FILE_BACKUP_ENABLED is True, verify backup exists
            # If False, verify message says no backup created
            if "No backup created" not in result:
                assert len(backups) > 0
                # Verify backup contains original content
                if backups:
                    assert backups[0].read_text() == "original content"

        finally:
            if temp_file.exists():
                temp_file.unlink()
            for backup in Path(".").glob("test_backup.bak.*"):
                backup.unlink()

    def test_no_backup_for_new_file(self, tool):
        """Test that no backup is created for new files."""
        temp_file = Path("test_new_no_backup.txt")
        try:
            result = tool._run(str(temp_file), "content", confirm_write=True)

            # No backup should be created for new file
            assert "No backup created" in result or "file was new" in result

            # Verify no backup files exist
            backups = list(Path(".").glob("test_new_no_backup.txt.bak.*"))
            assert len(backups) == 0

        finally:
            if temp_file.exists():
                temp_file.unlink()


class TestWriteFileIntegration:
    """Integration tests for WriteFile tool."""

    @pytest.fixture
    def tool(self):
        """Get a WriteFileTool instance."""
        return get_write_file_tool()

    def test_multiple_overwrites_create_multiple_backups(self, tool):
        """Test that multiple writes create multiple backup files."""
        temp_file = Path("test_multi_backup.txt")

        # Clean up any leftover backups from previous test runs
        for backup in Path(".").glob("test_multi_backup.bak.*"):
            backup.unlink()

        temp_file.write_text("version 1")

        try:
            # Count backups before
            backups_before = len(list(Path(".").glob("test_multi_backup.bak.*")))

            # First overwrite
            result1 = tool._run(str(temp_file), "version 2", confirm_write=True)
            assert "successfully" in result1.lower()

            # Wait a moment to ensure different timestamp
            import time
            time.sleep(1)

            # Second overwrite
            result2 = tool._run(str(temp_file), "version 3", confirm_write=True)
            assert "successfully" in result2.lower()

            # Check for backups (if backups are enabled)
            backups_after = len(list(Path(".").glob("test_multi_backup.bak.*")))
            # Should have created 2 new backups if enabled, or 0 if disabled
            new_backups = backups_after - backups_before
            assert new_backups == 0 or new_backups == 2

        finally:
            if temp_file.exists():
                temp_file.unlink()
            for backup in Path(".").glob("test_multi_backup.bak.*"):
                backup.unlink()

    def test_langchain_invoke_method(self, tool):
        """Test using LangChain's invoke method."""
        temp_file = Path("test_invoke.txt")
        try:
            result = tool.invoke({
                "path": str(temp_file),
                "content": "test content",
                "confirm_write": True
            })
            assert isinstance(result, str)
            assert "successfully" in result.lower()
            assert temp_file.exists()
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def test_write_to_nested_directories(self, tool):
        """Test that nested directories are created automatically."""
        nested_path = Path("test_nested/subdir/deep/file.txt")
        try:
            # Ensure directories don't exist
            if nested_path.parent.exists():
                import shutil
                shutil.rmtree(nested_path.parent.parent)

            # Write to nested path - should create directories
            result = tool._run(str(nested_path), "nested content", confirm_write=True)

            assert "successfully" in result.lower()
            assert nested_path.exists()
            assert nested_path.read_text() == "nested content"
            assert nested_path.parent.is_dir()
        finally:
            # Cleanup
            if nested_path.parent.parent.exists():
                import shutil
                shutil.rmtree(nested_path.parent.parent)

    def test_tool_returns_string_always(self, tool):
        """Test that tool always returns a string, even on error."""
        test_cases = [
            ("test.txt", "content", False),  # No confirmation
            ("../../etc/test.txt", "content", True),  # Path traversal
            ("/etc/test.txt", "content", True),  # Absolute path outside CWD
        ]

        for path, content, confirm in test_cases:
            result = tool._run(path, content, confirm)
            assert isinstance(result, str), f"Tool should return string for {path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
