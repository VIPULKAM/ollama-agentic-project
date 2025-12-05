"""Regression tests for Quick Wins Phase 1 features.

Run these tests to verify all Quick Win features work correctly:
    pytest tests/test_quick_wins_regression.py -v
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from src.agent.agent import CodingAgent
from src.agent.tools.smart_file_ops import get_update_file_section_tool


class TestStreamingResponses:
    """Test streaming response functionality."""

    def test_ask_stream_returns_generator(self):
        """Verify ask_stream returns a generator."""
        agent = CodingAgent()
        result = agent.ask_stream("Hello")
        assert hasattr(result, '__iter__'), "ask_stream should return an iterator"

    def test_stream_contains_response_type(self):
        """Verify stream yields updates with 'type' field."""
        agent = CodingAgent()
        agent.clear_history()

        updates = list(agent.ask_stream("What is 2+2?"))

        assert len(updates) > 0, "Should yield at least one update"

        for update in updates:
            assert 'type' in update, "Each update should have 'type' field"
            assert 'content' in update, "Each update should have 'content' field"
            assert update['type'] in ['tool_start', 'tool_end', 'response', 'error'], \
                f"Invalid update type: {update['type']}"

    def test_stream_with_tool_execution(self):
        """Verify streaming shows tool execution."""
        agent = CodingAgent()
        agent.clear_history()

        # Create a test file
        test_file = Path("test_streaming.txt")
        test_file.write_text("test content")

        try:
            updates = list(agent.ask_stream("Read test_streaming.txt"))

            # Should have tool_start for read_file
            tool_starts = [u for u in updates if u['type'] == 'tool_start']
            assert len(tool_starts) > 0, "Should show tool execution started"

            # Should have a response
            responses = [u for u in updates if u['type'] == 'response']
            assert len(responses) > 0, "Should have final response"

        finally:
            if test_file.exists():
                test_file.unlink()


class TestSmartFileChunking:
    """Test update_file_section tool."""

    def setup_method(self):
        """Create test file for each test."""
        self.test_file = Path("test_sections_regression.md")
        self.test_content = """# Test Document

## Section 1
This is section 1 content.
Some more text here.

## Section 2
This is section 2 content.
Middle section data.

## Section 3
This is section 3 content.
Final section.
"""
        self.test_file.write_text(self.test_content)
        self.tool = get_update_file_section_tool()

    def teardown_method(self):
        """Clean up test files."""
        if self.test_file.exists():
            self.test_file.unlink()

        # Clean up backups
        for backup in Path(".").glob("test_sections_regression.bak.*"):
            backup.unlink()

    def test_replace_section(self):
        """Test replacing a section."""
        result = self.tool._run(
            path=str(self.test_file),
            start_marker="## Section 2",
            new_content="REPLACED CONTENT",
            mode="replace"
        )

        assert "Successfully updated" in result, "Should report success"

        content = self.test_file.read_text()
        assert "REPLACED CONTENT" in content, "New content should be present"
        assert "Middle section data" not in content, "Old content should be replaced"

    def test_append_to_section(self):
        """Test appending to a section."""
        result = self.tool._run(
            path=str(self.test_file),
            start_marker="## Section 1",
            new_content="APPENDED TEXT",
            mode="append"
        )

        assert "Successfully updated" in result, "Should report success"

        content = self.test_file.read_text()
        assert "APPENDED TEXT" in content, "Appended content should be present"
        assert "Some more text here" in content, "Original content should remain"

    def test_prepend_to_section(self):
        """Test prepending to a section."""
        result = self.tool._run(
            path=str(self.test_file),
            start_marker="## Section 3",
            new_content="PREPENDED TEXT",
            mode="prepend"
        )

        assert "Successfully updated" in result, "Should report success"

        content = self.test_file.read_text()
        lines = content.split('\n')

        # Find Section 3
        section3_idx = next(i for i, line in enumerate(lines) if "## Section 3" in line)

        # Prepended content should come before original
        section_text = '\n'.join(lines[section3_idx:section3_idx+5])
        prepend_idx = section_text.find("PREPENDED TEXT")
        original_idx = section_text.find("This is section 3")

        assert prepend_idx < original_idx, "Prepended text should come before original"

    def test_backup_creation(self):
        """Test that backups are created."""
        backups_before = list(Path(".").glob("test_sections_regression.bak.*"))

        self.tool._run(
            path=str(self.test_file),
            start_marker="## Section 1",
            new_content="TEST",
            mode="replace"
        )

        backups_after = list(Path(".").glob("test_sections_regression.bak.*"))

        assert len(backups_after) > len(backups_before), "Backup should be created"

    def test_marker_not_found(self):
        """Test error when marker doesn't exist."""
        result = self.tool._run(
            path=str(self.test_file),
            start_marker="## NonExistent Section",
            new_content="TEST",
            mode="replace"
        )

        assert "not found" in result.lower(), "Should report marker not found"

    def test_invalid_mode(self):
        """Test error with invalid mode."""
        result = self.tool._run(
            path=str(self.test_file),
            start_marker="## Section 1",
            new_content="TEST",
            mode="invalid_mode"
        )

        assert "invalid mode" in result.lower(), "Should report invalid mode"

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        from src.agent.tools.file_ops import PathValidationError

        with pytest.raises(PathValidationError):
            self.tool._run(
                path="nonexistent_file.md",
                start_marker="## Test",
                new_content="TEST",
                mode="replace"
            )


class TestAgentIntegration:
    """Test that new tools integrate properly with agent."""

    def test_agent_has_six_tools(self):
        """Verify agent has 6 tools (5 original + 1 new)."""
        agent = CodingAgent()
        assert len(agent.tools) == 6, f"Expected 6 tools, got {len(agent.tools)}"

    def test_update_file_section_tool_registered(self):
        """Verify update_file_section tool is available."""
        agent = CodingAgent()
        tool_names = [tool.name for tool in agent.tools]

        assert "update_file_section" in tool_names, \
            "update_file_section should be registered"

    def test_all_original_tools_still_present(self):
        """Verify original tools are still available."""
        agent = CodingAgent()
        tool_names = [tool.name for tool in agent.tools]

        required_tools = [
            "read_file",
            "write_file",
            "list_directory",
            "search_code",
            "rag_search"
        ]

        for tool_name in required_tools:
            assert tool_name in tool_names, f"{tool_name} should be available"

    def test_tools_have_descriptions(self):
        """Verify all tools have descriptions."""
        agent = CodingAgent()

        for tool in agent.tools:
            assert tool.description, f"{tool.name} should have description"
            assert len(tool.description) > 10, \
                f"{tool.name} description too short"


class TestBackwardCompatibility:
    """Test that existing functionality still works."""

    def test_regular_ask_still_works(self):
        """Verify non-streaming ask() method still works."""
        agent = CodingAgent()
        agent.clear_history()

        response = agent.ask("What is 2+2?")

        assert response is not None, "Should return a response"
        assert hasattr(response, 'content'), "Should have content attribute"
        assert len(response.content) > 0, "Content should not be empty"

    def test_clear_history_works(self):
        """Verify clear_history still works."""
        agent = CodingAgent()

        agent.ask("Test message")
        history_before = agent.get_conversation_history()

        agent.clear_history()
        history_after = agent.get_conversation_history()

        assert len(history_before) > 0, "Should have history before clear"
        assert len(history_after) == 0, "History should be empty after clear"

    def test_write_file_tool_still_works(self):
        """Verify write_file tool still works (not broken by new tool)."""
        agent = CodingAgent()
        test_file = Path("test_write_regression.txt")

        try:
            # This should work if tools are properly configured
            write_tool = next(t for t in agent.tools if t.name == "write_file")
            assert write_tool is not None, "write_file tool should exist"

            result = write_tool._run(
                path=str(test_file),
                content="test content",
                confirm_write=True
            )

            assert "successfully" in result.lower(), "Write should succeed"
            assert test_file.exists(), "File should be created"

        finally:
            if test_file.exists():
                test_file.unlink()
            # Clean up backups
            for backup in Path(".").glob("test_write_regression.bak.*"):
                backup.unlink()


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_stream_with_error_query(self):
        """Test streaming with a query that causes errors."""
        agent = CodingAgent()

        # This should handle gracefully
        updates = list(agent.ask_stream("Read /nonexistent/path/file.txt"))

        # Should have some updates (even if error)
        assert len(updates) > 0, "Should yield updates even on error"

    def test_update_section_with_special_characters(self):
        """Test section update with special characters in content."""
        test_file = Path("test_special_chars.md")
        test_file.write_text("""## Test

Content here.

## End
""")

        tool = get_update_file_section_tool()

        try:
            result = tool._run(
                path=str(test_file),
                start_marker="## Test",
                new_content="Special: @#$%^&*() 'quotes' \"double\" `backticks`",
                mode="replace"
            )

            assert "Successfully updated" in result, "Should handle special chars"

            content = test_file.read_text()
            assert "@#$%^&*()" in content, "Special chars should be preserved"

        finally:
            test_file.unlink()
            for backup in Path(".").glob("test_special_chars.bak.*"):
                backup.unlink()

    def test_update_section_with_empty_content(self):
        """Test section update with empty content."""
        test_file = Path("test_empty.md")
        test_file.write_text("""## Section 1

Old content here.

## Section 2
""")

        tool = get_update_file_section_tool()

        try:
            result = tool._run(
                path=str(test_file),
                start_marker="## Section 1",
                new_content="",  # Empty content
                mode="replace"
            )

            assert "Successfully updated" in result, "Should handle empty content"

        finally:
            test_file.unlink()
            for backup in Path(".").glob("test_empty.bak.*"):
                backup.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
