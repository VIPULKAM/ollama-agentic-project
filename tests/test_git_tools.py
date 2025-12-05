"""Tests for git integration tools.

These tests cover all git tools: status, diff, commit, branch, and log.
"""

import pytest
import subprocess
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agent.tools.git_tools import (
    GitStatusTool,
    GitDiffTool,
    GitCommitTool,
    GitBranchTool,
    GitLogTool,
    check_git_available,
    check_git_repository,
    run_git_command,
    GitError,
    GitNotInstalledError,
    NotAGitRepositoryError,
    get_git_status_tool,
    get_git_diff_tool,
    get_git_commit_tool,
    get_git_branch_tool,
    get_git_log_tool,
)


class TestGitUtilities:
    """Test utility functions for git operations."""

    def test_check_git_available_success(self):
        """Test git availability check when git is installed."""
        # This will pass if git is actually installed
        result = check_git_available()
        assert isinstance(result, bool)

    @patch('subprocess.run')
    def test_check_git_available_not_installed(self, mock_run):
        """Test git availability check when git is not installed."""
        mock_run.side_effect = FileNotFoundError()
        result = check_git_available()
        assert result is False

    @patch('subprocess.run')
    def test_check_git_available_timeout(self, mock_run):
        """Test git availability check with timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired('git', 5)
        result = check_git_available()
        assert result is False

    @patch('subprocess.run')
    def test_check_git_repository_true(self, mock_run):
        """Test repository check when inside git repo."""
        mock_run.return_value = MagicMock(returncode=0)
        result = check_git_repository()
        assert result is True

    @patch('subprocess.run')
    def test_check_git_repository_false(self, mock_run):
        """Test repository check when not in git repo."""
        mock_run.return_value = MagicMock(returncode=1)
        result = check_git_repository()
        assert result is False

    @patch('src.agent.tools.git_tools.check_git_available')
    def test_run_git_command_git_not_available(self, mock_check):
        """Test run_git_command raises error when git not available."""
        mock_check.return_value = False

        with pytest.raises(GitNotInstalledError) as exc_info:
            run_git_command(['git', 'status'])

        assert "git is not installed" in str(exc_info.value)

    @patch('src.agent.tools.git_tools.check_git_available')
    @patch('src.agent.tools.git_tools.check_git_repository')
    def test_run_git_command_not_a_repo(self, mock_check_repo, mock_check_git):
        """Test run_git_command raises error when not in git repo."""
        mock_check_git.return_value = True
        mock_check_repo.return_value = False

        with pytest.raises(NotAGitRepositoryError) as exc_info:
            run_git_command(['git', 'status'])

        assert "Not a git repository" in str(exc_info.value)

    @patch('src.agent.tools.git_tools.check_git_available')
    @patch('src.agent.tools.git_tools.check_git_repository')
    @patch('subprocess.run')
    def test_run_git_command_timeout(self, mock_run, mock_check_repo, mock_check_git):
        """Test run_git_command raises error on timeout."""
        mock_check_git.return_value = True
        mock_check_repo.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired('git', 30)

        with pytest.raises(GitError) as exc_info:
            run_git_command(['git', 'status'])

        assert "timed out" in str(exc_info.value)


class TestGitStatusTool:
    """Test GitStatusTool functionality."""

    @pytest.fixture
    def tool(self):
        """Create GitStatusTool instance."""
        return get_git_status_tool()

    def test_tool_metadata(self, tool):
        """Test tool has correct metadata."""
        assert tool.name == "git_status"
        assert "git" in tool.description.lower()
        assert "status" in tool.description.lower()

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_status_success(self, mock_run, tool):
        """Test successful git status."""
        # Mock git status output
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="On branch main\nnothing to commit"),
            MagicMock(returncode=0, stdout="main")
        ]

        result = tool._run(short=False)

        assert "Git Status" in result
        assert "Branch" in result
        assert "main" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_status_short_format(self, mock_run, tool):
        """Test git status with short format."""
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="## main\n M file.py"),
            MagicMock(returncode=0, stdout="main")
        ]

        result = tool._run(short=True)

        assert "main" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_status_error(self, mock_run, tool):
        """Test git status with error."""
        mock_run.side_effect = GitError("Git failed")

        result = tool._run()

        assert "Error" in result


class TestGitDiffTool:
    """Test GitDiffTool functionality."""

    @pytest.fixture
    def tool(self):
        """Create GitDiffTool instance."""
        return get_git_diff_tool()

    def test_tool_metadata(self, tool):
        """Test tool has correct metadata."""
        assert tool.name == "git_diff"
        assert "diff" in tool.description.lower()

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_diff_unstaged(self, mock_run, tool):
        """Test git diff for unstaged changes."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="diff --git a/file.py b/file.py\n+new line"
        )

        result = tool._run(staged=False)

        assert "Git Diff" in result
        assert "unstaged" in result
        assert "diff" in result.lower()

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_diff_staged(self, mock_run, tool):
        """Test git diff for staged changes."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="diff --git a/file.py b/file.py\n+new line"
        )

        result = tool._run(staged=True)

        assert "staged" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_diff_specific_file(self, mock_run, tool):
        """Test git diff for specific file."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="diff --git a/test.py b/test.py\n+changes"
        )

        result = tool._run(file_path="test.py")

        assert "test.py" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_diff_no_changes(self, mock_run, tool):
        """Test git diff when no changes."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        result = tool._run()

        assert "No" in result and "changes" in result


class TestGitCommitTool:
    """Test GitCommitTool functionality."""

    @pytest.fixture
    def tool(self):
        """Create GitCommitTool instance."""
        return get_git_commit_tool()

    def test_tool_metadata(self, tool):
        """Test tool has correct metadata."""
        assert tool.name == "git_commit"
        assert "commit" in tool.description.lower()

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_commit_all_changes(self, mock_run, tool):
        """Test committing all changes."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add -A
            MagicMock(
                returncode=0,
                stdout="[main abc123] Test commit"
            )
        ]

        result = tool._run(message="Test commit")

        assert "Commit" in result
        assert "successfully" in result.lower()
        assert "Test commit" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_commit_specific_files(self, mock_run, tool):
        """Test committing specific files."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add file1.py
            MagicMock(returncode=0),  # git add file2.py
            MagicMock(
                returncode=0,
                stdout="[main def456] Feature: Add new files"
            )
        ]

        result = tool._run(
            message="Feature: Add new files",
            files=["file1.py", "file2.py"]
        )

        assert "successfully" in result.lower()
        assert "2 file(s)" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_commit_nothing_to_commit(self, mock_run, tool):
        """Test commit when nothing to commit."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add -A
            MagicMock(returncode=1, stdout="nothing to commit")
        ]

        result = tool._run(message="Test")

        assert "Nothing to commit" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_commit_amend(self, mock_run, tool):
        """Test amending previous commit."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add -A
            MagicMock(
                returncode=0,
                stdout="[main abc123] Amended commit"
            )
        ]

        result = tool._run(message="Amended commit", amend=True)

        assert "amended" in result.lower()


class TestGitBranchTool:
    """Test GitBranchTool functionality."""

    @pytest.fixture
    def tool(self):
        """Create GitBranchTool instance."""
        return get_git_branch_tool()

    def test_tool_metadata(self, tool):
        """Test tool has correct metadata."""
        assert tool.name == "git_branch"
        assert "branch" in tool.description.lower()

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_branch_list(self, mock_run, tool):
        """Test listing branches."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="* main\n  feature/test\n  develop"
        )

        result = tool._run(action="list")

        assert "Git Branches" in result
        assert "main" in result
        assert "feature/test" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_branch_create(self, mock_run, tool):
        """Test creating a branch."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        result = tool._run(action="create", branch_name="feature/new")

        assert "created" in result.lower()
        assert "feature/new" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_branch_switch(self, mock_run, tool):
        """Test switching branches."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Switched to branch 'develop'"
        )

        result = tool._run(action="switch", branch_name="develop")

        assert "Switched" in result
        assert "develop" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_branch_delete(self, mock_run, tool):
        """Test deleting a branch."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Deleted branch old-feature"
        )

        result = tool._run(action="delete", branch_name="old-feature")

        assert "deleted" in result.lower()
        assert "old-feature" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_branch_delete_force(self, mock_run, tool):
        """Test force deleting a branch."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Deleted branch unmerged-feature"
        )

        result = tool._run(
            action="delete",
            branch_name="unmerged-feature",
            force=True
        )

        assert "deleted" in result.lower()

    def test_branch_invalid_action(self, tool):
        """Test invalid branch action."""
        result = tool._run(action="invalid")

        assert "Error" in result
        assert "Invalid action" in result

    def test_branch_missing_name(self, tool):
        """Test branch action without required branch_name."""
        result = tool._run(action="create")

        assert "Error" in result
        assert "branch_name required" in result


class TestGitLogTool:
    """Test GitLogTool functionality."""

    @pytest.fixture
    def tool(self):
        """Create GitLogTool instance."""
        return get_git_log_tool()

    def test_tool_metadata(self, tool):
        """Test tool has correct metadata."""
        assert tool.name == "git_log"
        assert "commit" in tool.description.lower()
        assert "history" in tool.description.lower()

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_log_default(self, mock_run, tool):
        """Test git log with defaults."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123 Initial commit\ndef456 Add feature\n"
        )

        result = tool._run()

        assert "Git Log" in result
        assert "abc123" in result
        assert "Initial commit" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_log_custom_limit(self, mock_run, tool):
        """Test git log with custom limit."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123 Recent commit"
        )

        result = tool._run(limit=5)

        assert "last 5 commits" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_log_specific_file(self, mock_run, tool):
        """Test git log for specific file."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123 Update file.py"
        )

        result = tool._run(file_path="file.py")

        assert "file.py" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_log_with_graph(self, mock_run, tool):
        """Test git log with graph."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="* abc123 Commit\n* def456 Another"
        )

        result = tool._run(graph=True)

        assert "*" in result

    @patch('src.agent.tools.git_tools.run_git_command')
    def test_log_no_commits(self, mock_run, tool):
        """Test git log with no commits."""
        mock_run.return_value = MagicMock(returncode=0, stdout="")

        result = tool._run()

        assert "No commits found" in result


class TestGitToolsIntegration:
    """Integration tests for git tools."""

    def test_all_factory_functions(self):
        """Test all factory functions create correct tools."""
        status_tool = get_git_status_tool()
        diff_tool = get_git_diff_tool()
        commit_tool = get_git_commit_tool()
        branch_tool = get_git_branch_tool()
        log_tool = get_git_log_tool()

        assert isinstance(status_tool, GitStatusTool)
        assert isinstance(diff_tool, GitDiffTool)
        assert isinstance(commit_tool, GitCommitTool)
        assert isinstance(branch_tool, GitBranchTool)
        assert isinstance(log_tool, GitLogTool)

    def test_tools_have_descriptions(self):
        """Test all tools have proper descriptions."""
        tools = [
            get_git_status_tool(),
            get_git_diff_tool(),
            get_git_commit_tool(),
            get_git_branch_tool(),
            get_git_log_tool(),
        ]

        for tool in tools:
            assert tool.description
            assert len(tool.description) > 20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
