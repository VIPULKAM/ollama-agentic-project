"""Git integration tools for version control operations.

This module provides LangChain tools for git operations including status,
diff, commit, branch management, and log viewing.
"""

import subprocess
import re
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from ...utils.logging import file_ops_logger
from ...config.settings import settings


class GitError(Exception):
    """Base exception for git operation failures."""
    pass


class GitNotInstalledError(GitError):
    """Raised when git is not installed or not in PATH."""
    pass


class NotAGitRepositoryError(GitError):
    """Raised when operation is attempted outside a git repository."""
    pass


def check_git_available() -> bool:
    """Check if git is installed and available in PATH.

    Returns:
        bool: True if git is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_git_repository() -> bool:
    """Check if current directory is inside a git repository.

    Returns:
        bool: True if inside git repo, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path.cwd()
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_git_command(
    command: List[str],
    timeout: int = 30,
    check: bool = True
) -> subprocess.CompletedProcess:
    """Run a git command and return the result.

    Args:
        command: List of command arguments (e.g., ['git', 'status'])
        timeout: Command timeout in seconds
        check: If True, raise GitError on non-zero exit code

    Returns:
        subprocess.CompletedProcess: Command result

    Raises:
        GitNotInstalledError: If git is not available
        NotAGitRepositoryError: If not in a git repository
        GitError: If command fails and check=True
    """
    # Verify git is available
    if not check_git_available():
        raise GitNotInstalledError(
            "git is not installed or not available in PATH. "
            "Install git: https://git-scm.com/downloads"
        )

    # Verify we're in a git repository (unless it's git init)
    if command[1] != "init" and not check_git_repository():
        raise NotAGitRepositoryError(
            "Not a git repository. Run 'git init' first or navigate to a git repository."
        )

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd()
        )

        if check and result.returncode != 0:
            raise GitError(
                f"Git command failed: {' '.join(command)}\n"
                f"Error: {result.stderr}"
            )

        file_ops_logger.info(f"GIT: {' '.join(command)} (exit code: {result.returncode})")
        return result

    except subprocess.TimeoutExpired:
        raise GitError(f"Git command timed out after {timeout}s: {' '.join(command)}")


# ============================================================================
# Tool 1: GitStatusTool
# ============================================================================

class GitStatusInput(BaseModel):
    """Input schema for GitStatusTool."""
    short: bool = Field(
        default=False,
        description="Use short format (like git status -s)"
    )


class GitStatusTool(BaseTool):
    """Show git repository status including unstaged/staged changes and current branch.

    Displays:
    - Current branch
    - Uncommitted changes (staged and unstaged)
    - Untracked files
    - Branch tracking status (ahead/behind remote)
    """

    name: str = "git_status"
    description: str = """Show git repository status.

Use this tool to check:
- What branch you're on
- What files have been modified
- What files are staged for commit
- What files are untracked
- If branch is ahead/behind remote

Args:
- short: If True, use compact format (default: False)

Returns detailed status information about the git repository.
"""
    args_schema: type = GitStatusInput

    def _run(self, short: bool = False) -> str:
        """Get git status."""
        try:
            # Get status
            if short:
                result = run_git_command(["git", "status", "-s", "-b"])
            else:
                result = run_git_command(["git", "status"])

            # Get current branch
            branch_result = run_git_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            current_branch = branch_result.stdout.strip()

            output = f"📊 **Git Status**\n\n"
            output += f"**Branch**: {current_branch}\n\n"

            if result.stdout.strip():
                output += result.stdout
            else:
                output += "Working tree clean - no changes to commit"

            return output

        except GitError as e:
            return f"Error: {str(e)}"


# ============================================================================
# Tool 2: GitDiffTool
# ============================================================================

class GitDiffInput(BaseModel):
    """Input schema for GitDiffTool."""
    staged: bool = Field(
        default=False,
        description="Show staged changes (git diff --staged)"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Specific file to diff (optional)"
    )
    context_lines: int = Field(
        default=3,
        description="Number of context lines to show (default: 3)"
    )


class GitDiffTool(BaseTool):
    """Show changes in working directory or staged area.

    Can show:
    - Unstaged changes (working directory vs index)
    - Staged changes (index vs HEAD)
    - Changes for specific file
    """

    name: str = "git_diff"
    description: str = """Show git diff of changes.

Use this tool to see:
- What changes have been made to files
- Differences between working directory and last commit
- What changes are staged for commit

Args:
- staged: If True, show staged changes (default: False = show unstaged)
- file_path: Optional path to specific file to diff
- context_lines: Lines of context around changes (default: 3)

Returns unified diff showing line-by-line changes.
"""
    args_schema: type = GitDiffInput

    def _run(
        self,
        staged: bool = False,
        file_path: Optional[str] = None,
        context_lines: int = 3
    ) -> str:
        """Get git diff."""
        try:
            command = ["git", "diff", f"-U{context_lines}"]

            if staged:
                command.append("--staged")

            if file_path:
                command.append("--")
                command.append(file_path)

            result = run_git_command(command, check=False)

            if result.returncode == 0:
                if result.stdout.strip():
                    diff_type = "staged" if staged else "unstaged"
                    file_info = f" for {file_path}" if file_path else ""

                    output = f"📝 **Git Diff ({diff_type} changes{file_info})**\n\n"
                    output += "```diff\n"
                    output += result.stdout
                    output += "```"
                    return output
                else:
                    diff_type = "staged" if staged else "unstaged"
                    return f"No {diff_type} changes to show"
            else:
                return f"Error: {result.stderr}"

        except GitError as e:
            return f"Error: {str(e)}"


# ============================================================================
# Tool 3: GitCommitTool
# ============================================================================

class GitCommitInput(BaseModel):
    """Input schema for GitCommitTool."""
    message: str = Field(
        description="Commit message (use conventional commits format: feat/fix/docs/etc)"
    )
    files: Optional[List[str]] = Field(
        default=None,
        description="Specific files to stage and commit (if None, stages all changes)"
    )
    amend: bool = Field(
        default=False,
        description="Amend the previous commit instead of creating new one"
    )


class GitCommitTool(BaseTool):
    """Stage changes and create a git commit.

    Stages specified files (or all changes) and creates a commit with the provided message.
    Supports conventional commit format (feat:, fix:, docs:, etc.).
    """

    name: str = "git_commit"
    description: str = """Stage and commit changes to git.

Use this tool to:
- Stage files for commit
- Create a commit with a message
- Amend the previous commit

Args:
- message: Commit message (use conventional commits: feat:/fix:/docs:/refactor:/test:)
- files: List of files to stage (if None, stages all modified files)
- amend: If True, amend previous commit instead of creating new (default: False)

Example messages:
- "feat: Add git integration tools"
- "fix: Correct retry logic timeout"
- "docs: Update README with Phase 2 info"

Returns confirmation of commit creation.
"""
    args_schema: type = GitCommitInput

    def _run(
        self,
        message: str,
        files: Optional[List[str]] = None,
        amend: bool = False
    ) -> str:
        """Create a git commit."""
        try:
            # Stage files
            if files:
                for file in files:
                    add_result = run_git_command(["git", "add", file])
                    if add_result.returncode != 0:
                        return f"Error staging {file}: {add_result.stderr}"
                staged_info = f"Staged {len(files)} file(s)"
            else:
                # Stage all changes
                run_git_command(["git", "add", "-A"])
                staged_info = "Staged all changes"

            # Create commit
            commit_command = ["git", "commit", "-m", message]
            if amend:
                commit_command.append("--amend")

            result = run_git_command(commit_command, check=False)

            if result.returncode == 0:
                # Extract commit hash from output
                hash_match = re.search(r'\[[\w\-/]+ ([a-f0-9]+)\]', result.stdout)
                commit_hash = hash_match.group(1) if hash_match else "unknown"

                action = "amended" if amend else "created"
                output = f"✓ **Commit {action} successfully**\n\n"
                output += f"**Commit**: {commit_hash}\n"
                output += f"**Message**: {message}\n"
                output += f"**Staged**: {staged_info}\n\n"
                output += result.stdout

                file_ops_logger.info(f"GIT_COMMIT: {commit_hash[:7]} - {message}")
                return output
            else:
                # Check if it's "nothing to commit"
                if "nothing to commit" in result.stdout.lower():
                    return "Nothing to commit - working tree is clean"
                return f"Error creating commit: {result.stderr}"

        except GitError as e:
            return f"Error: {str(e)}"


# ============================================================================
# Tool 4: GitBranchTool
# ============================================================================

class GitBranchInput(BaseModel):
    """Input schema for GitBranchTool."""
    action: str = Field(
        description="Action to perform: 'list', 'create', 'switch', 'delete'"
    )
    branch_name: Optional[str] = Field(
        default=None,
        description="Branch name (required for create/switch/delete)"
    )
    force: bool = Field(
        default=False,
        description="Force operation (for delete)"
    )


class GitBranchTool(BaseTool):
    """Manage git branches - create, switch, list, or delete.

    Supports all common branch operations:
    - List all branches
    - Create new branch
    - Switch to existing branch
    - Delete branch
    """

    name: str = "git_branch"
    description: str = """Manage git branches.

Use this tool to:
- List all branches
- Create a new branch
- Switch to a different branch
- Delete a branch

Args:
- action: What to do - 'list', 'create', 'switch', 'delete'
- branch_name: Name of branch (required for create/switch/delete)
- force: Force deletion of unmerged branch (default: False)

Examples:
- List branches: action='list'
- Create branch: action='create', branch_name='feature/new-feature'
- Switch branch: action='switch', branch_name='main'
- Delete branch: action='delete', branch_name='old-feature', force=False

Returns result of branch operation.
"""
    args_schema: type = GitBranchInput

    def _run(
        self,
        action: str,
        branch_name: Optional[str] = None,
        force: bool = False
    ) -> str:
        """Perform branch operation."""
        try:
            if action == "list":
                # List all branches
                result = run_git_command(["git", "branch", "-a", "-v"])

                output = "🌿 **Git Branches**\n\n"
                output += "```\n"
                output += result.stdout
                output += "```"
                return output

            elif action == "create":
                if not branch_name:
                    return "Error: branch_name required for create action"

                # Create new branch
                result = run_git_command(["git", "branch", branch_name])

                if result.returncode == 0:
                    output = f"✓ **Branch created**: {branch_name}\n\n"
                    output += "Use git_branch(action='switch', branch_name='{branch_name}') to switch to it"

                    file_ops_logger.info(f"GIT_BRANCH: Created {branch_name}")
                    return output
                else:
                    return f"Error creating branch: {result.stderr}"

            elif action == "switch":
                if not branch_name:
                    return "Error: branch_name required for switch action"

                # Switch to branch
                result = run_git_command(["git", "checkout", branch_name])

                if result.returncode == 0:
                    output = f"✓ **Switched to branch**: {branch_name}\n\n"
                    output += result.stdout

                    file_ops_logger.info(f"GIT_BRANCH: Switched to {branch_name}")
                    return output
                else:
                    return f"Error switching branch: {result.stderr}"

            elif action == "delete":
                if not branch_name:
                    return "Error: branch_name required for delete action"

                # Delete branch
                delete_flag = "-D" if force else "-d"
                result = run_git_command(["git", "branch", delete_flag, branch_name], check=False)

                if result.returncode == 0:
                    output = f"✓ **Branch deleted**: {branch_name}\n\n"
                    output += result.stdout

                    file_ops_logger.info(f"GIT_BRANCH: Deleted {branch_name}")
                    return output
                else:
                    return f"Error deleting branch: {result.stderr}"

            else:
                return f"Error: Invalid action '{action}'. Use: list, create, switch, or delete"

        except GitError as e:
            return f"Error: {str(e)}"


# ============================================================================
# Tool 5: GitLogTool
# ============================================================================

class GitLogInput(BaseModel):
    """Input schema for GitLogTool."""
    limit: int = Field(
        default=10,
        description="Number of commits to show (default: 10)"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Show commits for specific file (optional)"
    )
    oneline: bool = Field(
        default=True,
        description="Use compact one-line format (default: True)"
    )
    graph: bool = Field(
        default=False,
        description="Show branch graph (default: False)"
    )


class GitLogTool(BaseTool):
    """Show git commit history.

    Displays recent commits with hash, author, date, and message.
    Can filter by file and show branch graph.
    """

    name: str = "git_log"
    description: str = """Show git commit history.

Use this tool to:
- View recent commits
- See commit history for specific file
- Understand branch history
- Find commit hashes for reverting

Args:
- limit: Number of commits to show (default: 10)
- file_path: Show commits for specific file (optional)
- oneline: Use compact format (default: True)
- graph: Show branch graph (default: False)

Returns formatted commit history.
"""
    args_schema: type = GitLogInput

    def _run(
        self,
        limit: int = 10,
        file_path: Optional[str] = None,
        oneline: bool = True,
        graph: bool = False
    ) -> str:
        """Get git log."""
        try:
            command = ["git", "log", f"-n{limit}"]

            if oneline:
                command.append("--oneline")

            if graph:
                command.append("--graph")
                command.append("--decorate")

            if file_path:
                command.append("--")
                command.append(file_path)

            result = run_git_command(command)

            if result.stdout.strip():
                file_info = f" for {file_path}" if file_path else ""
                output = f"📜 **Git Log (last {limit} commits{file_info})**\n\n"
                output += "```\n"
                output += result.stdout
                output += "```"
                return output
            else:
                return "No commits found"

        except GitError as e:
            return f"Error: {str(e)}"


# ============================================================================
# Factory Functions
# ============================================================================

def get_git_status_tool() -> GitStatusTool:
    """Factory function to create GitStatusTool instance."""
    return GitStatusTool()


def get_git_diff_tool() -> GitDiffTool:
    """Factory function to create GitDiffTool instance."""
    return GitDiffTool()


def get_git_commit_tool() -> GitCommitTool:
    """Factory function to create GitCommitTool instance."""
    return GitCommitTool()


def get_git_branch_tool() -> GitBranchTool:
    """Factory function to create GitBranchTool instance."""
    return GitBranchTool()


def get_git_log_tool() -> GitLogTool:
    """Factory function to create GitLogTool instance."""
    return GitLogTool()
