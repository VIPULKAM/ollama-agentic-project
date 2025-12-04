"""File operation tools for the AI Coding Agent.

These tools provide safe file access within the current working directory.
All operations include path validation, security checks, and comprehensive logging.
"""

import os
import shutil
import difflib
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator
from langchain.tools import BaseTool

from ...config.settings import settings
from ...utils.logging import file_ops_logger


# Current working directory - all paths must be within this directory
CWD = Path.cwd().resolve()

# Regex to exclude common, unhelpful directories for search operations
EXCLUDE_DIRS_PATTERN = re.compile(r'(\.git|\.venv|__pycache__|node_modules|\.mypy_cache)$', re.IGNORECASE)


class FileOperationError(Exception):
    """Base exception for file operation errors."""
    pass


class PathValidationError(FileOperationError):
    """Exception raised when path validation fails."""
    pass


class FileSizeError(FileOperationError):
    """Exception raised when file size exceeds limit."""
    pass


def validate_path(path_str: str, must_exist: bool = True) -> Path:
    """Validate and resolve a file path to ensure it's safe.

    Security checks:
    - Path must be within current working directory
    - No path traversal attacks (../)
    - No symlinks pointing outside CWD
    - File must exist (if must_exist=True)
    - File must be readable/writable

    Args:
        path_str: Path string to validate
        must_exist: Whether the path must exist (default: True)

    Returns:
        Path: Validated and resolved Path object

    Raises:
        PathValidationError: If validation fails
    """
    # Check for empty or whitespace-only strings
    if not path_str or not path_str.strip():
        raise PathValidationError("Path cannot be empty")

    try:
        # Convert to Path and resolve (resolves symlinks and ..)
        path = Path(path_str).expanduser().resolve()

        # Check if path is within CWD
        try:
            path.relative_to(CWD)
        except ValueError:
            raise PathValidationError(
                f"Access denied: Path '{path_str}' is outside current working directory. "
                f"Only files within {CWD} can be accessed."
            )

        # Check if path exists
        if must_exist and not path.exists():
            raise PathValidationError(f"Path does not exist: {path_str}")

        # If it's a symlink, ensure target is also within CWD
        if path.is_symlink():
            real_path = path.resolve()
            try:
                real_path.relative_to(CWD)
            except ValueError:
                raise PathValidationError(
                    f"Access denied: Symlink '{path_str}' points outside current working directory"
                )

        # Check if path is readable (only if it exists)
        if must_exist and not os.access(path, os.R_OK):
            raise PathValidationError(f"Permission denied: Cannot read {path_str}")
        
        # Prevent attempting to write to a directory
        if path.exists() and path.is_dir() and not must_exist:
            raise PathValidationError(f"Cannot write: '{path_str}' is a directory.")


        return path

    except (OSError, RuntimeError) as e:
        raise PathValidationError(f"Invalid path '{path_str}': {str(e)}")


def check_file_size(path: Path, max_size_mb: Optional[int] = None) -> None:
    """Check if file size is within limits. (Existing function)"""
    if max_size_mb is None:
        max_size_mb = settings.MAX_FILE_SIZE_MB

    max_size_bytes = max_size_mb * 1024 * 1024
    file_size = path.stat().st_size

    if file_size > max_size_bytes:
        size_mb = file_size / (1024 * 1024)
        raise FileSizeError(
            f"File too large: {size_mb:.2f}MB (max: {max_size_mb}MB). "
            f"Consider reading the file in chunks or increasing MAX_FILE_SIZE_MB."
        )


def read_file_content(path: Path) -> str:
    """Read file content with encoding fallback. (Existing function)"""
    # Try UTF-8 first
    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        file_ops_logger.warning(f"UTF-8 decode failed for {path}, trying latin-1")
        try:
            return path.read_text(encoding='latin-1')
        except Exception as e:
            raise FileOperationError(f"Failed to read file with encoding fallback: {str(e)}")
    except Exception as e:
        raise FileOperationError(f"Failed to read file: {str(e)}")


def create_backup(path: Path) -> Optional[Path]:
    """Creates a timestamped backup of an existing file. (NEW UTILITY)

    Args:
        path: The path to the file to backup.

    Returns:
        Optional[Path]: The path to the backup file, or None if backup is disabled.
    """
    # Check if backups are enabled and the file actually exists
    if not settings.FILE_BACKUP_ENABLED or not path.exists():
        return None

    try:
        # Create a timestamped backup name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = path.with_suffix(f".bak.{timestamp}")
        
        # Use copy2 to preserve file metadata
        shutil.copy2(path, backup_path)
        
        file_ops_logger.info(f"BACKUP: Created backup for {path.name} at {backup_path.name}")
        return backup_path
    except Exception as e:
        file_ops_logger.error(f"Failed to create backup for {path}: {e}")
        return None


# ============================================================================
# ReadFile Tool
# ============================================================================

class ReadFileInput(BaseModel):
    """Input schema for read_file tool."""

    path: str = Field(
        description="Path to the file to read (relative or absolute, must be within current directory)"
    )

    @field_validator('path')
    @classmethod
    def validate_path_field(cls, v: str) -> str:
        """Validate path field."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        return v.strip()


class ReadFileTool(BaseTool):
    """Tool to read file contents safely.

    Security features:
    - Path validation (must be within CWD)
    - File size limits
    - Encoding fallback (UTF-8 -> latin-1)
    - Comprehensive logging
    """

    name: str = "read_file"
    description: str = (
        "Read the contents of a file. "
        "Input should be a file path (relative or absolute). "
        "The file must be within the current working directory. "
        "Returns the file contents as a string."
    )
    args_schema: type[BaseModel] = ReadFileInput

    def _run(self, path: str) -> str:
        """Execute the read_file operation.

        Args:
            path: Path to the file to read

        Returns:
            str: File contents or error message
        """
        try:
            # Validate path
            validated_path = validate_path(path, must_exist=True)

            # Check if it's a file (not a directory)
            if not validated_path.is_file():
                error_msg = f"Error: '{path}' is not a file"
                file_ops_logger.error(error_msg)
                return error_msg

            # Check file size
            check_file_size(validated_path)

            # Read file content
            content = read_file_content(validated_path)

            # Log successful operation
            file_ops_logger.info(
                f"READ: {validated_path} "
                f"(size: {validated_path.stat().st_size} bytes, "
                f"lines: {len(content.splitlines())})"
            )

            return content

        except PathValidationError as e:
            error_msg = f"Path validation error: {str(e)}"
            file_ops_logger.error(error_msg)
            return error_msg

        except FileSizeError as e:
            error_msg = f"File size error: {str(e)}"
            file_ops_logger.error(error_msg)
            return error_msg

        except FileOperationError as e:
            error_msg = f"File operation error: {str(e)}"
            file_ops_logger.error(error_msg)
            return error_msg

        except Exception as e:
            error_msg = f"Unexpected error reading file: {str(e)}"
            file_ops_logger.exception(error_msg)
            return error_msg

    async def _arun(self, path: str) -> str:
        """Async version (not implemented, falls back to sync)."""
        return self._run(path)


# ============================================================================
# ListDirectory Tool
# ============================================================================

class ListDirectoryInput(BaseModel):
    """Input schema for list_directory tool."""

    path: str = Field(
        default=".",
        description="Path to the directory to list (default: current directory)"
    )

    @field_validator('path')
    @classmethod
    def validate_path_field(cls, v: str) -> str:
        """Validate path field."""
        if not v or not v.strip():
            return "."  # Default to current directory
        return v.strip()


class ListDirectoryTool(BaseTool):
    """Tool to list directory contents safely.

    Security features:
    - Path validation (must be within CWD)
    - Shows file metadata (size, type, modified time)
    - Formatted output for easy reading
    - Comprehensive logging
    """

    name: str = "list_directory"
    description: str = (
        "List the contents of a directory. "
        "Input should be a directory path (default: current directory). "
        "Returns a formatted list of files and directories with metadata "
        "(name, type, size, modified time)."
    )
    args_schema: type[BaseModel] = ListDirectoryInput

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format.

        Args:
            size_bytes: Size in bytes

        Returns:
            str: Formatted size (e.g., "1.5 KB", "2.3 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _format_time(self, timestamp: float) -> str:
        """Format timestamp in readable format.

        Args:
            timestamp: Unix timestamp

        Returns:
            str: Formatted time (e.g., "2024-01-15 14:30")
        """
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M")

    def _run(self, path: str = ".") -> str:
        """Execute the list_directory operation.

        Args:
            path: Path to the directory to list

        Returns:
            str: Formatted directory listing or error message
        """
        try:
            # Handle empty or whitespace-only paths (default to current directory)
            if not path or not path.strip():
                path = "."

            # Validate path
            validated_path = validate_path(path, must_exist=True)

            # Check if it's a directory
            if not validated_path.is_dir():
                error_msg = f"Error: '{path}' is not a directory"
                file_ops_logger.error(error_msg)
                return error_msg

            # Get directory contents
            entries = []
            try:
                for entry in sorted(validated_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                    try:
                        stat = entry.stat()
                        entry_info = {
                            'name': entry.name,
                            'type': 'DIR' if entry.is_dir() else 'FILE',
                            'size': self._format_size(stat.st_size) if entry.is_file() else '-',
                            'modified': self._format_time(stat.st_mtime),
                        }
                        entries.append(entry_info)
                    except (OSError, PermissionError) as e:
                        # Skip entries we can't access
                        file_ops_logger.warning(f"Cannot access {entry}: {e}")
                        continue

            except PermissionError as e:
                error_msg = f"Permission denied: Cannot list directory '{path}'"
                file_ops_logger.error(error_msg)
                return error_msg

            # Format output
            if not entries:
                result = f"Directory '{path}' is empty."
            else:
                # Create formatted output
                lines = [f"Contents of '{validated_path.relative_to(CWD) if validated_path != CWD else '.'}':\n"]
                lines.append(f"{'Name':<40} {'Type':<6} {'Size':<12} {'Modified':<20}")
                lines.append("-" * 80)

                for entry in entries:
                    lines.append(
                        f"{entry['name']:<40} {entry['type']:<6} "
                        f"{entry['size']:<12} {entry['modified']:<20}"
                    )

                # Add summary
                file_count = sum(1 for e in entries if e['type'] == 'FILE')
                dir_count = sum(1 for e in entries if e['type'] == 'DIR')
                lines.append("-" * 80)
                lines.append(f"Total: {file_count} file(s), {dir_count} directory(ies)")

                result = "\n".join(lines)

            # Log successful operation
            file_ops_logger.info(
                f"LIST: {validated_path} "
                f"(found {len(entries)} entries)"
            )

            return result

        except PathValidationError as e:
            error_msg = f"Path validation error: {str(e)}"
            file_ops_logger.error(error_msg)
            return error_msg

        except FileOperationError as e:
            error_msg = f"File operation error: {str(e)}"
            file_ops_logger.error(error_msg)
            return error_msg

        except Exception as e:
            error_msg = f"Unexpected error listing directory: {str(e)}"
            file_ops_logger.exception(error_msg)
            return error_msg

    async def _arun(self, path: str = ".") -> str:
        """Async version (not implemented, falls back to sync)."""
        return self._run(path)


# ============================================================================
# WriteFile Tool (NEW)
# ============================================================================

class WriteFileInput(BaseModel):
    """Input schema for write_file tool."""
    path: str = Field(
        description="Path to the file to write (relative or absolute, must be within current directory). Can be a new or existing file."
    )
    content: str = Field(
        description="The *full* new content to write to the file, replacing all existing content."
    )
    confirm_write: bool = Field(
        default=False,
        description="MANDATORY: Set to **True** to confirm the write operation. This is a crucial safety measure against accidental changes."
    )

    @field_validator('path')
    @classmethod
    def validate_path_field(cls, v: str) -> str:
        """Validate path field."""
        if not v or not v.strip():
            raise ValueError("Path cannot be empty")
        return v.strip()

    @field_validator('content')
    @classmethod
    def validate_content_field(cls, v: str) -> str:
        """Validate content field."""
        if not v.strip():
            # Allow empty string if it is an intentional file clear, but warn against whitespace only
            if v and not v.strip():
                 raise ValueError("Content to write is whitespace only. If intentionally clearing the file, use an empty string '' for content.")
        return v


class WriteFileTool(BaseTool):
    """Tool to write contents to a file safely.

    Security features:
    - Path validation (must be within CWD)
    - MANDATORY confirmation flag
    - Automatic backup creation before write
    - Diff preview returned to the agent
    - Comprehensive logging
    """

    name: str = "write_file"
    description: str = (
        "Write the full contents to a file. "
        "REQUIRES 'confirm_write=True' in the input. "
        "If the file exists, it returns a diff preview and creates a backup."
    )
    args_schema: type[BaseModel] = WriteFileInput

    def _run(self, path: str, content: str, confirm_write: bool) -> str:
        """Execute the write_file operation."""
        # 0. Pre-Flight Check: Confirmation
        if not confirm_write:
            return (
                "ACTION REQUIRED: Write operation cancelled. "
                "You must set 'confirm_write=True' in the input to execute file modification. "
                "Please verify your plan and try again."
            )

        try:
            # 1. Validate path (must_exist=False to allow creation of new files)
            validated_path = validate_path(path, must_exist=False)
            
            # Read old content for diff and backup (if exists)
            old_content = None
            if validated_path.exists():
                old_content = read_file_content(validated_path)
            
            # 2. Backup
            backup_path = create_backup(validated_path)

            # 3. Generate Diff (for agent feedback)
            diff_output = ""
            if old_content is not None:
                # Use difflib to create a unified diff
                diff = difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=f"a/{path}",
                    tofile=f"b/{path}",
                    lineterm="" # Ensure diff is clean for LLM context
                )
                diff_output = "".join(diff)

            # 4. Create parent directories if needed
            validated_path.parent.mkdir(parents=True, exist_ok=True)

            # 5. Write new content
            validated_path.write_text(content, encoding='utf-8')

            # 6. Result Summary
            log_action = "OVERWRITE" if old_content is not None else "CREATE"
            file_ops_logger.info(
                f"{log_action}: {validated_path} "
                f"(size: {len(content)} bytes, backup: {backup_path.name if backup_path else 'none'})"
            )

            result = [
                f"File successfully {log_action.lower()}: {validated_path.relative_to(CWD)}",
                f"Content size: {len(content)} bytes.",
                f"Backup location: {backup_path.name}" if backup_path else "No backup created (FILE_BACKUP_ENABLED is False or file was new).",
                "",
                "--- DIFF PREVIEW (for your review) ---",
                diff_output or "No diff available (File was newly created or content is identical).",
                "---------------------------------------"
            ]

            return "\n".join(result)

        except PathValidationError as e:
            error_msg = f"Path validation error: {str(e)}"
            file_ops_logger.error(error_msg)
            return error_msg

        except FileOperationError as e:
            error_msg = f"File operation error: {str(e)}"
            file_ops_logger.error(error_msg)
            return error_msg

        except Exception as e:
            error_msg = f"Unexpected error writing file: {str(e)}"
            file_ops_logger.exception(error_msg)
            return error_msg

    async def _arun(self, path: str, content: str, confirm_write: bool) -> str:
        """Async version (not implemented, falls back to sync)."""
        return self._run(path, content, confirm_write)


# ============================================================================
# SearchCode Tool (NEW)
# ============================================================================

class SearchCodeInput(BaseModel):
    """Input schema for search_code tool."""

    pattern: str = Field(
        description="The regular expression pattern to search for (e.g., 'def init')."
    )
    path: str = Field(
        default=".",
        description="The relative directory path to start the recursive search from (default: current directory)."
    )
    file_filter: str = Field(
        default="py,ts,js,md",
        description="Comma-separated file extensions to include in the search (e.g., 'py,js,html'). Ex: 'py' matches '.py' files."
    )
    
    @field_validator('path')
    @classmethod
    def validate_path_field(cls, v: str) -> str:
        """Validate path field."""
        return v.strip() if v and v.strip() else "."


class SearchCodeTool(BaseTool):
    """Tool to perform recursive regex search across the codebase.

    Allows the agent to quickly find function definitions, variable names,
    and specific code patterns. Automatically excludes common build directories.
    """

    name: str = "search_code"
    description: str = (
        "Recursively search the codebase for a given regular expression pattern. "
        "Returns a list of matching lines with file paths and line numbers. "
        "Use this before reading a file to find where code is located."
    )
    args_schema: type[BaseModel] = SearchCodeInput

    def _run(self, pattern: str, path: str = ".", file_filter: str = "py,ts,js,md") -> str:
        """Execute the search_code operation."""
        MAX_RESULTS = 50
        results: List[str] = []
        
        try:
            # 1. Validation and Setup
            search_dir = validate_path(path, must_exist=True)
            if not search_dir.is_dir():
                return f"Error: '{path}' is not a directory. Please provide a directory path."
            
            compiled_pattern = re.compile(pattern)
            
            # Process file extensions
            allowed_extensions = {f".{ext.strip().lower()}" for ext in file_filter.split(',') if ext.strip()}

            # 2. Recursive Search
            for root, dirs, files in os.walk(search_dir):
                # Modify dirs in-place to prune search
                dirs[:] = [d for d in dirs if not EXCLUDE_DIRS_PATTERN.search(d)]
                
                for file_name in files:
                    file_path = Path(root) / file_name
                    relative_path = file_path.relative_to(CWD)
                    
                    # Check extension filter
                    if file_path.suffix.lower() not in allowed_extensions:
                        continue
                    
                    # Skip if file is a symlink pointing outside CWD (already checked by validate_path if used)
                    # We must re-validate the path for safety/size before reading it
                    try:
                        validated_file_path = validate_path(str(file_path), must_exist=True)
                        check_file_size(validated_file_path)
                    except (PathValidationError, FileSizeError):
                        # Skip files that fail security or size checks
                        continue 
                    
                    # Read and search file content
                    try:
                        with validated_file_path.open('r', encoding='utf-8', errors='ignore') as f:
                            for line_num, line in enumerate(f, 1):
                                if compiled_pattern.search(line):
                                    # Format: path:line_num: content
                                    results.append(f"{relative_path}:{line_num}: {line.strip()}")
                                    
                                    if len(results) >= MAX_RESULTS:
                                        # Optimization: stop searching early
                                        raise StopIteration 
                    except StopIteration:
                        break
                    except Exception:
                        # Skip unreadable or corrupted files
                        continue
            
            # 3. Format Output
            file_ops_logger.info(f"SEARCH: Pattern '{pattern}' in '{path}' found {len(results)} matches.")
            
            if not results:
                return f"Search complete: No matches found for pattern '{pattern}' in directory '{path}'."

            summary = [
                f"Search Results ({len(results)} matches found, max {MAX_RESULTS}):",
                "Format: [file_path]:[line_number]: [matching_line_content]"
            ]
            summary.extend(results)
            
            return "\n".join(summary)
            
        except PathValidationError as e:
            error_msg = f"Path validation error: {str(e)}"
            file_ops_logger.error(error_msg)
            return error_msg

        except re.error:
            error_msg = f"Invalid regular expression pattern: '{pattern}'"
            file_ops_logger.error(error_msg)
            return error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during code search: {str(e)}"
            file_ops_logger.exception(error_msg)
            return error_msg

    async def _arun(self, pattern: str, path: str = ".", file_filter: str = "py,ts,js,md") -> str:
        """Async version (not implemented, falls back to sync)."""
        return self._run(pattern, path, file_filter)


# ============================================================================
# Tool Factory Functions
# ============================================================================

def get_read_file_tool() -> ReadFileTool:
    """Get an instance of the ReadFile tool."""
    return ReadFileTool()


def get_list_directory_tool() -> ListDirectoryTool:
    """Get an instance of the ListDirectory tool."""
    return ListDirectoryTool()


def get_write_file_tool() -> WriteFileTool:
    """Get an instance of the WriteFile tool. (NEW FACTORY)"""
    return WriteFileTool()


def get_search_code_tool() -> SearchCodeTool:
    """Get an instance of the SearchCode tool. (NEW FACTORY)"""
    return SearchCodeTool()
