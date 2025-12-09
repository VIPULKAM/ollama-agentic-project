"""Shared path validation utilities.

This module provides security-focused path validation to prevent:
- Path traversal attacks
- Access outside working directory
- Symlink attacks
"""

import os
from pathlib import Path
from typing import Optional

from ..config.settings import settings


# Current working directory - all files must be within this directory
CWD = Path.cwd().resolve()


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
    """Check if file size is within limits.

    Args:
        path: Path to file to check
        max_size_mb: Maximum size in MB (defaults to settings.MAX_FILE_SIZE_MB)

    Raises:
        FileSizeError: If file exceeds size limit
    """
    if max_size_mb is None:
        max_size_mb = settings.MAX_FILE_SIZE_MB

    if not path.exists() or not path.is_file():
        return

    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if size_mb > max_size_mb:
        raise FileSizeError(
            f"File too large: {size_mb:.1f}MB exceeds {max_size_mb}MB limit"
        )
