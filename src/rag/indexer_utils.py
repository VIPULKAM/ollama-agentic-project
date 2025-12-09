"""Utility functions for file discovery and filtering in the RAG indexer.

This module provides:
- FileInfo dataclass for file metadata
- .gitignore pattern loading using pathspec
- File discovery generator with filtering and security checks
"""

import os
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import pathspec

from ..config.settings import settings
from ..utils.path_validation import validate_path, PathValidationError

# Configure logging
logger = logging.getLogger("ai_agent.indexer")

# Current working directory - all files must be within this directory
CWD = Path.cwd().resolve()

# Hardcoded exclusion patterns (always applied, even without .gitignore)
HARDCODED_EXCLUSIONS = [
    # Version control
    ".git/",
    ".svn/",
    ".hg/",

    # Python
    ".venv/",
    "venv/",
    "env/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".Python",
    "build/",
    "develop-eggs/",
    "dist/",
    "downloads/",
    "eggs/",
    ".eggs/",
    "lib/",
    "lib64/",
    "parts/",
    "sdist/",
    "var/",
    "wheels/",
    "*.egg-info/",
    ".installed.cfg",
    "*.egg",

    # Node.js
    "node_modules/",
    "npm-debug.log",
    "yarn-error.log",
    ".npm/",
    ".yarn/",

    # IDEs
    ".vscode/",
    ".idea/",
    "*.swp",
    "*.swo",
    "*~",
    ".DS_Store",
    "Thumbs.db",

    # Binary/compiled
    "*.so",
    "*.dll",
    "*.dylib",
    "*.exe",
    "*.out",
    "*.app",

    # Coverage/test artifacts
    ".coverage",
    ".tox/",
    "htmlcov/",
    ".nox/",

    # Documentation builds
    "docs/_build/",
    "site/",

    # Jupyter
    ".ipynb_checkpoints/",

    # Environment files
    ".env.local",
    ".env.*.local",
]


@dataclass
class FileInfo:
    """Metadata for a discovered file.

    This dataclass holds all relevant information about a file discovered
    during the indexing process. It enables efficient filtering, change
    detection, and incremental indexing.

    Attributes:
        path: Absolute path to the file
        relative_path: Path relative to CWD (for display and indexing)
        size: File size in bytes
        mtime: Last modification timestamp (Unix time)
        content_hash: MD5 hash of file content (optional, computed lazily)
    """
    path: Path
    relative_path: Path
    size: int
    mtime: float
    content_hash: Optional[str] = None

    def get_content_hash(self) -> str:
        """Compute MD5 hash of file content for change detection.

        This is computed lazily (only when needed) to avoid overhead
        during initial file discovery.

        Returns:
            str: MD5 hash of file content (hexadecimal)
        """
        if self.content_hash is None:
            try:
                with open(self.path, 'rb') as f:
                    self.content_hash = hashlib.md5(f.read()).hexdigest()
            except (OSError, IOError) as e:
                logger.warning(f"Cannot compute hash for {self.relative_path}: {e}")
                self.content_hash = ""
        return self.content_hash

    def __repr__(self) -> str:
        """String representation for debugging."""
        size_kb = self.size / 1024
        return f"FileInfo({self.relative_path}, {size_kb:.1f}KB)"


def load_gitignore_patterns(cwd: Optional[Path] = None) -> pathspec.PathSpec:
    """Load exclusion patterns from .gitignore and hardcoded defaults.

    This function combines patterns from:
    1. Hardcoded exclusions (always applied)
    2. Project's .gitignore file (if exists)
    3. Settings override (INDEXER_EXCLUDE_PATTERNS)

    Args:
        cwd: Working directory to search for .gitignore (default: current directory)

    Returns:
        pathspec.PathSpec: Compiled pattern matcher for file exclusions
    """
    if cwd is None:
        cwd = CWD

    patterns = []

    # 1. Start with hardcoded exclusions
    patterns.extend(HARDCODED_EXCLUSIONS)
    logger.debug(f"Loaded {len(HARDCODED_EXCLUSIONS)} hardcoded exclusion patterns")

    # 2. Load from .gitignore (if exists)
    gitignore_path = cwd / ".gitignore"
    if gitignore_path.exists():
        try:
            gitignore_content = gitignore_path.read_text(encoding='utf-8')
            gitignore_patterns = [
                line.strip()
                for line in gitignore_content.splitlines()
                if line.strip() and not line.strip().startswith('#')
            ]
            patterns.extend(gitignore_patterns)
            logger.info(f"Loaded {len(gitignore_patterns)} patterns from .gitignore")
        except Exception as e:
            logger.warning(f"Failed to read .gitignore: {e}")
    else:
        logger.debug("No .gitignore file found")

    # 3. Load from settings override (if configured)
    if hasattr(settings, 'INDEXER_EXCLUDE_PATTERNS'):
        custom_patterns = settings.INDEXER_EXCLUDE_PATTERNS
        patterns.extend(custom_patterns)
        logger.debug(f"Loaded {len(custom_patterns)} custom exclusion patterns from settings")

    # 4. Create PathSpec matcher using gitwildmatch style
    try:
        spec = pathspec.PathSpec.from_lines('gitwildmatch', patterns)
        logger.info(f"Created PathSpec with {len(patterns)} total patterns")
        return spec
    except Exception as e:
        logger.error(f"Failed to create PathSpec: {e}")
        # Fallback to empty spec (no exclusions)
        return pathspec.PathSpec.from_lines('gitwildmatch', [])


def file_discovery_generator(
    cwd: Optional[Path] = None,
    extensions: Optional[List[str]] = None,
    exclude_spec: Optional[pathspec.PathSpec] = None,
    max_size_mb: Optional[int] = None,
    follow_symlinks: bool = False,
    show_progress: bool = False,
) -> Iterator[FileInfo]:
    """Discover indexable files in the codebase.

    This generator walks the directory tree and yields FileInfo objects for
    each file that passes all filters:
    - Not excluded by .gitignore patterns
    - Has an allowed file extension
    - Within size limit
    - Not a symlink pointing outside CWD (security)

    Args:
        cwd: Working directory to search (default: current directory)
        extensions: List of allowed extensions (default: from settings)
        exclude_spec: PathSpec for exclusions (default: load from .gitignore)
        max_size_mb: Maximum file size in MB (default: from settings)
        follow_symlinks: Whether to follow symbolic links (default: False)
        show_progress: Display progress bar using tqdm (default: False)

    Yields:
        FileInfo: Metadata for each discovered file

    Example:
        >>> for file_info in file_discovery_generator():
        ...     print(file_info.relative_path, file_info.size)
    """
    # Set defaults
    if cwd is None:
        cwd = CWD

    if extensions is None:
        extensions = getattr(settings, 'ALLOWED_FILE_EXTENSIONS', [
            '.py', '.js', '.ts', '.tsx', '.jsx',
            '.md', '.txt', '.json', '.yaml', '.yml'
        ])

    if exclude_spec is None:
        exclude_spec = load_gitignore_patterns(cwd)

    if max_size_mb is None:
        max_size_mb = settings.MAX_FILE_SIZE_MB

    max_size_bytes = max_size_mb * 1024 * 1024

    # Statistics
    file_count = 0
    skipped_excluded = 0
    skipped_extension = 0
    skipped_size = 0
    skipped_symlink = 0
    skipped_error = 0

    # Progress bar (optional)
    walker = os.walk(cwd, followlinks=follow_symlinks)
    if show_progress:
        try:
            from tqdm import tqdm
            # Estimate total directories for progress
            walker = tqdm(
                walker,
                desc="Scanning directories",
                unit="dir"
            )
        except ImportError:
            logger.debug("tqdm not available, skipping progress bar")

    # Walk directory tree
    for root, dirs, files in walker:
        root_path = Path(root)

        try:
            relative_root = root_path.relative_to(cwd)
        except ValueError:
            # Root is outside CWD, skip
            continue

        # Filter directories in-place to skip excluded subdirectories
        # This modifies the dirs list that os.walk uses
        original_dirs = dirs.copy()
        dirs[:] = []
        for d in original_dirs:
            dir_relative_path = relative_root / d if relative_root != Path('.') else Path(d)

            # Check if directory should be excluded
            if exclude_spec.match_file(str(dir_relative_path) + '/'):
                logger.debug(f"Excluding directory: {dir_relative_path}/")
                continue

            dirs.append(d)

        # Process files in current directory
        for filename in files:
            file_path = root_path / filename

            try:
                relative_path = file_path.relative_to(cwd)
            except ValueError:
                # File is outside CWD, skip
                skipped_error += 1
                continue

            try:
                # 1. Security check: Ensure not a symlink pointing outside CWD
                if file_path.is_symlink():
                    if not follow_symlinks:
                        # Skip all symlinks when follow_symlinks=False
                        skipped_symlink += 1
                        continue

                    # If following symlinks, ensure target is within CWD
                    try:
                        # Use existing validate_path logic for security
                        validate_path(str(file_path), must_exist=True)
                    except PathValidationError:
                        logger.warning(f"Skipping symlink outside CWD: {relative_path}")
                        skipped_symlink += 1
                        continue

                # 2. Check exclusion patterns
                if exclude_spec.match_file(str(relative_path)):
                    skipped_excluded += 1
                    continue

                # 3. Check file extension
                if file_path.suffix.lower() not in extensions:
                    skipped_extension += 1
                    continue

                # 4. Check file size
                stat = file_path.stat()
                if stat.st_size > max_size_bytes:
                    size_mb = stat.st_size / (1024 * 1024)
                    logger.warning(
                        f"Skipping large file: {relative_path} "
                        f"({size_mb:.2f}MB > {max_size_mb}MB)"
                    )
                    skipped_size += 1
                    continue

                # 5. All checks passed - yield FileInfo
                yield FileInfo(
                    path=file_path,
                    relative_path=relative_path,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                )
                file_count += 1

            except (OSError, PermissionError) as e:
                logger.warning(f"Cannot access {relative_path}: {e}")
                skipped_error += 1
                continue

    # Log statistics
    logger.info(
        f"File discovery complete: {file_count} files found, "
        f"{skipped_excluded} excluded by patterns, "
        f"{skipped_extension} wrong extension, "
        f"{skipped_size} too large, "
        f"{skipped_symlink} symlinks skipped, "
        f"{skipped_error} errors"
    )
