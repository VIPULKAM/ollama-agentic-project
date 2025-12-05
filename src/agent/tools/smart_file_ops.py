"""Smart file operations for handling large files efficiently.

This module provides tools that can update specific sections of large files
without loading the entire content into the LLM's context window.
"""

from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import re

from .file_ops import validate_path
from ...utils.logging import file_ops_logger
from ...config.settings import settings


class SmartReadFileInput(BaseModel):
    """Input schema for SmartReadFileTool."""
    path: str = Field(description="Path to the file to read")
    mode: str = Field(
        default="auto",
        description="Reading mode: 'auto', 'full', 'lines', 'pattern'"
    )
    start_line: Optional[int] = Field(
        default=None,
        description="Start line number (1-indexed) for 'lines' mode"
    )
    end_line: Optional[int] = Field(
        default=None,
        description="End line number (inclusive) for 'lines' mode"
    )
    pattern: Optional[str] = Field(
        default=None,
        description="Regex pattern to search for in 'pattern' mode"
    )
    context_lines: int = Field(
        default=5,
        description="Number of context lines around pattern matches"
    )
    max_size_kb: int = Field(
        default=10,
        description="Max file size (KB) for full read in 'auto' mode"
    )


class SmartReadFileTool(BaseTool):
    """Read files with intelligent context management.

    For small files (< 10KB): reads full content.
    For large files: requires line range or pattern to avoid context overflow.
    """

    name: str = "read_file_smart"
    description: str = """Read files with intelligent context management.

Use this tool to read files efficiently, especially large ones:

Modes:
- 'auto': Automatically decides based on file size (< 10KB = full, else requires range/pattern)
- 'full': Read entire file (use for small files only)
- 'lines': Read specific line range (start_line to end_line)
- 'pattern': Search for regex pattern and return matches with context

Args:
- path: File path
- mode: Reading mode (default: 'auto')
- start_line: Start line for 'lines' mode (1-indexed)
- end_line: End line for 'lines' mode (inclusive)
- pattern: Regex pattern for 'pattern' mode
- context_lines: Context lines around matches (default: 5)
- max_size_kb: Max size for full read in auto mode (default: 10KB)

Examples:
1. Auto mode (decides based on size):
   read_file_smart(path="config.json", mode="auto")

2. Read specific lines:
   read_file_smart(path="CLAUDE.md", mode="lines", start_line=100, end_line=200)

3. Search for pattern with context:
   read_file_smart(path="app.py", mode="pattern", pattern="class.*Agent", context_lines=5)
"""
    args_schema: type = SmartReadFileInput

    def _run(
        self,
        path: str,
        mode: str = "auto",
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        pattern: Optional[str] = None,
        context_lines: int = 5,
        max_size_kb: int = 10
    ) -> str:
        """Read file with context management."""

        # Validate path
        try:
            validated_path = validate_path(path)
            if not validated_path.exists():
                return f"Error: File not found: {path}"
        except Exception as e:
            return f"Error: Invalid path: {str(e)}"

        # Get file size
        file_size_kb = validated_path.stat().st_size / 1024

        # Auto mode: decide based on file size
        if mode == "auto":
            if file_size_kb <= max_size_kb:
                mode = "full"
                file_ops_logger.info(
                    f"SMART_READ: Auto mode selected 'full' for {path} ({file_size_kb:.1f}KB)"
                )
            elif pattern:
                mode = "pattern"
                file_ops_logger.info(
                    f"SMART_READ: Auto mode selected 'pattern' for {path} ({file_size_kb:.1f}KB)"
                )
            elif start_line and end_line:
                mode = "lines"
                file_ops_logger.info(
                    f"SMART_READ: Auto mode selected 'lines' for {path} ({file_size_kb:.1f}KB)"
                )
            else:
                return (
                    f"⚠️ File too large ({file_size_kb:.1f}KB > {max_size_kb}KB)\n\n"
                    f"Please specify:\n"
                    f"- Line range: start_line and end_line\n"
                    f"- Search pattern: pattern parameter\n\n"
                    f"Example: read_file_smart(path='{path}', mode='lines', start_line=1, end_line=50)"
                )

        # Read file
        try:
            content = validated_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                content = validated_path.read_text(encoding='latin-1')
            except Exception as e:
                return f"Error: Unable to read file with any encoding: {str(e)}"
        except Exception as e:
            return f"Error reading file: {str(e)}"

        lines = content.split('\n')
        total_lines = len(lines)

        # Full mode
        if mode == "full":
            file_ops_logger.info(f"SMART_READ: Full read of {path} ({total_lines} lines)")
            return self._format_full_content(path, lines, file_size_kb)

        # Lines mode
        if mode == "lines":
            if not start_line or not end_line:
                return "Error: 'lines' mode requires start_line and end_line parameters"

            if start_line < 1 or end_line > total_lines or start_line > end_line:
                return (
                    f"Error: Invalid line range. File has {total_lines} lines.\n"
                    f"Requested: lines {start_line}-{end_line}"
                )

            file_ops_logger.info(
                f"SMART_READ: Lines {start_line}-{end_line} of {path} ({total_lines} total)"
            )
            return self._format_line_range(path, lines, start_line, end_line, total_lines)

        # Pattern mode
        if mode == "pattern":
            if not pattern:
                return "Error: 'pattern' mode requires pattern parameter"

            try:
                matches = self._find_pattern_matches(lines, pattern, context_lines)
                file_ops_logger.info(
                    f"SMART_READ: Pattern '{pattern}' in {path} - {len(matches)} matches"
                )
                return self._format_pattern_matches(path, matches, pattern, total_lines)
            except re.error as e:
                return f"Error: Invalid regex pattern '{pattern}': {str(e)}"

        return f"Error: Invalid mode '{mode}'. Use 'auto', 'full', 'lines', or 'pattern'"

    def _format_full_content(self, path: str, lines: List[str], file_size_kb: float) -> str:
        """Format full file content."""
        numbered_lines = '\n'.join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
        return f"""📄 **File: {path}**
Size: {file_size_kb:.1f}KB | Lines: {len(lines)}

{numbered_lines}"""

    def _format_line_range(
        self,
        path: str,
        lines: List[str],
        start: int,
        end: int,
        total: int
    ) -> str:
        """Format specific line range."""
        selected_lines = lines[start-1:end]
        numbered_lines = '\n'.join(
            f"{i:4d} | {line}" for i, line in enumerate(selected_lines, start=start)
        )

        return f"""📄 **File: {path}**
Lines: {start}-{end} of {total}

{numbered_lines}"""

    def _find_pattern_matches(
        self,
        lines: List[str],
        pattern: str,
        context_lines: int
    ) -> List[dict]:
        """Find all pattern matches with context."""
        matches = []

        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                # Calculate context range
                start_ctx = max(1, i - context_lines)
                end_ctx = min(len(lines), i + context_lines)

                matches.append({
                    'line_num': i,
                    'line': line,
                    'context_start': start_ctx,
                    'context_end': end_ctx,
                    'context': lines[start_ctx-1:end_ctx]
                })

        return matches

    def _format_pattern_matches(
        self,
        path: str,
        matches: List[dict],
        pattern: str,
        total_lines: int
    ) -> str:
        """Format pattern match results."""
        if not matches:
            return f"""📄 **File: {path}**
Pattern: `{pattern}`
Total lines: {total_lines}

❌ No matches found"""

        result = f"""📄 **File: {path}**
Pattern: `{pattern}`
Matches: {len(matches)} | Total lines: {total_lines}

"""

        for idx, match in enumerate(matches, 1):
            result += f"\n--- Match {idx} at line {match['line_num']} ---\n"

            # Show context with match highlighted
            for i, ctx_line in enumerate(match['context'], start=match['context_start']):
                line_prefix = ">>>" if i == match['line_num'] else "   "
                result += f"{line_prefix} {i:4d} | {ctx_line}\n"

        return result


class UpdateFileSectionInput(BaseModel):
    """Input schema for UpdateFileSectionTool."""
    path: str = Field(description="Path to the file to update")
    start_marker: str = Field(
        description="Section start marker (e.g., '## Future Enhancements')"
    )
    end_marker: Optional[str] = Field(
        default=None,
        description="Section end marker (next heading, or None for auto-detect)"
    )
    new_content: str = Field(description="New content for this section")
    mode: str = Field(
        default="replace",
        description="Update mode: 'replace', 'append', 'prepend'"
    )


class UpdateFileSectionTool(BaseTool):
    """Update a specific section of a file using markers.

    Perfect for large files where reading the entire content would exceed context window.
    """

    name: str = "update_file_section"
    description: str = """Update a specific section of a file using start/end markers.

Use this for large files (> 10KB) to update specific sections without loading entire file.

Args:
- path: File path (e.g., "CLAUDE.md")
- start_marker: Section start (e.g., "## Future Enhancements")
- end_marker: Section end (optional, auto-detects next heading)
- new_content: New content for the section
- mode: 'replace' (default), 'append', or 'prepend'

Example: Update Future Enhancements section in CLAUDE.md
"""
    args_schema: type = UpdateFileSectionInput

    def _run(
        self,
        path: str,
        start_marker: str,
        end_marker: Optional[str] = None,
        new_content: str = "",
        mode: str = "replace"
    ) -> str:
        """Update a file section."""

        # Validate path
        validated_path = validate_path(path)
        if not validated_path.exists():
            return f"Error: File not found: {path}"

        # Read file
        try:
            content = validated_path.read_text(encoding='utf-8')
            lines = content.split('\n')
        except Exception as e:
            file_ops_logger.error(f"Failed to read {path}: {e}")
            return f"Error reading file: {str(e)}"

        # Find start marker
        start_line = None
        for i, line in enumerate(lines):
            if start_marker in line:
                start_line = i
                break

        if start_line is None:
            return f"Error: Start marker '{start_marker}' not found in file"

        # Find end marker
        end_line = None

        if end_marker:
            # User provided explicit end marker
            for i in range(start_line + 1, len(lines)):
                if end_marker in lines[i]:
                    end_line = i
                    break
        else:
            # Auto-detect end marker for markdown headings
            if start_marker.strip().startswith('#'):
                # Extract heading level
                match = re.match(r'^#+', start_marker.strip())
                if match:
                    heading_level = len(match.group())

                    # Find next heading of same or higher level
                    for i in range(start_line + 1, len(lines)):
                        line_stripped = lines[i].strip()
                        if line_stripped.startswith('#'):
                            next_match = re.match(r'^#+', line_stripped)
                            if next_match:
                                next_level = len(next_match.group())
                                if next_level <= heading_level:
                                    end_line = i
                                    break

        # If no end marker found, use EOF
        if end_line is None:
            end_line = len(lines)

        # Create backup
        from datetime import datetime
        backup_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = validated_path.parent / f"{validated_path.stem}.bak.{backup_suffix}"

        try:
            validated_path.rename(backup_path)
            validated_path.write_text(content, encoding='utf-8')  # Restore original
            file_ops_logger.info(f"BACKUP: Created backup at {backup_path.name}")
        except Exception as e:
            return f"Error creating backup: {str(e)}"

        # Build new section content
        if mode == "replace":
            # Replace entire section
            new_section_lines = [lines[start_line]]  # Keep the heading
            if new_content.strip():
                new_section_lines.append("")  # Blank line
                new_section_lines.extend(new_content.split('\n'))
                new_section_lines.append("")  # Blank line
            new_section = '\n'.join(new_section_lines)

        elif mode == "append":
            # Append to existing section
            section_content = '\n'.join(lines[start_line+1:end_line])
            new_section = lines[start_line] + '\n' + section_content + '\n' + new_content + '\n'

        elif mode == "prepend":
            # Prepend to existing section
            section_content = '\n'.join(lines[start_line+1:end_line])
            new_section = lines[start_line] + '\n' + new_content + '\n' + section_content + '\n'
        else:
            return f"Error: Invalid mode '{mode}'. Use 'replace', 'append', or 'prepend'"

        # Build new file content
        new_lines = lines[:start_line] + new_section.split('\n') + lines[end_line:]
        new_content_full = '\n'.join(new_lines)

        # Write updated content
        try:
            validated_path.write_text(new_content_full, encoding='utf-8')
            file_size = validated_path.stat().st_size

            file_ops_logger.info(
                f"UPDATE_SECTION: {path} (marker: '{start_marker}', mode: {mode}, "
                f"size: {file_size} bytes, backup: {backup_path.name})"
            )

            return f"""✓ Successfully updated section in {path}

**Section**: {start_marker}
**Mode**: {mode}
**File size**: {file_size} bytes
**Backup**: {backup_path.name}

Section updated successfully!"""

        except Exception as e:
            # Restore from backup on error
            try:
                backup_content = backup_path.read_text(encoding='utf-8')
                validated_path.write_text(backup_content, encoding='utf-8')
                file_ops_logger.error(f"Update failed, restored from backup: {e}")
            except Exception as restore_error:
                file_ops_logger.error(f"Failed to restore from backup: {restore_error}")

            return f"Error updating file (restored from backup): {str(e)}"


def get_smart_read_file_tool() -> SmartReadFileTool:
    """Factory function to create SmartReadFileTool instance."""
    return SmartReadFileTool()


def get_update_file_section_tool() -> UpdateFileSectionTool:
    """Factory function to create UpdateFileSectionTool instance."""
    return UpdateFileSectionTool()
