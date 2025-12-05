"""Smart file operations for handling large files efficiently.

This module provides tools that can update specific sections of large files
without loading the entire content into the LLM's context window.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
import re

from .file_ops import validate_path
from ...utils.logging import file_ops_logger


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


def get_update_file_section_tool() -> UpdateFileSectionTool:
    """Factory function to create UpdateFileSectionTool instance."""
    return UpdateFileSectionTool()
