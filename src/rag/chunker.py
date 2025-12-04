"""Code and text chunking for RAG indexing.

This module provides intelligent chunking strategies:
- CodeChunker: AST-based chunking for Python using tree-sitter
- TextChunker: Sliding window fallback for non-code files
- chunk_file: Orchestration function that selects the appropriate chunker
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..config.settings import settings

# Configure logging
logger = logging.getLogger("ai_agent.chunker")


@dataclass
class CodeChunk:
    """A chunk of code or text with metadata.

    This dataclass represents a single chunk extracted from a file.
    It includes both the content and metadata needed for indexing
    and retrieval.

    Attributes:
        content: The actual text content of the chunk
        file_path: Path to the source file (relative to CWD)
        start_line: Starting line number in the source file (1-indexed)
        end_line: Ending line number in the source file (1-indexed)
        chunk_type: Type of chunk ("function", "class", "text", etc.)
        language: Programming language or file type ("python", "markdown", etc.)
        metadata: Additional metadata (function name, class name, etc.)
    """
    content: str
    file_path: Path
    start_line: int
    end_line: int
    chunk_type: str = "text"
    language: str = "unknown"
    metadata: Optional[dict] = None

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"CodeChunk({self.chunk_type} in {self.file_path}:"
            f"{self.start_line}-{self.end_line}, {len(self.content)} chars)"
        )


class CodeChunker:
    """AST-based chunker for Python code using tree-sitter.

    This chunker uses tree-sitter to parse Python code and extract
    meaningful units like functions and classes with full context.
    """

    def __init__(self):
        """Initialize the CodeChunker with tree-sitter parser."""
        try:
            from tree_sitter import Language, Parser
            import tree_sitter_python
        except ImportError as e:
            raise ImportError(
                "tree-sitter dependencies not installed. "
                "Install with: pip install tree-sitter tree-sitter-python"
            ) from e

        # Initialize Python parser
        self.language = Language(tree_sitter_python.language())
        self.parser = Parser(self.language)

        logger.debug("CodeChunker initialized with tree-sitter")

    def chunk(
        self,
        content: str,
        file_path: Path,
        include_context: bool = True
    ) -> List[CodeChunk]:
        """Chunk Python code into functions and classes.

        Args:
            content: Python source code as string
            file_path: Path to the source file (for metadata)
            include_context: Include imports and class context (default: True)

        Returns:
            List[CodeChunk]: List of extracted chunks

        Example:
            >>> chunker = CodeChunker()
            >>> chunks = chunker.chunk(python_code, Path("script.py"))
            >>> for chunk in chunks:
            ...     print(chunk.chunk_type, chunk.start_line)
        """
        chunks = []

        try:
            # Parse the code
            tree = self.parser.parse(bytes(content, 'utf-8'))
            root_node = tree.root_node

            # Extract imports for context (if enabled)
            imports_text = ""
            if include_context:
                imports_text = self._extract_imports(root_node, content)

            # Extract functions and classes
            self._extract_functions_and_classes(
                root_node,
                content,
                file_path,
                imports_text,
                chunks
            )

            logger.debug(f"Extracted {len(chunks)} chunks from {file_path}")

        except Exception as e:
            logger.warning(f"Failed to parse {file_path} with tree-sitter: {e}")
            # Return empty list - fallback chunker will be used
            return []

        return chunks

    def _extract_imports(self, root_node, content: str) -> str:
        """Extract import statements from the file."""
        imports = []
        for node in root_node.children:
            if node.type in ['import_statement', 'import_from_statement']:
                import_text = content[node.start_byte:node.end_byte]
                imports.append(import_text)

        return "\n".join(imports) if imports else ""

    def _extract_functions_and_classes(
        self,
        root_node,
        content: str,
        file_path: Path,
        imports_text: str,
        chunks: List[CodeChunk]
    ):
        """Recursively extract function and class definitions."""

        def visit(node, parent_class_name=None, parent_class_def=None):
            """Visit AST nodes recursively."""

            # Extract class definitions
            if node.type == 'class_definition':
                class_name_node = node.child_by_field_name('name')
                class_name = content[class_name_node.start_byte:class_name_node.end_byte] if class_name_node else "UnnamedClass"

                # Get class content (including decorators and docstring)
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                class_content = content[node.start_byte:node.end_byte]

                # Add imports context if available
                if imports_text:
                    chunk_content = f"{imports_text}\n\n{class_content}"
                else:
                    chunk_content = class_content

                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type="class",
                    language="python",
                    metadata={"class_name": class_name}
                ))

                # Recurse into class body to find methods
                for child in node.children:
                    visit(child, parent_class_name=class_name, parent_class_def=class_content)

            # Extract function/method definitions
            elif node.type == 'function_definition':
                func_name_node = node.child_by_field_name('name')
                func_name = content[func_name_node.start_byte:func_name_node.end_byte] if func_name_node else "unnamed_function"

                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                func_content = content[node.start_byte:node.end_byte]

                # Build context: imports + class definition (if method)
                chunk_content_parts = []
                if imports_text:
                    chunk_content_parts.append(imports_text)

                if parent_class_def:
                    # Extract just the class signature (first line)
                    class_signature = parent_class_def.split('\n')[0]
                    chunk_content_parts.append(f"{class_signature}\n    # ... class body ...")

                chunk_content_parts.append(func_content)
                chunk_content = "\n\n".join(chunk_content_parts)

                chunk_type = "method" if parent_class_name else "function"
                metadata = {"function_name": func_name}
                if parent_class_name:
                    metadata["class_name"] = parent_class_name

                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type=chunk_type,
                    language="python",
                    metadata=metadata
                ))

            # Recurse into child nodes
            else:
                for child in node.children:
                    visit(child, parent_class_name, parent_class_def)

        # Start visiting from root
        visit(root_node)


class TextChunker:
    """Fallback chunker using sliding window for non-code or unparseable files.

    This chunker uses a simple sliding window approach to split text into
    overlapping chunks. It's used for:
    - Markdown files
    - Text files
    - Code files that fail AST parsing
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_chunk_size: Optional[int] = None
    ):
        """Initialize the TextChunker.

        Args:
            chunk_size: Size of each chunk in characters (default: from settings)
            chunk_overlap: Overlap between chunks (default: from settings)
            min_chunk_size: Minimum chunk size to avoid tiny chunks (default: 50)
        """
        self.chunk_size = chunk_size or getattr(settings, 'CHUNK_SIZE', 500)
        self.chunk_overlap = chunk_overlap or getattr(settings, 'CHUNK_OVERLAP', 50)
        self.min_chunk_size = min_chunk_size or getattr(settings, 'CHUNK_MIN_SIZE', 50)

        logger.debug(
            f"TextChunker initialized: size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}, min={self.min_chunk_size}"
        )

    def chunk(self, content: str, file_path: Path) -> List[CodeChunk]:
        """Chunk text using sliding window approach.

        Args:
            content: Text content to chunk
            file_path: Path to the source file (for metadata)

        Returns:
            List[CodeChunk]: List of text chunks
        """
        chunks = []

        # Determine language/type from file extension
        language = file_path.suffix.lstrip('.') or "text"

        # Split content into lines for accurate line number tracking
        lines = content.splitlines(keepends=True)
        total_chars = sum(len(line) for line in lines)

        if total_chars < self.min_chunk_size:
            # File is too small, create single chunk
            chunks.append(CodeChunk(
                content=content,
                file_path=file_path,
                start_line=1,
                end_line=len(lines),
                chunk_type="text",
                language=language
            ))
            return chunks

        # Sliding window chunking
        char_position = 0
        current_line = 1

        while char_position < total_chars:
            # Calculate chunk end position
            chunk_end = min(char_position + self.chunk_size, total_chars)

            # Extract chunk text
            chunk_text = ""
            chunk_start_line = current_line
            chars_collected = 0
            chunk_end_line = current_line

            for line_idx in range(current_line - 1, len(lines)):
                line = lines[line_idx]
                if chars_collected + len(line) > self.chunk_size:
                    break
                chunk_text += line
                chars_collected += len(line)
                chunk_end_line = line_idx + 1

            # Only add chunk if it meets minimum size
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(CodeChunk(
                    content=chunk_text,
                    file_path=file_path,
                    start_line=chunk_start_line,
                    end_line=chunk_end_line,
                    chunk_type="text",
                    language=language
                ))

            # Move window forward (with overlap)
            char_position += (self.chunk_size - self.chunk_overlap)

            # Update current line
            overlap_chars = self.chunk_overlap
            for line_idx in range(current_line - 1, len(lines)):
                if overlap_chars <= 0:
                    current_line = line_idx + 1
                    break
                overlap_chars -= len(lines[line_idx])

        logger.debug(f"Created {len(chunks)} text chunks from {file_path}")
        return chunks


# Module-level chunker instances (cached)
_code_chunker = None
_text_chunker = None


def get_code_chunker() -> CodeChunker:
    """Get cached CodeChunker instance."""
    global _code_chunker
    if _code_chunker is None:
        _code_chunker = CodeChunker()
    return _code_chunker


def get_text_chunker() -> TextChunker:
    """Get cached TextChunker instance."""
    global _text_chunker
    if _text_chunker is None:
        _text_chunker = TextChunker()
    return _text_chunker


def chunk_file(file_path: Path, content: Optional[str] = None) -> List[CodeChunk]:
    """Orchestration function to chunk a file using the appropriate strategy.

    This function determines whether to use CodeChunker (for Python) or
    TextChunker (for everything else) based on the file extension.

    Args:
        file_path: Path to the file to chunk
        content: File content (if None, will be read from file_path)

    Returns:
        List[CodeChunk]: List of chunks extracted from the file

    Raises:
        FileNotFoundError: If file doesn't exist and content not provided
        IOError: If file cannot be read

    Example:
        >>> chunks = chunk_file(Path("script.py"))
        >>> for chunk in chunks:
        ...     print(f"{chunk.chunk_type}: {chunk.content[:50]}...")
    """
    # Read content if not provided
    if content is None:
        try:
            content = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Try with latin-1 fallback
            logger.warning(f"UTF-8 decode failed for {file_path}, trying latin-1")
            content = file_path.read_text(encoding='latin-1')

    # Determine chunking strategy based on file extension
    extension = file_path.suffix.lower()

    # Try CodeChunker for Python files
    if extension == '.py':
        try:
            code_chunker = get_code_chunker()
            chunks = code_chunker.chunk(content, file_path)

            # If CodeChunker returns empty list, fall back to TextChunker
            if chunks:
                logger.debug(f"Used CodeChunker for {file_path}: {len(chunks)} chunks")
                return chunks
            else:
                logger.debug(f"CodeChunker failed for {file_path}, using TextChunker fallback")

        except ImportError:
            # tree-sitter not available, fall back to TextChunker
            logger.warning("tree-sitter not available, using TextChunker for Python files")
        except Exception as e:
            # AST parsing failed, fall back to TextChunker
            logger.warning(f"CodeChunker failed for {file_path}: {e}, using TextChunker fallback")

    # Use TextChunker for non-Python files or fallback
    text_chunker = get_text_chunker()
    chunks = text_chunker.chunk(content, file_path)
    logger.debug(f"Used TextChunker for {file_path}: {len(chunks)} chunks")

    return chunks
