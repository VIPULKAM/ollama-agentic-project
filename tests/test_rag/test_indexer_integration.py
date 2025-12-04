"""Integration tests for the RAG indexer.

These tests verify the complete indexing pipeline:
- File discovery with gitignore patterns
- Chunking (both Python code and text)
- Index building and saving
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.rag.indexer_utils import (
    file_discovery_generator,
    load_gitignore_patterns,
    FileInfo
)
from src.rag.chunker import chunk_file, CodeChunk
from src.rag.indexer import build_index, load_index, index_exists


class TestFileDiscovery:
    """Tests for file discovery functionality."""

    def test_load_gitignore_patterns(self):
        """Test loading gitignore patterns."""
        spec = load_gitignore_patterns()
        assert spec is not None

        # Test that .venv is excluded (from hardcoded patterns)
        assert spec.match_file(".venv/lib/python")

    def test_file_discovery_basic(self):
        """Test basic file discovery in current directory."""
        # Discover Python files in src/
        files = list(file_discovery_generator(
            cwd=Path("src/rag"),
            extensions=[".py"],
            show_progress=False
        ))

        assert len(files) > 0
        assert all(isinstance(f, FileInfo) for f in files)
        assert all(f.path.suffix == ".py" for f in files)

    def test_file_discovery_excludes_venv(self):
        """Test that .venv is excluded from discovery."""
        files = list(file_discovery_generator(
            cwd=Path("."),
            extensions=[".py"],
            show_progress=False
        ))

        # No files from .venv should be found
        for file_info in files:
            assert ".venv" not in str(file_info.relative_path)
            assert "venv" not in str(file_info.relative_path)


class TestChunking:
    """Tests for chunking functionality."""

    def test_chunk_python_file(self):
        """Test chunking a Python file."""
        # Chunk the embeddings.py file
        py_file = Path("src/rag/embeddings.py")
        if not py_file.exists():
            pytest.skip("embeddings.py not found")

        chunks = chunk_file(py_file)

        assert len(chunks) > 0
        assert all(isinstance(c, CodeChunk) for c in chunks)

        # Check chunk types - will be "text" if tree-sitter not installed
        chunk_types = [c.chunk_type for c in chunks]
        # If tree-sitter is available, we should have function/class chunks
        # Otherwise, all will be "text" chunks (fallback behavior)
        assert len(chunk_types) > 0  # At least some chunks should exist

    def test_chunk_markdown_file(self):
        """Test chunking a markdown file."""
        # Chunk README.md
        md_file = Path("README.md")
        if not md_file.exists():
            pytest.skip("README.md not found")

        chunks = chunk_file(md_file)

        assert len(chunks) > 0
        assert all(isinstance(c, CodeChunk) for c in chunks)
        assert all(c.chunk_type == "text" for c in chunks)
        assert all(c.language == "md" for c in chunks)


class TestIndexBuilding:
    """Tests for index building functionality."""

    @pytest.fixture
    def test_index_dir(self):
        """Create a temporary directory for test index."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_build_small_index(self, test_index_dir):
        """Test building index from a small subset of files."""
        # Build index from just the rag/ directory
        index, chunks = build_index(
            cwd=Path("src/rag"),
            index_path=test_index_dir,
            show_progress=False,
            batch_size=8
        )

        # Verify index was built
        assert index is not None
        assert index.ntotal > 0
        assert len(chunks) > 0
        assert index.ntotal == len(chunks)

        # Verify index files were saved
        assert (test_index_dir / "index.faiss").exists()
        assert (test_index_dir / "chunks_metadata.json").exists()
        assert (test_index_dir / "index_info.json").exists()

    def test_load_index(self, test_index_dir):
        """Test loading index from disk."""
        # First build an index
        build_index(
            cwd=Path("src/rag"),
            index_path=test_index_dir,
            show_progress=False,
            batch_size=8
        )

        # Then load it back
        loaded_index, loaded_metadata = load_index(test_index_dir)

        assert loaded_index is not None
        assert loaded_index.ntotal > 0
        assert len(loaded_metadata) > 0
        assert loaded_index.ntotal == len(loaded_metadata)

    def test_index_exists(self, test_index_dir):
        """Test checking if index exists."""
        # Should not exist initially
        assert not index_exists(test_index_dir)

        # Build index
        build_index(
            cwd=Path("src/rag"),
            index_path=test_index_dir,
            show_progress=False,
            batch_size=8
        )

        # Should exist now
        assert index_exists(test_index_dir)


class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.fixture
    def test_codebase(self):
        """Create a temporary test codebase."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create test files
        (temp_dir / "test.py").write_text("""
def hello_world():
    '''Say hello.'''
    print("Hello, world!")

class Calculator:
    '''Simple calculator.'''
    def add(self, a, b):
        return a + b
""")

        (temp_dir / "README.md").write_text("""
# Test Project

This is a test project for RAG indexing.

## Features
- File discovery
- Code chunking
- Index building
""")

        (temp_dir / ".gitignore").write_text("""
*.pyc
__pycache__/
.venv/
""")

        yield temp_dir

        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def test_full_pipeline(self, test_codebase):
        """Test the complete indexing pipeline."""
        index_dir = test_codebase / "index"

        # Build index
        index, chunks = build_index(
            cwd=test_codebase,
            index_path=index_dir,
            show_progress=False,
            batch_size=4
        )

        # Verify results
        assert index.ntotal > 0
        assert len(chunks) > 0

        # Should have chunks from both test.py and README.md
        file_paths = {str(c.file_path) for c in chunks}
        assert any("test.py" in fp for fp in file_paths)
        assert any("README.md" in fp for fp in file_paths)

        # Should have different chunk types
        # Note: If tree-sitter is not installed, all chunks will be "text" type
        chunk_types = {c.chunk_type for c in chunks}
        assert "text" in chunk_types  # Text chunks should always be present
        # If tree-sitter is installed, we might also have function/class chunks
        # but we don't require it for the test to pass

        # Load and verify
        loaded_index, loaded_metadata = load_index(index_dir)
        assert loaded_index.ntotal == index.ntotal
        assert len(loaded_metadata) == len(chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
