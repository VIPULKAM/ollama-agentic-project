"""Tests for index management utilities."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.rag.index_manager import (
    get_index_info,
    rebuild_index,
    clean_index,
    search_index,
    IndexNotFoundError
)


class TestGetIndexInfo:
    """Tests for get_index_info() function."""

    def test_index_not_found(self):
        """Test when index does not exist."""
        with patch('src.rag.index_manager.index_exists', return_value=False):
            info = get_index_info()

            assert info["exists"] is False
            assert "message" in info

    def test_index_info_success(self, tmp_path):
        """Test successful index info retrieval."""
        # Create mock index files
        index_path = tmp_path / "faiss_index"
        index_path.mkdir()

        # Create metadata
        chunks = [
            {
                "file_path": str(tmp_path / "file1.py"),
                "content": "test",
                "language": "python"
            },
            {
                "file_path": str(tmp_path / "file2.py"),
                "content": "test",
                "language": "python"
            }
        ]

        with open(index_path / "chunks_metadata.json", 'w') as f:
            json.dump(chunks, f)

        # Create index info
        index_info = {
            "version": "1.0",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }

        with open(index_path / "index_info.json", 'w') as f:
            json.dump(index_info, f)

        # Create dummy index file
        (index_path / "index.faiss").touch()

        with patch('src.rag.index_manager.index_exists', return_value=True):
            with patch('src.rag.index_manager.settings') as mock_settings:
                mock_settings.FAISS_INDEX_PATH = str(index_path)
                mock_settings.CRAWLED_DOCS_PATH = str(tmp_path / "crawled_docs")

                with patch('src.rag.index_manager.get_crawl_tracker') as mock_tracker:
                    mock_tracker.return_value.get_stats.return_value = {
                        "total_urls": 0,
                        "total_chunks": 0
                    }

                    info = get_index_info()

                    assert info["exists"] is True
                    assert info["total_chunks"] == 2
                    assert info["total_files"] == 2
                    assert info["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"


class TestRebuildIndex:
    """Tests for rebuild_index() function."""

    def test_rebuild_success(self, tmp_path):
        """Test successful index rebuild."""
        with patch('src.rag.index_manager.build_index') as mock_build:
            # Mock build_index to return test data (returns tuple of index, chunks)
            from src.rag.chunker import CodeChunk

            chunks = [
                CodeChunk("chunk1", Path("test.py"), 1, 10, "function", "python"),
                CodeChunk("chunk2", Path("test.py"), 11, 20, "function", "python")
            ]

            mock_build.return_value = (MagicMock(), chunks)

            result = rebuild_index(root_path=tmp_path, show_progress=False)

            assert result["success"] is True
            assert result["chunks_indexed"] == 2
            assert result["files_processed"] == 1
            assert result["errors"] == []
            assert result["duration_seconds"] >= 0

    def test_rebuild_failure(self, tmp_path):
        """Test rebuild failure handling."""
        with patch('src.rag.index_manager.build_index', side_effect=Exception("Test error")):
            result = rebuild_index(root_path=tmp_path, show_progress=False)

            assert result["success"] is False
            assert result["chunks_indexed"] == 0
            assert "Test error" in result["errors"][0]


class TestCleanIndex:
    """Tests for clean_index() function."""

    def test_clean_no_index(self):
        """Test cleaning when no index exists."""
        with patch('src.rag.index_manager.index_exists', return_value=False):
            with pytest.raises(IndexNotFoundError):
                clean_index()

    def test_clean_no_orphans(self, tmp_path):
        """Test cleaning when no orphaned chunks."""
        # Create a real file
        test_file = tmp_path / "test.py"
        test_file.write_text("# test")

        chunks = [
            {
                "file_path": str(test_file),
                "content": "test"
            }
        ]

        mock_index = MagicMock()

        with patch('src.rag.index_manager.index_exists', return_value=True):
            with patch('src.rag.index_manager.load_index', return_value=(mock_index, chunks)):
                result = clean_index()

                assert result["success"] is True
                assert result["chunks_removed"] == 0
                assert result["chunks_remaining"] == 1

    def test_clean_with_orphans(self, tmp_path):
        """Test cleaning with orphaned chunks."""
        # Create one existing file
        existing_file = tmp_path / "exists.py"
        existing_file.write_text("# exists")

        # Reference a non-existent file
        deleted_file = tmp_path / "deleted.py"

        chunks = [
            {"file_path": str(existing_file), "content": "test1"},
            {"file_path": str(deleted_file), "content": "test2"}  # This file doesn't exist
        ]

        mock_index = MagicMock()

        with patch('src.rag.index_manager.index_exists', return_value=True):
            with patch('src.rag.index_manager.load_index', return_value=(mock_index, chunks)):
                with patch('src.rag.embeddings.get_embeddings', return_value=[[0.1] * 384]):
                    import faiss
                    mock_new_index = MagicMock()
                    with patch.object(faiss, 'IndexFlatL2', return_value=mock_new_index):
                        with patch('src.rag.index_manager.save_index'):
                            with patch('src.rag.index_manager.settings') as mock_settings:
                                mock_settings.FAISS_INDEX_PATH = str(tmp_path / "index")
                                result = clean_index()

                                assert result["success"] is True
                                assert result["chunks_removed"] == 1
                                assert result["chunks_remaining"] == 1
                                assert str(deleted_file) in result["files_removed"]


class TestSearchIndex:
    """Tests for search_index() function."""

    def test_search_no_index(self):
        """Test search when no index exists."""
        with patch('src.rag.index_manager.index_exists', return_value=False):
            with pytest.raises(IndexNotFoundError):
                search_index("test query")

    def test_search_success(self):
        """Test successful search."""
        mock_results = [
            {
                "content": "test result",
                "file_path": "/test/file.py",
                "score": 0.8,
                "language": "python",
                "chunk_type": "function"
            }
        ]

        with patch('src.rag.index_manager.index_exists', return_value=True):
            with patch('src.rag.index_manager.SemanticRetriever') as mock_retriever_class:
                mock_retriever = MagicMock()
                mock_retriever.search.return_value = mock_results
                mock_retriever_class.return_value = mock_retriever

                with patch('src.rag.index_manager.settings') as mock_settings:
                    results = search_index("test query", top_k=10, threshold=0.5)

                    assert len(results) == 1
                    assert results[0]["content"] == "test result"
                    assert results[0]["score"] == 0.8
                    # Verify SemanticRetriever was initialized with settings
                    mock_retriever_class.assert_called_once_with(mock_settings)
                    # Verify search was called with correct parameters
                    mock_retriever.search.assert_called_once_with("test query", top_k=10, similarity_threshold=0.5)

    def test_search_with_source_filter(self, tmp_path):
        """Test search with source filter."""
        crawled_path = tmp_path / "crawled_docs"
        crawled_path.mkdir()

        mock_results = [
            {"file_path": str(tmp_path / "local_file.py"), "content": "local", "score": 0.8},
            {"file_path": str(crawled_path / "crawled.md"), "content": "crawled", "score": 0.7}
        ]

        with patch('src.rag.index_manager.index_exists', return_value=True):
            with patch('src.rag.index_manager.SemanticRetriever') as mock_retriever_class:
                mock_retriever = MagicMock()
                mock_retriever.search.return_value = mock_results
                mock_retriever_class.return_value = mock_retriever

                with patch('src.rag.index_manager.settings') as mock_settings:
                    mock_settings.CRAWLED_DOCS_PATH = str(crawled_path)

                    # Filter for crawled only
                    results = search_index("test", source_filter="crawled")
                    assert len(results) == 1
                    assert "crawled.md" in results[0]["file_path"]

                    # Filter for local only
                    results = search_index("test", source_filter="local")
                    assert len(results) == 1
                    assert "local_file.py" in results[0]["file_path"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
