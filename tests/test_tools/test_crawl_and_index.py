"""Tests for crawl_and_index tool."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from src.agent.tools.crawl_and_index import get_crawl_and_index_tool
from src.rag.web_crawler import CrawlResult
from src.rag.chunker import CodeChunk
from src.config.settings import settings


class TestCrawlAndIndexTool:
    """Tests for the CrawlAndIndexTool."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = get_crawl_and_index_tool(settings)
        self.test_url = "https://docs.python.org/3/library/argparse.html"

    def test_tool_initialization(self):
        """Test that tool initializes correctly."""
        assert self.tool.name == "crawl_and_index"
        assert "crawl" in self.tool.description.lower()
        assert self.tool.settings == settings

    def test_tool_has_correct_input_schema(self):
        """Test that tool has correct input schema."""
        schema = self.tool.args_schema.model_json_schema()
        assert "url" in schema["properties"]
        assert schema["properties"]["url"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_successful_crawl_and_index(self):
        """Test successful crawl and index operation."""
        # Mock the crawler
        mock_crawl_result = CrawlResult(
            url=self.test_url,
            title="argparse - Python Documentation",
            content="<html>content</html>",
            markdown="# argparse\n\nSome documentation content here.",
            success=True
        )

        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.return_value = mock_crawl_result

        with patch('src.agent.tools.crawl_and_index.WebDocumentationCrawler') as mock_crawler_class:
            mock_crawler_class.return_value = mock_crawler

            # Mock chunk_file to return some chunks
            mock_chunks = [
                CodeChunk(
                    content="# argparse documentation",
                    file_path=Path("crawled_docs/test.md"),
                    start_line=1,
                    end_line=10,
                    chunk_type="text",
                    language="markdown"
                )
            ]

            with patch('src.agent.tools.crawl_and_index.chunk_file', return_value=mock_chunks):
                # Mock the index operations
                with patch('src.agent.tools.crawl_and_index.index_exists', return_value=False):
                    with patch('src.agent.tools.crawl_and_index.get_embeddings', return_value=[[0.1] * 384]):
                        with patch('src.agent.tools.crawl_and_index.save_index'):
                            # Run the tool
                            result = await self.tool._async_crawl_and_index(self.test_url)

                            # Verify result
                            assert "Successfully crawled and indexed" in result
                            assert self.test_url in result
                            assert "argparse - Python Documentation" in result
                            assert "Chunks extracted: 1" in result

    @pytest.mark.asyncio
    async def test_failed_crawl(self):
        """Test handling of failed crawl."""
        # Mock failed crawl
        mock_crawl_result = CrawlResult(
            url=self.test_url,
            title="",
            content="",
            markdown="",
            success=False,
            error="Connection timeout"
        )

        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.return_value = mock_crawl_result

        with patch('src.agent.tools.crawl_and_index.WebDocumentationCrawler') as mock_crawler_class:
            mock_crawler_class.return_value = mock_crawler

            result = await self.tool._async_crawl_and_index(self.test_url)

            assert "Failed to crawl" in result
            assert "Connection timeout" in result

    @pytest.mark.asyncio
    async def test_no_content_extracted(self):
        """Test handling when no markdown content is extracted."""
        # Mock successful crawl but no markdown
        mock_crawl_result = CrawlResult(
            url=self.test_url,
            title="Test Page",
            content="<html>content</html>",
            markdown="",  # Empty markdown
            success=True
        )

        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.return_value = mock_crawl_result

        with patch('src.agent.tools.crawl_and_index.WebDocumentationCrawler') as mock_crawler_class:
            mock_crawler_class.return_value = mock_crawler

            result = await self.tool._async_crawl_and_index(self.test_url)

            assert "No content extracted" in result

    @pytest.mark.asyncio
    async def test_no_chunks_extracted(self):
        """Test handling when chunking returns no chunks."""
        # Mock successful crawl with markdown
        mock_crawl_result = CrawlResult(
            url=self.test_url,
            title="Test Page",
            content="<html>content</html>",
            markdown="# Title\nShort content.",
            success=True
        )

        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.return_value = mock_crawl_result

        with patch('src.agent.tools.crawl_and_index.WebDocumentationCrawler') as mock_crawler_class:
            mock_crawler_class.return_value = mock_crawler

            # Mock chunk_file to return empty list
            with patch('src.agent.tools.crawl_and_index.chunk_file', return_value=[]):
                result = await self.tool._async_crawl_and_index(self.test_url)

                assert "No chunks extracted" in result

    def test_url_to_filename(self):
        """Test URL to filename conversion."""
        test_cases = [
            ("https://docs.python.org/3/library/argparse.html", "docs.python.org_3_library_argparse.html.md"),
            ("http://example.com/api/docs", "example.com_api_docs.md"),
            ("https://fastapi.tiangolo.com/", "fastapi.tiangolo.com_.md"),
        ]

        for url, expected_filename in test_cases:
            filename = self.tool._url_to_filename(url)
            assert filename == expected_filename

    @pytest.mark.asyncio
    async def test_add_chunks_to_new_index(self):
        """Test adding chunks to a new (non-existent) index."""
        mock_chunks = [
            CodeChunk(
                content="Test content 1",
                file_path=Path("crawled_docs/test1.md"),
                start_line=1,
                end_line=5,
                chunk_type="text",
                language="markdown"
            ),
            CodeChunk(
                content="Test content 2",
                file_path=Path("crawled_docs/test2.md"),
                start_line=1,
                end_line=5,
                chunk_type="text",
                language="markdown"
            ),
        ]

        with patch('src.agent.tools.crawl_and_index.index_exists', return_value=False):
            with patch('src.agent.tools.crawl_and_index.get_embeddings', return_value=[[0.1] * 384, [0.2] * 384]):
                with patch('src.agent.tools.crawl_and_index.save_index') as mock_save:
                    count = self.tool._add_chunks_to_index(mock_chunks)

                    assert count == 2
                    mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_chunks_to_existing_index(self):
        """Test adding chunks to an existing index."""
        mock_chunks = [
            CodeChunk(
                content="New content",
                file_path=Path("crawled_docs/new.md"),
                start_line=1,
                end_line=5,
                chunk_type="text",
                language="markdown"
            )
        ]

        # Mock existing index with 2 chunks
        mock_existing_metadata = [
            {
                "content": "Existing content 1",
                "file_path": "crawled_docs/existing1.md",
                "start_line": 1,
                "end_line": 5,
                "chunk_type": "text",
                "language": "markdown",
                "metadata": {}
            },
            {
                "content": "Existing content 2",
                "file_path": "crawled_docs/existing2.md",
                "start_line": 1,
                "end_line": 5,
                "chunk_type": "text",
                "language": "markdown",
                "metadata": {}
            }
        ]

        import faiss
        import numpy as np
        mock_index = faiss.IndexFlatL2(384)
        mock_index.add(np.array([[0.1] * 384, [0.2] * 384], dtype=np.float32))

        with patch('src.agent.tools.crawl_and_index.index_exists', return_value=True):
            with patch('src.agent.tools.crawl_and_index.load_index', return_value=(mock_index, mock_existing_metadata)):
                with patch('src.agent.tools.crawl_and_index.get_embeddings', return_value=[[0.3] * 384]):
                    with patch('src.agent.tools.crawl_and_index.save_index') as mock_save:
                        count = self.tool._add_chunks_to_index(mock_chunks)

                        assert count == 1
                        # Verify save was called with updated index (3 total chunks)
                        mock_save.assert_called_once()
                        saved_chunks = mock_save.call_args[0][1]
                        assert len(saved_chunks) == 3  # 2 existing + 1 new

    def test_sync_run_wrapper(self):
        """Test that the sync _run method works."""
        # Mock the async implementation
        async def mock_async_crawl(url):
            return "Success"

        with patch.object(self.tool, '_async_crawl_and_index', new=mock_async_crawl):
            result = self.tool._run(self.test_url)

            assert result == "Success"

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test that exceptions are properly caught and returned."""
        # Mock crawler to raise exception
        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.side_effect = Exception("Network error")

        with patch('src.agent.tools.crawl_and_index.WebDocumentationCrawler') as mock_crawler_class:
            mock_crawler_class.return_value = mock_crawler

            result = await self.tool._async_crawl_and_index(self.test_url)

            assert "Error crawling" in result
            assert "Network error" in result

    @pytest.mark.asyncio
    async def test_cleanup_called(self):
        """Test that crawler cleanup is called even on error."""
        # Mock crawler
        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.side_effect = Exception("Test error")
        mock_crawler.cleanup = AsyncMock()

        with patch('src.agent.tools.crawl_and_index.WebDocumentationCrawler') as mock_crawler_class:
            mock_crawler_class.return_value = mock_crawler

            result = await self.tool._async_crawl_and_index(self.test_url)

            # Verify cleanup was called
            mock_crawler.cleanup.assert_called_once()


class TestCrawlAndIndexToolIntegration:
    """Integration tests for CrawlAndIndexTool (require network access)."""

    @pytest.mark.skip(reason="Integration test - requires network access")
    @pytest.mark.asyncio
    async def test_real_crawl_and_index(self):
        """Test real crawl and index operation (skipped by default)."""
        tool = get_crawl_and_index_tool(settings)

        # Use a small, stable documentation page
        url = "https://docs.python.org/3/library/json.html"

        result = await tool._async_crawl_and_index(url)

        assert "Successfully crawled and indexed" in result
        assert url in result
        assert "Chunks extracted:" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
