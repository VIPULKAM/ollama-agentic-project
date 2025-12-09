"""Integration tests for web crawling with URL tracking.

These tests verify the complete workflow from crawling to indexing
with URL tracking and deduplication.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock

from src.agent.tools.crawl_and_index import get_crawl_and_index_tool
from src.rag.crawl_tracker import CrawlTracker
from src.rag.web_crawler import CrawlResult
from src.config.settings import settings


@pytest.fixture
def temp_crawl_dirs(tmp_path):
    """Provide temporary directories for crawl tests."""
    crawled_docs = tmp_path / "crawled_docs"
    faiss_index = tmp_path / "faiss_index"
    crawled_docs.mkdir()
    faiss_index.mkdir()

    yield {
        "crawled_docs": crawled_docs,
        "faiss_index": faiss_index,
        "tracker_file": crawled_docs / "crawled_urls.json"
    }


@pytest.fixture
def isolated_tracker(temp_crawl_dirs):
    """Provide an isolated CrawlTracker for testing."""
    return CrawlTracker(tracker_file=temp_crawl_dirs["tracker_file"])


class TestCrawlAndIndexWithTracking:
    """Tests for CrawlAndIndexTool with URL tracking."""

    @pytest.mark.asyncio
    async def test_first_crawl_creates_record(self, temp_crawl_dirs, isolated_tracker):
        """Test that first crawl creates a tracking record."""
        test_url = "https://docs.python.org/3/library/json.html"

        # Mock the crawler
        mock_crawl_result = CrawlResult(
            url=test_url,
            title="json — JSON encoder and decoder",
            content="<html>test</html>",
            markdown="# JSON Module\n\nTest content for JSON module.",
            success=True
        )

        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.return_value = mock_crawl_result

        with patch('src.agent.tools.crawl_and_index.WebDocumentationCrawler') as mock_crawler_class:
            mock_crawler_class.return_value = mock_crawler

            # Mock chunk_file
            from src.rag.chunker import CodeChunk
            mock_chunks = [
                CodeChunk(
                    content="# JSON Module",
                    file_path=Path("test.md"),
                    start_line=1,
                    end_line=5,
                    chunk_type="text",
                    language="markdown"
                )
            ]

            with patch('src.agent.tools.crawl_and_index.chunk_file', return_value=mock_chunks):
                with patch('src.agent.tools.crawl_and_index.index_exists', return_value=False):
                    with patch('src.agent.tools.crawl_and_index.get_embeddings', return_value=[[0.1] * 384]):
                        with patch('src.agent.tools.crawl_and_index.save_index'):
                            with patch('src.agent.tools.crawl_and_index.get_crawl_tracker', return_value=isolated_tracker):
                                # Create tool with temp settings
                                test_settings = settings
                                test_settings.CRAWLED_DOCS_PATH = str(temp_crawl_dirs["crawled_docs"])

                                tool = get_crawl_and_index_tool(test_settings)

                                # Run crawl
                                result = await tool._async_crawl_and_index(test_url)

                                # Verify result
                                assert "Successfully crawled and indexed" in result
                                assert test_url in result

                                # Verify tracking record was created
                                assert isolated_tracker.is_crawled(test_url)
                                record = isolated_tracker.get_record(test_url)
                                assert record is not None
                                assert record.chunk_count == 1
                                assert record.title == "json — JSON encoder and decoder"

    @pytest.mark.asyncio
    async def test_duplicate_crawl_skipped(self, temp_crawl_dirs, isolated_tracker):
        """Test that duplicate crawl with same content is skipped."""
        test_url = "https://docs.python.org/3/library/json.html"
        test_content = "# JSON Module\n\nTest content."

        # Add existing record
        isolated_tracker.add_record(
            url=test_url,
            content=test_content,
            chunk_count=5,
            file_path="/test.md",
            title="JSON Module"
        )

        # Mock the crawler to return same content
        mock_crawl_result = CrawlResult(
            url=test_url,
            title="JSON Module",
            content="<html>test</html>",
            markdown=test_content,  # Same content
            success=True
        )

        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.return_value = mock_crawl_result

        with patch('src.agent.tools.crawl_and_index.WebDocumentationCrawler') as mock_crawler_class:
            mock_crawler_class.return_value = mock_crawler
            with patch('src.agent.tools.crawl_and_index.get_crawl_tracker', return_value=isolated_tracker):
                test_settings = settings
                test_settings.CRAWLED_DOCS_PATH = str(temp_crawl_dirs["crawled_docs"])

                tool = get_crawl_and_index_tool(test_settings)

                # Run crawl
                result = await tool._async_crawl_and_index(test_url)

                # Verify it was skipped
                assert "already indexed with identical content" in result
                assert "No changes detected" in result

    @pytest.mark.asyncio
    async def test_changed_content_reindexed(self, temp_crawl_dirs, isolated_tracker):
        """Test that changed content triggers re-indexing."""
        test_url = "https://docs.python.org/3/library/json.html"

        # Add existing record with old content
        isolated_tracker.add_record(
            url=test_url,
            content="Old content",
            chunk_count=3,
            file_path="/old.md",
            title="Old Title"
        )

        # Mock crawler to return updated content
        mock_crawl_result = CrawlResult(
            url=test_url,
            title="Updated JSON Module",
            content="<html>new</html>",
            markdown="# Updated JSON Module\n\nNew content here.",
            success=True
        )

        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.return_value = mock_crawl_result

        with patch('src.agent.tools.crawl_and_index.WebDocumentationCrawler') as mock_crawler_class:
            mock_crawler_class.return_value = mock_crawler

            from src.rag.chunker import CodeChunk
            mock_chunks = [
                CodeChunk("chunk1", Path("test.md"), 1, 5, "text", "markdown"),
                CodeChunk("chunk2", Path("test.md"), 6, 10, "text", "markdown"),
            ]

            with patch('src.agent.tools.crawl_and_index.chunk_file', return_value=mock_chunks):
                with patch('src.agent.tools.crawl_and_index.index_exists', return_value=False):
                    with patch('src.agent.tools.crawl_and_index.get_embeddings', return_value=[[0.1] * 384] * 2):
                        with patch('src.agent.tools.crawl_and_index.save_index'):
                            with patch('src.agent.tools.crawl_and_index.get_crawl_tracker', return_value=isolated_tracker):
                                test_settings = settings
                                test_settings.CRAWLED_DOCS_PATH = str(temp_crawl_dirs["crawled_docs"])

                                tool = get_crawl_and_index_tool(test_settings)

                                # Run crawl
                                result = await tool._async_crawl_and_index(test_url)

                                # Verify it was re-indexed
                                assert "Successfully updated" in result
                                assert "Previous crawl:" in result

                                # Verify record was updated
                                record = isolated_tracker.get_record(test_url)
                                assert record.chunk_count == 2  # Updated
                                assert record.title == "Updated JSON Module"


class TestMultipleCrawls:
    """Tests for crawling multiple URLs."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("url,title", [
        ("https://docs.python.org/3/library/json.html", "JSON Module"),
        ("https://docs.python.org/3/library/asyncio.html", "Asyncio"),
        ("https://docs.python.org/3/library/pathlib.html", "Pathlib"),
    ])
    async def test_crawl_multiple_urls_parallel(self, temp_crawl_dirs, url, title):
        """Test crawling multiple URLs (can run in parallel)."""
        isolated_tracker = CrawlTracker(tracker_file=temp_crawl_dirs["tracker_file"])

        mock_crawl_result = CrawlResult(
            url=url,
            title=title,
            content="<html>test</html>",
            markdown=f"# {title}\n\nContent for {title}",
            success=True
        )

        mock_crawler = AsyncMock()
        mock_crawler.crawl_url.return_value = mock_crawl_result

        with patch('src.agent.tools.crawl_and_index.WebDocumentationCrawler') as mock_crawler_class:
            mock_crawler_class.return_value = mock_crawler

            from src.rag.chunker import CodeChunk
            mock_chunks = [CodeChunk(f"chunk for {title}", Path("test.md"), 1, 5, "text", "markdown")]

            with patch('src.agent.tools.crawl_and_index.chunk_file', return_value=mock_chunks):
                with patch('src.agent.tools.crawl_and_index.index_exists', return_value=False):
                    with patch('src.agent.tools.crawl_and_index.get_embeddings', return_value=[[0.1] * 384]):
                        with patch('src.agent.tools.crawl_and_index.save_index'):
                            with patch('src.agent.tools.crawl_and_index.get_crawl_tracker', return_value=isolated_tracker):
                                test_settings = settings
                                test_settings.CRAWLED_DOCS_PATH = str(temp_crawl_dirs["crawled_docs"])

                                tool = get_crawl_and_index_tool(test_settings)
                                result = await tool._async_crawl_and_index(url)

                                assert "Successfully crawled and indexed" in result
                                assert isolated_tracker.is_crawled(url)


class TestTrackerStatistics:
    """Tests for crawler statistics."""

    def test_statistics_after_multiple_crawls(self, isolated_tracker):
        """Test statistics calculation with multiple crawls."""
        # Simulate multiple crawls
        urls = [
            ("https://url1.com", "content1" * 10, 5),
            ("https://url2.com", "content2" * 20, 10),
            ("https://url3.com", "content3" * 30, 15),
        ]

        for url, content, chunks in urls:
            isolated_tracker.add_record(url, content, chunks, f"/{url}.md")

        stats = isolated_tracker.get_stats()

        assert stats["total_urls"] == 3
        assert stats["total_chunks"] == 30  # 5 + 10 + 15
        assert stats["avg_chunks_per_url"] == 10


class TestErrorHandling:
    """Tests for error handling in crawl tracking."""

    @pytest.mark.asyncio
    async def test_crawl_failure_no_record_created(self, temp_crawl_dirs, isolated_tracker):
        """Test that failed crawl doesn't create tracking record."""
        test_url = "https://invalid-url-that-fails.com"

        mock_crawl_result = CrawlResult(
            url=test_url,
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
            with patch('src.agent.tools.crawl_and_index.get_crawl_tracker', return_value=isolated_tracker):
                test_settings = settings
                test_settings.CRAWLED_DOCS_PATH = str(temp_crawl_dirs["crawled_docs"])

                tool = get_crawl_and_index_tool(test_settings)
                result = await tool._async_crawl_and_index(test_url)

                # Verify failure message
                assert "Failed to crawl" in result

                # Verify no record was created
                assert not isolated_tracker.is_crawled(test_url)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "auto"])
