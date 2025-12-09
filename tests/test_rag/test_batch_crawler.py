"""Tests for batch documentation crawler."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from src.rag.batch_crawler import (
    BatchCrawler,
    BatchCrawlResult,
    batch_crawl_urls,
    batch_crawl_from_file,
    batch_crawl_from_sitemap
)


class TestBatchCrawler:
    """Tests for BatchCrawler class."""

    def test_initialization(self):
        """Test BatchCrawler initialization with default parameters."""
        crawler = BatchCrawler()

        assert crawler.max_concurrent == 5
        assert crawler.rate_limit_delay == 1.0
        assert crawler.skip_duplicates is True

    def test_initialization_custom_params(self):
        """Test BatchCrawler initialization with custom parameters."""
        crawler = BatchCrawler(
            max_concurrent=10,
            rate_limit_delay=0.5,
            skip_duplicates=False
        )

        assert crawler.max_concurrent == 10
        assert crawler.rate_limit_delay == 0.5
        assert crawler.skip_duplicates is False

    @pytest.mark.asyncio
    async def test_crawl_batch_success(self):
        """Test successful batch crawl of multiple URLs."""
        crawler = BatchCrawler(max_concurrent=2, rate_limit_delay=0.1)

        urls = [
            "https://example.com/doc1",
            "https://example.com/doc2",
            "https://example.com/doc3"
        ]

        # Mock the crawl_and_index tool
        with patch('src.rag.batch_crawler.get_crawl_and_index_tool') as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool._async_crawl_and_index = AsyncMock(
                return_value="Successfully crawled and indexed"
            )
            mock_get_tool.return_value = mock_tool

            result = await crawler.crawl_batch(urls, show_progress=False)

            assert result.total_urls == 3
            assert result.successful == 3
            assert result.failed == 0
            assert result.skipped == 0
            assert len(result.urls_crawled) == 3
            assert len(result.urls_failed) == 0
            assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_crawl_batch_with_failures(self):
        """Test batch crawl with some failures."""
        crawler = BatchCrawler(max_concurrent=2, rate_limit_delay=0.1)

        urls = [
            "https://example.com/doc1",
            "https://example.com/doc2",
            "https://example.com/doc3"
        ]

        # Mock the crawl_and_index tool with mixed results
        async def mock_crawl(url):
            if "doc2" in url:
                raise Exception("Network error")
            return "Successfully crawled and indexed"

        with patch('src.rag.batch_crawler.get_crawl_and_index_tool') as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool._async_crawl_and_index = mock_crawl
            mock_get_tool.return_value = mock_tool

            result = await crawler.crawl_batch(urls, show_progress=False)

            assert result.total_urls == 3
            assert result.successful == 2
            assert result.failed == 1
            assert result.skipped == 0
            assert len(result.urls_crawled) == 2
            assert len(result.urls_failed) == 1
            assert "doc2" in result.urls_failed[0]["url"]

    @pytest.mark.asyncio
    async def test_crawl_batch_skip_duplicates(self):
        """Test batch crawl skips already crawled URLs."""
        crawler = BatchCrawler(skip_duplicates=True, rate_limit_delay=0.1)

        urls = [
            "https://example.com/doc1",
            "https://example.com/doc2",
            "https://example.com/doc3"
        ]

        # Mock tracker to say doc2 is already crawled
        with patch.object(crawler.tracker, 'is_crawled') as mock_is_crawled:
            mock_is_crawled.side_effect = lambda url: "doc2" in url

            with patch('src.rag.batch_crawler.get_crawl_and_index_tool') as mock_get_tool:
                mock_tool = MagicMock()
                mock_tool._async_crawl_and_index = AsyncMock(
                    return_value="Successfully crawled and indexed"
                )
                mock_get_tool.return_value = mock_tool

                result = await crawler.crawl_batch(urls, show_progress=False)

                assert result.total_urls == 3
                assert result.successful == 2
                assert result.failed == 0
                assert result.skipped == 1
                assert len(result.urls_crawled) == 2
                assert len(result.urls_skipped) == 1
                assert urls[1] in result.urls_skipped

    @pytest.mark.asyncio
    async def test_crawl_from_file(self, tmp_path):
        """Test crawling URLs from a file."""
        # Create test file with URLs
        url_file = tmp_path / "urls.txt"
        url_file.write_text("""# Test URLs
https://example.com/doc1
https://example.com/doc2

# Another comment
https://example.com/doc3
""")

        crawler = BatchCrawler(rate_limit_delay=0.1)

        with patch('src.rag.batch_crawler.get_crawl_and_index_tool') as mock_get_tool:
            mock_tool = MagicMock()
            mock_tool._async_crawl_and_index = AsyncMock(
                return_value="Successfully crawled and indexed"
            )
            mock_get_tool.return_value = mock_tool

            result = await crawler.crawl_from_file(url_file, show_progress=False)

            assert result.total_urls == 3
            assert result.successful == 3
            assert result.failed == 0

    @pytest.mark.asyncio
    async def test_crawl_from_file_not_found(self):
        """Test error when URL file doesn't exist."""
        crawler = BatchCrawler()

        with pytest.raises(FileNotFoundError):
            await crawler.crawl_from_file(Path("/nonexistent/urls.txt"))

    @pytest.mark.asyncio
    async def test_crawl_from_sitemap(self):
        """Test crawling URLs from a sitemap."""
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/doc1</loc>
  </url>
  <url>
    <loc>https://example.com/doc2</loc>
  </url>
  <url>
    <loc>https://example.com/doc3</loc>
  </url>
</urlset>
"""

        crawler = BatchCrawler(rate_limit_delay=0.1)

        # Mock aiohttp session and response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=sitemap_xml)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('src.rag.batch_crawler.get_crawl_and_index_tool') as mock_get_tool:
                mock_tool = MagicMock()
                mock_tool._async_crawl_and_index = AsyncMock(
                    return_value="Successfully crawled and indexed"
                )
                mock_get_tool.return_value = mock_tool

                result = await crawler.crawl_from_sitemap(
                    "https://example.com/sitemap.xml",
                    show_progress=False
                )

                assert result.total_urls == 3
                assert result.successful == 3
                assert result.failed == 0

    @pytest.mark.asyncio
    async def test_crawl_from_sitemap_with_filter(self):
        """Test crawling from sitemap with URL filter."""
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/api/doc1</loc>
  </url>
  <url>
    <loc>https://example.com/guide/doc2</loc>
  </url>
  <url>
    <loc>https://example.com/api/doc3</loc>
  </url>
</urlset>
"""

        crawler = BatchCrawler(rate_limit_delay=0.1)

        # Mock aiohttp session and response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=sitemap_xml)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            with patch('src.rag.batch_crawler.get_crawl_and_index_tool') as mock_get_tool:
                mock_tool = MagicMock()
                mock_tool._async_crawl_and_index = AsyncMock(
                    return_value="Successfully crawled and indexed"
                )
                mock_get_tool.return_value = mock_tool

                # Only crawl /api/ URLs
                result = await crawler.crawl_from_sitemap(
                    "https://example.com/sitemap.xml",
                    url_filter="/api/",
                    show_progress=False
                )

                # Should only crawl 2 URLs (doc1 and doc3)
                assert result.total_urls == 2
                assert result.successful == 2
                assert result.failed == 0

    @pytest.mark.asyncio
    async def test_crawl_from_sitemap_http_error(self):
        """Test error handling when sitemap fetch fails."""
        crawler = BatchCrawler()

        # Mock failed HTTP response
        mock_response = MagicMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession', return_value=mock_session):
            with pytest.raises(Exception, match="Failed to fetch sitemap"):
                await crawler.crawl_from_sitemap("https://example.com/sitemap.xml")

    def test_parse_sitemap(self):
        """Test sitemap XML parsing."""
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/doc1</loc>
  </url>
  <url>
    <loc>https://example.com/doc2</loc>
  </url>
</urlset>
"""

        crawler = BatchCrawler()
        urls = crawler._parse_sitemap(sitemap_xml)

        assert len(urls) == 2
        assert "https://example.com/doc1" in urls
        assert "https://example.com/doc2" in urls

    def test_parse_sitemap_no_namespace(self):
        """Test parsing sitemap without namespace."""
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset>
  <url>
    <loc>https://example.com/doc1</loc>
  </url>
  <url>
    <loc>https://example.com/doc2</loc>
  </url>
</urlset>
"""

        crawler = BatchCrawler()
        urls = crawler._parse_sitemap(sitemap_xml)

        # Should fallback to parsing without namespace
        assert len(urls) == 2
        assert "https://example.com/doc1" in urls

    def test_parse_sitemap_with_filter(self):
        """Test parsing sitemap with URL filter."""
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url>
    <loc>https://example.com/api/doc1</loc>
  </url>
  <url>
    <loc>https://example.com/guide/doc2</loc>
  </url>
  <url>
    <loc>https://example.com/api/doc3</loc>
  </url>
</urlset>
"""

        crawler = BatchCrawler()
        urls = crawler._parse_sitemap(sitemap_xml, url_filter="/api/")

        assert len(urls) == 2
        assert all("/api/" in url for url in urls)

    def test_parse_sitemap_invalid_xml(self):
        """Test error handling for invalid XML."""
        invalid_xml = "This is not valid XML"

        crawler = BatchCrawler()

        with pytest.raises(Exception):
            crawler._parse_sitemap(invalid_xml)

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup of crawler resources."""
        crawler = BatchCrawler()

        # Cleanup should succeed without errors (no-op)
        await crawler.cleanup()

        # No assertion needed - just ensure it doesn't raise


class TestConvenienceFunctions:
    """Tests for convenience wrapper functions."""

    @pytest.mark.asyncio
    async def test_batch_crawl_urls(self):
        """Test batch_crawl_urls convenience function."""
        urls = ["https://example.com/doc1", "https://example.com/doc2"]

        with patch('src.rag.batch_crawler.BatchCrawler') as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_result = BatchCrawlResult(
                total_urls=2,
                successful=2,
                failed=0,
                skipped=0,
                duration_seconds=1.5,
                urls_crawled=urls,
                urls_failed=[],
                urls_skipped=[]
            )
            mock_crawler.crawl_batch = AsyncMock(return_value=mock_result)
            mock_crawler.cleanup = AsyncMock()
            mock_crawler_class.return_value = mock_crawler

            result = await batch_crawl_urls(urls, max_concurrent=3, rate_limit_delay=0.5)

            # Verify BatchCrawler was initialized with correct params
            mock_crawler_class.assert_called_once_with(
                max_concurrent=3,
                rate_limit_delay=0.5
            )

            # Verify crawl_batch was called
            mock_crawler.crawl_batch.assert_called_once_with(urls, show_progress=True)

            # Verify cleanup was called
            mock_crawler.cleanup.assert_called_once()

            # Verify result
            assert result.total_urls == 2
            assert result.successful == 2

    @pytest.mark.asyncio
    async def test_batch_crawl_from_file(self, tmp_path):
        """Test batch_crawl_from_file convenience function."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://example.com/doc1\n")

        with patch('src.rag.batch_crawler.BatchCrawler') as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_result = BatchCrawlResult(
                total_urls=1,
                successful=1,
                failed=0,
                skipped=0,
                duration_seconds=1.0,
                urls_crawled=["https://example.com/doc1"],
                urls_failed=[],
                urls_skipped=[]
            )
            mock_crawler.crawl_from_file = AsyncMock(return_value=mock_result)
            mock_crawler.cleanup = AsyncMock()
            mock_crawler_class.return_value = mock_crawler

            result = await batch_crawl_from_file(url_file, max_concurrent=5)

            # Verify cleanup was called
            mock_crawler.cleanup.assert_called_once()

            # Verify result
            assert result.total_urls == 1
            assert result.successful == 1

    @pytest.mark.asyncio
    async def test_batch_crawl_from_sitemap(self):
        """Test batch_crawl_from_sitemap convenience function."""
        sitemap_url = "https://example.com/sitemap.xml"

        with patch('src.rag.batch_crawler.BatchCrawler') as mock_crawler_class:
            mock_crawler = MagicMock()
            mock_result = BatchCrawlResult(
                total_urls=3,
                successful=3,
                failed=0,
                skipped=0,
                duration_seconds=2.0,
                urls_crawled=[
                    "https://example.com/doc1",
                    "https://example.com/doc2",
                    "https://example.com/doc3"
                ],
                urls_failed=[],
                urls_skipped=[]
            )
            mock_crawler.crawl_from_sitemap = AsyncMock(return_value=mock_result)
            mock_crawler.cleanup = AsyncMock()
            mock_crawler_class.return_value = mock_crawler

            result = await batch_crawl_from_sitemap(
                sitemap_url,
                url_filter="/docs/",
                max_concurrent=5
            )

            # Verify crawl_from_sitemap was called with correct params
            mock_crawler.crawl_from_sitemap.assert_called_once_with(
                sitemap_url,
                url_filter="/docs/",
                show_progress=True
            )

            # Verify cleanup was called
            mock_crawler.cleanup.assert_called_once()

            # Verify result
            assert result.total_urls == 3
            assert result.successful == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
