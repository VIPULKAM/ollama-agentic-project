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


class TestMultiProfileCrawling:
    """Tests for multi-profile and stack crawling functionality."""

    @pytest.mark.asyncio
    async def test_crawl_multiple_profiles_success(self):
        """Test successful multi-profile crawling."""
        from src.rag.batch_crawler import crawl_multiple_profiles, ProfileCrawlResult

        profile_names = ["fastapi", "django"]

        # Mock profile manager
        with patch('src.rag.batch_crawler.get_profile_manager') as mock_get_manager:
            mock_manager = MagicMock()

            # Mock profiles
            fastapi_profile = MagicMock()
            fastapi_profile.name = "fastapi"
            fastapi_profile.description = "FastAPI docs"
            fastapi_profile.sitemap_url = "https://fastapi.tiangolo.com/sitemap.xml"
            fastapi_profile.url_filter = "/tutorial/"
            fastapi_profile.urls = None
            fastapi_profile.max_concurrent = 5

            django_profile = MagicMock()
            django_profile.name = "django"
            django_profile.description = "Django docs"
            django_profile.sitemap_url = None
            django_profile.url_filter = None
            django_profile.urls = ["https://docs.djangoproject.com/en/stable/topics/db/models/"]
            django_profile.max_concurrent = 5

            mock_manager.get_profile.side_effect = lambda name: {
                "fastapi": fastapi_profile,
                "django": django_profile
            }.get(name)

            mock_get_manager.return_value = mock_manager

            # Mock batch crawl functions
            with patch('src.rag.batch_crawler.batch_crawl_from_sitemap') as mock_sitemap:
                with patch('src.rag.batch_crawler.batch_crawl_urls') as mock_urls:
                    # Mock successful results
                    mock_sitemap_result = BatchCrawlResult(
                        total_urls=10,
                        successful=10,
                        failed=0,
                        skipped=0,
                        duration_seconds=5.0,
                        urls_crawled=["url1", "url2"],
                        urls_failed=[],
                        urls_skipped=[]
                    )

                    mock_urls_result = BatchCrawlResult(
                        total_urls=5,
                        successful=5,
                        failed=0,
                        skipped=0,
                        duration_seconds=3.0,
                        urls_crawled=["url3"],
                        urls_failed=[],
                        urls_skipped=[]
                    )

                    mock_sitemap.return_value = mock_sitemap_result
                    mock_urls.return_value = mock_urls_result

                    result = await crawl_multiple_profiles(
                        profile_names,
                        max_concurrent=5,
                        show_progress=False
                    )

                    # Verify aggregated results
                    assert result.total_profiles == 2
                    assert result.successful_profiles == 2
                    assert result.failed_profiles == 0
                    assert result.total_urls == 15
                    assert result.successful_urls == 15
                    assert result.failed_urls == 0
                    assert result.skipped_urls == 0

                    # Verify individual profile results
                    assert len(result.profile_results) == 2
                    assert result.profile_results[0].profile_name == "fastapi"
                    assert result.profile_results[1].profile_name == "django"

    @pytest.mark.asyncio
    async def test_crawl_multiple_profiles_with_failure(self):
        """Test multi-profile crawling with one profile failing."""
        from src.rag.batch_crawler import crawl_multiple_profiles

        profile_names = ["fastapi", "nonexistent"]

        with patch('src.rag.batch_crawler.get_profile_manager') as mock_get_manager:
            mock_manager = MagicMock()

            fastapi_profile = MagicMock()
            fastapi_profile.name = "fastapi"
            fastapi_profile.description = "FastAPI docs"
            fastapi_profile.urls = ["https://example.com/doc1"]
            fastapi_profile.sitemap_url = None
            fastapi_profile.max_concurrent = 5

            mock_manager.get_profile.side_effect = lambda name: {
                "fastapi": fastapi_profile,
                "nonexistent": None
            }.get(name)

            mock_get_manager.return_value = mock_manager

            with patch('src.rag.batch_crawler.batch_crawl_urls') as mock_urls:
                mock_urls.return_value = BatchCrawlResult(
                    total_urls=1,
                    successful=1,
                    failed=0,
                    skipped=0,
                    duration_seconds=1.0,
                    urls_crawled=["url1"],
                    urls_failed=[],
                    urls_skipped=[]
                )

                result = await crawl_multiple_profiles(
                    profile_names,
                    max_concurrent=5,
                    show_progress=False
                )

                # One successful, one failed
                assert result.total_profiles == 2
                assert result.successful_profiles == 1
                assert result.failed_profiles == 1
                assert result.profile_results[1].error is not None

    @pytest.mark.asyncio
    async def test_crawl_stack_success(self):
        """Test successful stack crawling."""
        from src.rag.batch_crawler import crawl_stack

        # Mock crawl_multiple_profiles
        with patch('src.rag.batch_crawler.crawl_multiple_profiles') as mock_multi:
            from src.rag.batch_crawler import MultiProfileCrawlResult

            mock_result = MultiProfileCrawlResult(
                total_profiles=4,
                successful_profiles=4,
                failed_profiles=0,
                total_urls=20,
                successful_urls=20,
                failed_urls=0,
                skipped_urls=0,
                duration_seconds=10.0,
                profile_results=[]
            )

            mock_multi.return_value = mock_result

            result = await crawl_stack("backend", max_concurrent=5, show_progress=False)

            # Verify crawl_multiple_profiles was called with backend stack profiles
            mock_multi.assert_called_once()
            call_args = mock_multi.call_args
            assert "fastapi" in call_args[0][0]
            assert "django" in call_args[0][0]
            assert "flask" in call_args[0][0]
            assert "postgresql" in call_args[0][0]

            # Verify result
            assert result.total_profiles == 4
            assert result.successful_profiles == 4

    @pytest.mark.asyncio
    async def test_crawl_stack_invalid_name(self):
        """Test error when stack name is invalid."""
        from src.rag.batch_crawler import crawl_stack

        with pytest.raises(ValueError, match="Stack 'invalid' not found"):
            await crawl_stack("invalid", max_concurrent=5, show_progress=False)

    def test_get_stack_profiles(self):
        """Test getting stack profile names."""
        from src.rag.batch_crawler import get_stack_profiles

        # Test valid stacks
        backend = get_stack_profiles("backend")
        assert backend is not None
        assert "fastapi" in backend
        assert "django" in backend

        frontend = get_stack_profiles("frontend")
        assert frontend is not None
        assert "react" in frontend
        assert "vue" in frontend

        # Test invalid stack
        invalid = get_stack_profiles("nonexistent")
        assert invalid is None

    def test_list_stacks(self):
        """Test listing all available stacks."""
        from src.rag.batch_crawler import list_stacks

        stacks = list_stacks()

        assert isinstance(stacks, dict)
        assert "backend" in stacks
        assert "frontend" in stacks
        assert "fullstack" in stacks
        assert "python" in stacks
        assert "javascript" in stacks
        assert "database" in stacks
        assert "ai" in stacks

        # Verify it's a copy (modifications don't affect original)
        stacks["test"] = []
        stacks2 = list_stacks()
        assert "test" not in stacks2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
