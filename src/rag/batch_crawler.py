"""Batch documentation crawling utilities.

This module provides functionality for crawling multiple URLs efficiently:
- Batch crawl from URL list files
- Sitemap parsing and crawling
- Parallel/concurrent crawling
- Rate limiting and error handling
- Progress tracking
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET

from .crawl_tracker import get_crawl_tracker
from ..agent.tools.crawl_and_index import get_crawl_and_index_tool
from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class BatchCrawlResult:
    """Result from batch crawling operation."""
    total_urls: int
    successful: int
    failed: int
    skipped: int
    duration_seconds: float
    urls_crawled: List[str]
    urls_failed: List[Dict[str, str]]  # {url, error}
    urls_skipped: List[str]


class BatchCrawler:
    """Handle batch crawling of multiple documentation URLs."""

    def __init__(
        self,
        max_concurrent: int = 5,
        rate_limit_delay: float = 1.0,
        skip_duplicates: bool = True
    ):
        """Initialize batch crawler.

        Args:
            max_concurrent: Maximum concurrent crawl operations (default: 5)
            rate_limit_delay: Delay between crawls in seconds (default: 1.0)
            skip_duplicates: Skip already crawled URLs (default: True)
        """
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self.skip_duplicates = skip_duplicates
        self.tracker = get_crawl_tracker()

    async def crawl_batch(
        self,
        urls: List[str],
        show_progress: bool = True
    ) -> BatchCrawlResult:
        """Crawl multiple URLs in parallel.

        Args:
            urls: List of URLs to crawl
            show_progress: Show progress indicators (default: True)

        Returns:
            BatchCrawlResult with statistics
        """
        start_time = datetime.now()

        # Filter out duplicates if requested
        urls_to_crawl = []
        skipped_urls = []

        if self.skip_duplicates:
            for url in urls:
                if self.tracker.is_crawled(url):
                    skipped_urls.append(url)
                    logger.info(f"Skipping already crawled URL: {url}")
                else:
                    urls_to_crawl.append(url)
        else:
            urls_to_crawl = urls

        logger.info(f"Batch crawling {len(urls_to_crawl)} URLs ({len(skipped_urls)} skipped)")

        # Crawl URLs with concurrency limit
        successful_urls = []
        failed_urls = []

        # Use semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def crawl_with_limit(url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)

                    # Crawl using the tool (handles indexing automatically)
                    tool = get_crawl_and_index_tool(settings)
                    result = await tool._async_crawl_and_index(url)

                    if "Successfully" in result or "Updated" in result:
                        return {"url": url, "success": True}
                    else:
                        return {"url": url, "success": False, "error": result}

                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
                    return {"url": url, "success": False, "error": str(e)}

        # Execute all crawls concurrently (but limited by semaphore)
        if show_progress:
            try:
                from tqdm.asyncio import tqdm
                tasks = [crawl_with_limit(url) for url in urls_to_crawl]
                results = await tqdm.gather(*tasks, desc="Crawling URLs")
            except ImportError:
                # Fallback if tqdm not available
                tasks = [crawl_with_limit(url) for url in urls_to_crawl]
                results = await asyncio.gather(*tasks)
        else:
            tasks = [crawl_with_limit(url) for url in urls_to_crawl]
            results = await asyncio.gather(*tasks)

        # Process results
        for result in results:
            if result["success"]:
                successful_urls.append(result["url"])
            else:
                failed_urls.append({"url": result["url"], "error": result.get("error", "Unknown error")})

        duration = (datetime.now() - start_time).total_seconds()

        return BatchCrawlResult(
            total_urls=len(urls),
            successful=len(successful_urls),
            failed=len(failed_urls),
            skipped=len(skipped_urls),
            duration_seconds=duration,
            urls_crawled=successful_urls,
            urls_failed=failed_urls,
            urls_skipped=skipped_urls
        )

    async def crawl_from_file(
        self,
        file_path: Path,
        show_progress: bool = True
    ) -> BatchCrawlResult:
        """Crawl URLs from a text file (one URL per line).

        Args:
            file_path: Path to file containing URLs
            show_progress: Show progress indicators

        Returns:
            BatchCrawlResult with statistics
        """
        logger.info(f"Loading URLs from {file_path}")

        if not file_path.exists():
            raise FileNotFoundError(f"URL file not found: {file_path}")

        # Read URLs from file
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        logger.info(f"Found {len(urls)} URLs in file")

        return await self.crawl_batch(urls, show_progress=show_progress)

    async def crawl_from_sitemap(
        self,
        sitemap_url: str,
        url_filter: Optional[str] = None,
        show_progress: bool = True
    ) -> BatchCrawlResult:
        """Crawl URLs from a sitemap.xml.

        Args:
            sitemap_url: URL to sitemap.xml
            url_filter: Only crawl URLs containing this string (optional)
            show_progress: Show progress indicators

        Returns:
            BatchCrawlResult with statistics
        """
        logger.info(f"Fetching sitemap from {sitemap_url}")

        # Fetch sitemap
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(sitemap_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch sitemap: HTTP {response.status}")

                sitemap_xml = await response.text()

        # Parse sitemap
        urls = self._parse_sitemap(sitemap_xml, url_filter)

        logger.info(f"Found {len(urls)} URLs in sitemap")

        return await self.crawl_batch(urls, show_progress=show_progress)

    def _parse_sitemap(self, sitemap_xml: str, url_filter: Optional[str] = None) -> List[str]:
        """Parse sitemap XML and extract URLs.

        Args:
            sitemap_xml: Sitemap XML content
            url_filter: Only include URLs containing this string

        Returns:
            List of URLs
        """
        try:
            root = ET.fromstring(sitemap_xml)

            # Handle XML namespaces
            namespaces = {
                'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'
            }

            # Extract all <loc> tags (URLs)
            urls = []
            for url_elem in root.findall('.//ns:loc', namespaces):
                url = url_elem.text
                if url:
                    # Apply filter if specified
                    if url_filter is None or url_filter in url:
                        urls.append(url)

            # Also try without namespace (some sitemaps don't use it)
            if not urls:
                for url_elem in root.findall('.//loc'):
                    url = url_elem.text
                    if url:
                        if url_filter is None or url_filter in url:
                            urls.append(url)

            return urls

        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap XML: {e}")
            raise

    async def cleanup(self):
        """Clean up crawler resources."""
        # No cleanup needed - using tools directly
        pass


async def batch_crawl_urls(
    urls: List[str],
    max_concurrent: int = 5,
    rate_limit_delay: float = 1.0,
    show_progress: bool = True
) -> BatchCrawlResult:
    """Convenience function for batch crawling.

    Args:
        urls: List of URLs to crawl
        max_concurrent: Max concurrent operations
        rate_limit_delay: Delay between crawls
        show_progress: Show progress

    Returns:
        BatchCrawlResult
    """
    crawler = BatchCrawler(
        max_concurrent=max_concurrent,
        rate_limit_delay=rate_limit_delay
    )

    try:
        return await crawler.crawl_batch(urls, show_progress=show_progress)
    finally:
        await crawler.cleanup()


async def batch_crawl_from_file(
    file_path: Path,
    max_concurrent: int = 5,
    show_progress: bool = True
) -> BatchCrawlResult:
    """Convenience function for file-based batch crawling.

    Args:
        file_path: Path to URL list file
        max_concurrent: Max concurrent operations
        show_progress: Show progress

    Returns:
        BatchCrawlResult
    """
    crawler = BatchCrawler(max_concurrent=max_concurrent)

    try:
        return await crawler.crawl_from_file(file_path, show_progress=show_progress)
    finally:
        await crawler.cleanup()


async def batch_crawl_from_sitemap(
    sitemap_url: str,
    url_filter: Optional[str] = None,
    max_concurrent: int = 5,
    show_progress: bool = True
) -> BatchCrawlResult:
    """Convenience function for sitemap-based crawling.

    Args:
        sitemap_url: URL to sitemap.xml
        url_filter: Only crawl URLs containing this string
        max_concurrent: Max concurrent operations
        show_progress: Show progress

    Returns:
        BatchCrawlResult
    """
    crawler = BatchCrawler(max_concurrent=max_concurrent)

    try:
        return await crawler.crawl_from_sitemap(
            sitemap_url,
            url_filter=url_filter,
            show_progress=show_progress
        )
    finally:
        await crawler.cleanup()
