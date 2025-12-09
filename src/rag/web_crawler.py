"""Web crawling and documentation indexing using CrawlAI.

This module provides functionality to crawl web documentation and automatically
index it into the RAG system for semantic search.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """Result from crawling a URL."""
    url: str
    title: str
    content: str
    markdown: str
    success: bool
    error: Optional[str] = None


class WebDocumentationCrawler:
    """Crawl and index web documentation using CrawlAI."""

    def __init__(
        self,
        user_agent: Optional[str] = None,
        headless: bool = True,
        verbose: bool = False
    ):
        """Initialize the web crawler.

        Args:
            user_agent: Custom user agent string
            headless: Run browser in headless mode
            verbose: Enable verbose logging
        """
        self.user_agent = user_agent or "Mozilla/5.0 (compatible; AI-Agent/1.0)"
        self.headless = headless
        self.verbose = verbose
        self.crawler = None

    async def crawl_url(self, url: str) -> CrawlResult:
        """Crawl a single URL and extract content.

        Args:
            url: URL to crawl

        Returns:
            CrawlResult with extracted content
        """
        try:
            if not self.crawler:
                self.crawler = AsyncWebCrawler(verbose=self.verbose)
                await self.crawler.start()

            # Configure crawl
            config = CrawlerRunConfig(
                markdown_generator=DefaultMarkdownGenerator(),
                cache_mode=CacheMode.BYPASS,  # Don't cache for now
            )

            # Crawl the URL
            logger.info(f"Crawling {url}...")
            result = await self.crawler.arun(url, config=config)

            if result.success:
                # Extract markdown (API changed in 0.7.7+)
                markdown_text = ""
                if hasattr(result, 'markdown'):
                    markdown_result = result.markdown
                    if hasattr(markdown_result, 'raw_markdown'):
                        markdown_text = markdown_result.raw_markdown or ""
                    else:
                        markdown_text = str(markdown_result)

                return CrawlResult(
                    url=url,
                    title=result.metadata.get("title", "Untitled"),
                    content=result.cleaned_html or "",
                    markdown=markdown_text,
                    success=True
                )
            else:
                return CrawlResult(
                    url=url,
                    title="",
                    content="",
                    markdown="",
                    success=False,
                    error=result.error_message
                )

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return CrawlResult(
                url=url,
                title="",
                content="",
                markdown="",
                success=False,
                error=str(e)
            )

    async def crawl_documentation(
        self,
        url: str,
        save_markdown: bool = True,
        index_immediately: bool = True
    ) -> Dict[str, Any]:
        """Crawl documentation and optionally index it.

        Args:
            url: Documentation URL to crawl
            save_markdown: Save markdown to file
            index_immediately: Index into RAG immediately

        Returns:
            Dict with crawl statistics and results
        """
        result = await self.crawl_url(url)

        if not result.success:
            return {
                "success": False,
                "error": result.error,
                "url": url
            }

        stats = {
            "success": True,
            "url": url,
            "title": result.title,
            "content_length": len(result.content),
            "markdown_length": len(result.markdown)
        }

        # Save markdown file if requested
        if save_markdown and result.markdown:
            filename = self._url_to_filename(url)
            save_path = Path(settings.FAISS_INDEX_PATH).parent / "crawled_docs" / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)

            save_path.write_text(result.markdown, encoding="utf-8")
            stats["saved_to"] = str(save_path)
            logger.info(f"Saved markdown to {save_path}")

        # Index into RAG if requested
        if index_immediately and result.markdown:
            logger.info(f"Indexing content from {url}...")
            # TODO: Integrate with existing indexer
            # For now, just save the file and it can be indexed later
            stats["indexed"] = False  # Will implement in next step

        return stats

    def _url_to_filename(self, url: str) -> str:
        """Convert URL to safe filename.

        Args:
            url: URL to convert

        Returns:
            Safe filename
        """
        # Simple conversion: remove protocol and replace / with _
        filename = url.replace("https://", "").replace("http://", "")
        filename = filename.replace("/", "_").replace(":", "_")
        # Limit length and add extension
        filename = filename[:200] + ".md"
        return filename

    async def cleanup(self):
        """Clean up crawler resources."""
        if self.crawler:
            await self.crawler.close()
            self.crawler = None


async def crawl_and_save(url: str, verbose: bool = False) -> Dict[str, Any]:
    """Convenience function to crawl a URL and save results.

    Args:
        url: URL to crawl
        verbose: Enable verbose logging

    Returns:
        Crawl statistics
    """
    crawler = WebDocumentationCrawler(verbose=verbose)
    try:
        return await crawler.crawl_documentation(url)
    finally:
        await crawler.cleanup()
