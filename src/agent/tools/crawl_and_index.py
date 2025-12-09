"""
CrawlAndIndex Tool for the AI Coding Agent.

This tool allows the agent to autonomously crawl web documentation
and add it to the RAG index for semantic search.
"""

import asyncio
import logging
from pathlib import Path
from typing import List

import faiss
import numpy as np
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from src.rag.web_crawler import WebDocumentationCrawler
from src.rag.chunker import chunk_file, CodeChunk
from src.rag.embeddings import get_embeddings
from src.rag.indexer import load_index, save_index, index_exists
from src.rag.crawl_tracker import get_crawl_tracker
from src.config.settings import Settings

logger = logging.getLogger("ai_agent.crawl_and_index")


class CrawlAndIndexInput(BaseModel):
    """Input schema for the CrawlAndIndex tool."""
    url: str = Field(description="The URL of the documentation page to crawl and index.")


class CrawlAndIndexTool(BaseTool):
    """
    Tool to crawl web documentation and add it to the RAG index.

    This tool autonomously:
    1. Crawls the specified URL using CrawlAI
    2. Extracts markdown content
    3. Chunks the content for RAG
    4. Adds chunks to the existing FAISS index
    5. Returns a summary of what was indexed
    """

    name: str = "crawl_and_index"

    description: str = (
        "Crawls a documentation URL, extracts content, and adds it to the RAG index "
        "for semantic search. Use this when you need to learn about a library, framework, "
        "or API that isn't already in the codebase. Provide the URL to the documentation page."
    )

    args_schema: type[BaseModel] = CrawlAndIndexInput

    settings: Settings

    def _run(self, url: str) -> str:
        """
        Execute the crawl and index operation.

        Args:
            url: The documentation URL to crawl.

        Returns:
            A summary of the indexing operation.
        """
        try:
            # Create a new event loop for this thread (handles ThreadPoolExecutor case)
            # This is necessary because LangChain runs tools in thread pools
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._async_crawl_and_index(url))
                return result
            finally:
                loop.close()

        except Exception as e:
            error_msg = f"Error crawling and indexing {url}: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def _async_crawl_and_index(self, url: str) -> str:
        """
        Async implementation of crawl and index.

        Args:
            url: The documentation URL to crawl.

        Returns:
            A summary of the indexing operation.
        """
        logger.info(f"Starting crawl and index for {url}")

        # Step 0: Check if URL was already crawled
        tracker = get_crawl_tracker()
        existing_record = tracker.get_record(url)

        # Step 1: Crawl the URL
        crawler = WebDocumentationCrawler(
            headless=self.settings.CRAWLER_HEADLESS,
            verbose=self.settings.CRAWLER_VERBOSE
        )

        try:
            try:
                crawl_result = await crawler.crawl_url(url)
            except Exception as e:
                error_msg = f"Error crawling {url}: {str(e)}"
                logger.error(error_msg)
                return error_msg

            if not crawl_result.success:
                return f"Failed to crawl {url}: {crawl_result.error}"

            if not crawl_result.markdown:
                return f"No content extracted from {url}"

            logger.info(f"Crawled {url}: {len(crawl_result.markdown)} bytes of markdown")

            # Check if content has changed
            if existing_record and not tracker.has_changed(url, crawl_result.markdown):
                return (
                    f"URL already indexed with identical content\n"
                    f"- URL: {url}\n"
                    f"- Last crawled: {existing_record.crawl_date}\n"
                    f"- Chunks in index: {existing_record.chunk_count}\n"
                    f"- No changes detected, skipping re-index"
                )

            # Step 2: Save markdown to temp file for chunking
            crawled_docs_path = Path(self.settings.CRAWLED_DOCS_PATH).expanduser()
            crawled_docs_path.mkdir(parents=True, exist_ok=True)

            # Create filename from URL
            filename = self._url_to_filename(url)
            markdown_file = crawled_docs_path / filename
            markdown_file.write_text(crawl_result.markdown, encoding='utf-8')

            logger.info(f"Saved markdown to {markdown_file}")

            # Step 3: Chunk the markdown content
            chunks = chunk_file(markdown_file, content=crawl_result.markdown)

            if not chunks:
                return f"No chunks extracted from {url}"

            logger.info(f"Extracted {len(chunks)} chunks from {url}")

            # Step 4: Add chunks to FAISS index
            added_count = self._add_chunks_to_index(chunks)

            # Step 5: Track the crawled URL
            tracker.add_record(
                url=url,
                content=crawl_result.markdown,
                chunk_count=len(chunks),
                file_path=str(markdown_file),
                title=crawl_result.title
            )

            # Build summary message
            status = "updated" if existing_record else "crawled and indexed"
            summary = (
                f"Successfully {status} {url}\n"
                f"- Title: {crawl_result.title}\n"
                f"- Content size: {len(crawl_result.markdown)} bytes\n"
                f"- Chunks extracted: {len(chunks)}\n"
                f"- Chunks added to index: {added_count}\n"
                f"- Saved to: {markdown_file}"
            )

            if existing_record:
                summary += f"\n- Previous crawl: {existing_record.crawl_date}"

            logger.info(summary)
            return summary

        finally:
            await crawler.cleanup()

    def _add_chunks_to_index(self, new_chunks: List[CodeChunk]) -> int:
        """
        Add new chunks to the existing FAISS index.

        Args:
            new_chunks: List of chunks to add to the index.

        Returns:
            Number of chunks added.
        """
        index_path = Path(self.settings.FAISS_INDEX_PATH).expanduser()

        # Load existing index or create new one
        if index_exists(index_path):
            logger.info("Loading existing index...")
            index, existing_metadata = load_index(index_path)

            # Convert metadata dicts back to CodeChunk objects (for consistency)
            # Note: existing_metadata is a list of dicts from load_index
            all_chunks = existing_metadata + [
                {
                    "content": chunk.content,
                    "file_path": str(chunk.file_path),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                    "metadata": chunk.metadata or {}
                }
                for chunk in new_chunks
            ]
        else:
            logger.info("No existing index found, creating new one...")
            # Create new index
            from src.rag.embeddings import get_embedding_dimension
            embedding_dim = get_embedding_dimension()
            index = faiss.IndexFlatL2(embedding_dim)

            all_chunks = [
                {
                    "content": chunk.content,
                    "file_path": str(chunk.file_path),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                    "metadata": chunk.metadata or {}
                }
                for chunk in new_chunks
            ]

        # Generate embeddings for new chunks
        logger.info(f"Generating embeddings for {len(new_chunks)} new chunks...")
        chunk_texts = [chunk.content for chunk in new_chunks]
        new_embeddings = get_embeddings(
            chunk_texts,
            batch_size=self.settings.INDEX_BATCH_SIZE,
            show_progress=False,
            normalize=True
        )

        # Add new embeddings to index
        embeddings_array = np.array(new_embeddings, dtype=np.float32)
        index.add(embeddings_array)

        logger.info(f"Added {len(new_chunks)} vectors to index (total: {index.ntotal})")

        # Save updated index
        # Convert all_chunks back to CodeChunk objects for save_index
        chunk_objects = []
        for chunk_dict in all_chunks:
            chunk_obj = CodeChunk(
                content=chunk_dict["content"],
                file_path=Path(chunk_dict["file_path"]),
                start_line=chunk_dict["start_line"],
                end_line=chunk_dict["end_line"],
                chunk_type=chunk_dict["chunk_type"],
                language=chunk_dict["language"],
                metadata=chunk_dict.get("metadata")
            )
            chunk_objects.append(chunk_obj)

        save_index(index, chunk_objects, index_path)
        logger.info(f"Saved updated index to {index_path}")

        return len(new_chunks)

    def _url_to_filename(self, url: str) -> str:
        """Convert URL to safe filename."""
        filename = url.replace("https://", "").replace("http://", "")
        filename = filename.replace("/", "_").replace(":", "_")
        filename = filename[:200] + ".md"
        return filename

    async def _arun(self, url: str) -> str:
        """Async version."""
        return await self._async_crawl_and_index(url)


def get_crawl_and_index_tool(settings: Settings) -> CrawlAndIndexTool:
    """
    Factory function to get an instance of CrawlAndIndexTool.

    Args:
        settings: The application settings object.

    Returns:
        An instance of CrawlAndIndexTool.
    """
    return CrawlAndIndexTool(settings=settings)
