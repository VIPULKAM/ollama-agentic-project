"""Index management utilities for FAISS index operations.

This module provides high-level operations for managing the RAG index:
- Statistics and information about the index
- Rebuilding the index from scratch
- Cleaning orphaned chunks
- Direct search interface
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .indexer import build_index, load_index, index_exists, save_index
from .retriever import SemanticRetriever
from .crawl_tracker import get_crawl_tracker
from ..config.settings import settings

logger = logging.getLogger(__name__)


class IndexNotFoundError(Exception):
    """Raised when index does not exist."""
    pass


def get_index_info() -> Dict[str, Any]:
    """Get comprehensive statistics about the current index.

    Returns:
        Dict with index statistics:
        - exists: bool
        - total_chunks: int
        - total_files: int
        - crawled_urls: int
        - index_size_mb: float
        - last_updated: str (ISO datetime)
        - chunks_by_source: Dict (local vs crawled breakdown)
        - chunks_by_language: Dict (breakdown by file type)

    Raises:
        IndexNotFoundError: If index does not exist
    """
    index_path = Path(settings.FAISS_INDEX_PATH)

    if not index_exists():
        return {
            "exists": False,
            "message": "No index found. Run 'index --rebuild' to create one."
        }

    # Load index metadata
    metadata_file = index_path / "chunks_metadata.json"
    info_file = index_path / "index_info.json"

    try:
        with open(metadata_file, 'r') as f:
            chunks = json.load(f)

        with open(info_file, 'r') as f:
            index_info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise IndexNotFoundError(f"Failed to load index metadata: {e}")

    # Calculate statistics
    total_chunks = len(chunks)

    # Get unique files
    files = set()
    for chunk in chunks:
        files.add(chunk.get('file_path', 'unknown'))
    total_files = len(files)

    # Get crawled docs count
    tracker = get_crawl_tracker()
    crawled_stats = tracker.get_stats()

    # Breakdown by source (local vs crawled)
    crawled_docs_path = Path(settings.CRAWLED_DOCS_PATH)
    chunks_by_source = {
        "local": 0,
        "crawled": 0
    }

    for chunk in chunks:
        file_path = Path(chunk.get('file_path', ''))
        try:
            if crawled_docs_path in file_path.parents or file_path.parent == crawled_docs_path:
                chunks_by_source["crawled"] += 1
            else:
                chunks_by_source["local"] += 1
        except (ValueError, OSError):
            chunks_by_source["local"] += 1

    # Breakdown by language/type
    chunks_by_language = {}
    for chunk in chunks:
        lang = chunk.get('language', 'unknown')
        chunks_by_language[lang] = chunks_by_language.get(lang, 0) + 1

    # Calculate index size on disk
    index_size_bytes = 0
    for file in index_path.glob('*'):
        if file.is_file():
            index_size_bytes += file.stat().st_size
    index_size_mb = index_size_bytes / (1024 * 1024)

    # Get last updated timestamp
    index_file = index_path / "index.faiss"
    last_updated = datetime.fromtimestamp(index_file.stat().st_mtime).isoformat()

    return {
        "exists": True,
        "total_chunks": total_chunks,
        "total_files": total_files,
        "crawled_urls": crawled_stats["total_urls"],
        "index_size_mb": round(index_size_mb, 2),
        "last_updated": last_updated,
        "chunks_by_source": chunks_by_source,
        "chunks_by_language": dict(sorted(chunks_by_language.items(), key=lambda x: x[1], reverse=True)),
        "index_version": index_info.get("version", "unknown"),
        "embedding_model": index_info.get("embedding_model", "unknown")
    }


def rebuild_index(
    root_path: Optional[Path] = None,
    include_crawled: bool = True,
    show_progress: bool = True
) -> Dict[str, Any]:
    """Rebuild the entire FAISS index from scratch.

    Args:
        root_path: Root directory to index (defaults to current directory)
        include_crawled: Whether to include crawled documentation (default: True)
        show_progress: Show progress bars (default: True)

    Returns:
        Dict with rebuild statistics:
        - success: bool
        - chunks_indexed: int
        - files_processed: int
        - errors: List[str]
        - duration_seconds: float
    """
    start_time = datetime.now()

    if root_path is None:
        root_path = Path.cwd()

    logger.info(f"Rebuilding index from {root_path}")

    try:
        # Build index from local files - force_reindex=True to rebuild from scratch
        index, chunks = build_index(
            force_reindex=True,
            cwd=root_path,
            show_progress=show_progress
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Get statistics
        files_processed = len(set(chunk.file_path for chunk in chunks))

        logger.info(f"Index rebuilt successfully: {len(chunks)} chunks from {files_processed} files in {duration:.1f}s")

        return {
            "success": True,
            "chunks_indexed": len(chunks),
            "files_processed": files_processed,
            "errors": [],
            "duration_seconds": round(duration, 2)
        }

    except Exception as e:
        logger.error(f"Failed to rebuild index: {e}")
        duration = (datetime.now() - start_time).total_seconds()
        return {
            "success": False,
            "chunks_indexed": 0,
            "files_processed": 0,
            "errors": [str(e)],
            "duration_seconds": round(duration, 2)
        }


def clean_index() -> Dict[str, Any]:
    """Clean orphaned chunks from the index.

    Removes chunks whose source files no longer exist.

    Returns:
        Dict with cleaning statistics:
        - success: bool
        - chunks_removed: int
        - chunks_remaining: int
        - files_removed: List[str]
    """
    if not index_exists():
        raise IndexNotFoundError("No index found. Nothing to clean.")

    logger.info("Cleaning orphaned chunks from index...")

    # Load current index
    index, chunks = load_index()

    # Check which chunks have missing source files
    valid_chunks = []
    removed_chunks = []
    removed_files = set()

    for chunk_dict in chunks:
        file_path = Path(chunk_dict.get('file_path', ''))

        # Check if file exists
        if file_path.exists():
            valid_chunks.append(chunk_dict)
        else:
            removed_chunks.append(chunk_dict)
            removed_files.add(str(file_path))

    chunks_removed = len(removed_chunks)

    if chunks_removed == 0:
        logger.info("No orphaned chunks found. Index is clean.")
        return {
            "success": True,
            "chunks_removed": 0,
            "chunks_remaining": len(valid_chunks),
            "files_removed": []
        }

    # Rebuild index with only valid chunks
    logger.info(f"Removing {chunks_removed} orphaned chunks from {len(removed_files)} deleted files")

    # Re-create index with valid chunks only
    from .embeddings import get_embeddings
    from .chunker import CodeChunk
    import faiss
    import numpy as np

    # Get embeddings for valid chunks
    valid_texts = [chunk.get('content', '') for chunk in valid_chunks]
    embeddings = get_embeddings(valid_texts, show_progress=True)

    # Create new index
    embeddings_array = np.array(embeddings, dtype=np.float32)
    dimension = embeddings_array.shape[1]
    new_index = faiss.IndexFlatL2(dimension)
    new_index.add(embeddings_array)

    # Convert dict chunks back to CodeChunk objects for save_index
    chunk_objects = []
    for chunk_dict in valid_chunks:
        chunk_obj = CodeChunk(
            content=chunk_dict.get('content', ''),
            file_path=Path(chunk_dict.get('file_path', '')),
            start_line=chunk_dict.get('start_line', 0),
            end_line=chunk_dict.get('end_line', 0),
            chunk_type=chunk_dict.get('chunk_type', 'text'),
            language=chunk_dict.get('language', 'unknown'),
            metadata=chunk_dict.get('metadata', {})
        )
        chunk_objects.append(chunk_obj)

    # Save cleaned index
    index_path = Path(settings.FAISS_INDEX_PATH)
    save_index(new_index, chunk_objects, index_path)

    logger.info(f"Index cleaned: {chunks_removed} chunks removed, {len(valid_chunks)} remaining")

    return {
        "success": True,
        "chunks_removed": chunks_removed,
        "chunks_remaining": len(valid_chunks),
        "files_removed": sorted(removed_files)
    }


def search_index(
    query: str,
    top_k: int = 10,
    threshold: float = 0.5,
    source_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search the index directly and return results.

    Useful for testing and debugging search relevance.

    Args:
        query: Search query
        top_k: Number of results to return (default: 10)
        threshold: Minimum similarity threshold (default: 0.5)
        source_filter: Filter by source ("local" or "crawled", optional)

    Returns:
        List of search results with:
        - content: str
        - file_path: str
        - score: float
        - language: str
        - chunk_type: str
        - line_numbers: str (if available)

    Raises:
        IndexNotFoundError: If index does not exist
    """
    if not index_exists():
        raise IndexNotFoundError("No index found. Run 'index --rebuild' to create one.")

    # Use the semantic retriever
    retriever = SemanticRetriever(settings)
    results = retriever.search(query, top_k=top_k, similarity_threshold=threshold)

    # Apply source filter if specified
    if source_filter:
        crawled_docs_path = Path(settings.CRAWLED_DOCS_PATH)
        filtered_results = []

        for result in results:
            file_path = Path(result["file_path"])

            try:
                is_crawled = crawled_docs_path in file_path.parents or file_path.parent == crawled_docs_path
            except (ValueError, OSError):
                is_crawled = False

            if source_filter == "crawled" and is_crawled:
                filtered_results.append(result)
            elif source_filter == "local" and not is_crawled:
                filtered_results.append(result)

        return filtered_results

    return results
