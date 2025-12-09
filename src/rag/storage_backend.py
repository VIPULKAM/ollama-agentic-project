"""Abstract storage backend interface for RAG vector storage.

This module defines the abstract base class that all vector storage backends
(FAISS, PostgreSQL+pgvector, etc.) must implement. This allows the RAG system
to switch between different storage solutions via configuration.

Typical usage:
    backend = get_storage_backend()  # From backend_factory
    backend.build_index(chunks, embeddings)
    results = backend.search(query_embedding, top_k=5)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path

from .chunker import CodeChunk

logger = logging.getLogger("ai_agent.storage_backend")


# Custom exceptions
class StorageBackendError(Exception):
    """Base exception for storage backend errors."""
    pass


class IndexNotFoundError(StorageBackendError):
    """Raised when attempting to use a non-existent index."""
    pass


class IndexBuildError(StorageBackendError):
    """Raised when index building fails."""
    pass


class SearchError(StorageBackendError):
    """Raised when search operation fails."""
    pass


@dataclass
class SearchResult:
    """Unified search result format across all backends.

    Attributes:
        file_path: Path to the source file
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)
        content: The actual code/text content
        score: Similarity score (0.0-1.0, higher is better)
        chunk_type: Type of chunk (function, class, method, text)
        language: Programming language or file type
        metadata: Additional metadata (function name, class name, etc.)
    """
    file_path: str
    start_line: int
    end_line: int
    content: str
    score: float
    chunk_type: str = "text"
    language: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "score": self.score,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "metadata": self.metadata or {}
        }


class StorageBackend(ABC):
    """Abstract base class for vector storage backends.

    All storage backends (FAISS, PostgreSQL+pgvector, etc.) must implement
    this interface to ensure consistent behavior across different storage
    solutions.

    The backend is responsible for:
    - Storing code chunks with their vector embeddings
    - Performing semantic similarity search
    - Managing index lifecycle (create, update, delete)
    - Providing statistics about the stored data
    """

    @abstractmethod
    def index_exists(self) -> bool:
        """Check if an index exists.

        Returns:
            True if index exists and is accessible, False otherwise
        """
        pass

    @abstractmethod
    def build_index(
        self,
        chunks: List[CodeChunk],
        embeddings: List[List[float]],
        force_reindex: bool = False
    ) -> int:
        """Build or rebuild the index from chunks and embeddings.

        This method creates a new index from scratch, replacing any existing
        index if force_reindex is True. If the index exists and force_reindex
        is False, behavior is backend-specific (may raise error or append).

        Args:
            chunks: List of CodeChunk objects containing content and metadata
            embeddings: List of embedding vectors (one per chunk)
                       Each embedding should be a list of floats (384-dim for all-MiniLM-L6-v2)
            force_reindex: If True, replace existing index. If False, behavior is backend-specific

        Returns:
            Number of chunks successfully indexed

        Raises:
            IndexBuildError: If index building fails
            ValueError: If chunks and embeddings have different lengths
        """
        pass

    @abstractmethod
    def add_chunks(
        self,
        chunks: List[CodeChunk],
        embeddings: List[List[float]]
    ) -> int:
        """Add new chunks to an existing index (incremental indexing).

        This method adds new chunks without rebuilding the entire index.
        If no index exists, creates a new one.

        Args:
            chunks: List of CodeChunk objects to add
            embeddings: List of embedding vectors (one per chunk)

        Returns:
            Number of chunks successfully added

        Raises:
            IndexBuildError: If adding chunks fails
            ValueError: If chunks and embeddings have different lengths
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        **kwargs
    ) -> List[SearchResult]:
        """Perform semantic similarity search.

        Given a query embedding, find the most similar code chunks in the index.

        Args:
            query_embedding: Embedding vector for the search query (384-dim)
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0) to include in results
            **kwargs: Backend-specific search parameters (e.g., filters, metadata queries)

        Returns:
            List of SearchResult objects, sorted by similarity (highest first)

        Raises:
            IndexNotFoundError: If no index exists
            SearchError: If search operation fails
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index.

        Returns:
            Dictionary with statistics, including:
                - total_chunks: Total number of indexed chunks
                - total_vectors: Total number of vectors (should match chunks)
                - unique_files: Number of unique source files
                - backend: Backend type (e.g., "faiss", "postgresql")
                - index_size_mb: Approximate size of index in MB (if applicable)
                - last_updated: Last update timestamp (if tracked)
                Additional backend-specific statistics may be included

        Raises:
            IndexNotFoundError: If no index exists
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the index.

        This removes all chunks, embeddings, and metadata. The operation
        cannot be undone.

        Warning:
            This is a destructive operation. All data will be permanently lost.

        Raises:
            StorageBackendError: If clear operation fails
        """
        pass

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks from a specific file (optional).

        This is an optional method that backends may implement for
        fine-grained control over indexed content.

        Args:
            file_path: Path to the file whose chunks should be deleted

        Returns:
            Number of chunks deleted

        Raises:
            NotImplementedError: If backend doesn't support this operation
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support delete_by_file"
        )

    def update_chunk(
        self,
        chunk_id: Any,
        chunk: CodeChunk,
        embedding: List[float]
    ) -> bool:
        """Update a specific chunk (optional).

        This is an optional method that backends may implement for
        updating individual chunks without rebuilding the entire index.

        Args:
            chunk_id: Backend-specific identifier for the chunk
            chunk: Updated CodeChunk object
            embedding: Updated embedding vector

        Returns:
            True if successful, False otherwise

        Raises:
            NotImplementedError: If backend doesn't support this operation
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support update_chunk"
        )
