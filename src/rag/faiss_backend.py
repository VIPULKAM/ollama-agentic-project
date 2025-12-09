"""FAISS storage backend for RAG system.

This module implements the StorageBackend interface using FAISS (Facebook AI
Similarity Search) for local vector storage. It wraps the existing FAISS logic
from indexer.py and retriever.py to provide a clean backend interface.

FAISS is ideal for:
- Local/offline development (no database required)
- Small to medium datasets (< 1M vectors)
- Fast prototyping and testing
- Edge deployments without network access

Features:
- File-based persistence (index.faiss + chunks_metadata.json)
- IndexFlatL2 for exact similarity search
- Fast search with normalized vectors
- No external dependencies (beyond faiss-cpu)

Typical usage:
    backend = FaissBackend(index_path="~/.ai-agent/faiss_index")
    backend.build_index(chunks, embeddings)
    results = backend.search(query_embedding, top_k=5)
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS backend requires faiss-cpu. "
        "Install with: pip install faiss-cpu"
    )

from .storage_backend import (
    StorageBackend,
    SearchResult,
    IndexNotFoundError,
    IndexBuildError,
    SearchError,
    StorageBackendError
)
from .chunker import CodeChunk
from .embeddings import get_embeddings, get_embedding_dimension
from ..config.settings import settings

logger = logging.getLogger("ai_agent.faiss_backend")


class FaissBackend(StorageBackend):
    """FAISS storage backend implementation.

    This backend uses FAISS for local vector storage with file-based persistence.
    It provides exact similarity search using IndexFlatL2 (L2 distance with
    normalized vectors = cosine similarity).

    Args:
        index_path: Path to directory for storing FAISS index files
                   Default: ~/.ai-agent/faiss_index (from settings)
    """

    def __init__(self, index_path: Optional[str] = None):
        """Initialize FAISS backend."""
        if index_path is None:
            index_path = settings.FAISS_INDEX_PATH

        self.index_path = Path(index_path).expanduser()
        logger.info(f"FAISS backend initialized (path: {self.index_path})")

        # Lazy-load index (loaded on first use)
        self._index: Optional[faiss.Index] = None
        self._chunks_metadata: Optional[List[dict]] = None

    def _get_file_paths(self) -> Tuple[Path, Path, Path]:
        """Get paths to FAISS index files.

        Returns:
            Tuple of (index_file, metadata_file, info_file)
        """
        return (
            self.index_path / "index.faiss",
            self.index_path / "chunks_metadata.json",
            self.index_path / "index_info.json"
        )

    def index_exists(self) -> bool:
        """Check if FAISS index files exist.

        Returns:
            True if both index.faiss and chunks_metadata.json exist
        """
        index_file, metadata_file, _ = self._get_file_paths()
        exists = index_file.exists() and metadata_file.exists()
        logger.debug(f"Index exists check: {exists}")
        return exists

    def _load_index_from_disk(self) -> Tuple[faiss.Index, List[dict]]:
        """Load FAISS index and metadata from disk.

        Returns:
            Tuple of (FAISS index, chunks metadata list)

        Raises:
            IndexNotFoundError: If index files don't exist
        """
        index_file, metadata_file, _ = self._get_file_paths()

        # Check if files exist
        if not index_file.exists():
            raise IndexNotFoundError(
                f"Index file not found: {index_file}\n"
                f"Build index first with build_index()"
            )

        if not metadata_file.exists():
            raise IndexNotFoundError(f"Metadata file not found: {metadata_file}")

        try:
            # Load FAISS index
            index = faiss.read_index(str(index_file))
            logger.info(f"Loaded FAISS index ({index.ntotal} vectors)")

            # Load chunk metadata
            with open(metadata_file, 'r', encoding='utf-8') as f:
                chunks_metadata = json.load(f)

            logger.info(f"Loaded {len(chunks_metadata)} chunk metadata")

            return index, chunks_metadata

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise IndexNotFoundError(f"Failed to load index: {e}") from e

    def _save_index_to_disk(
        self,
        index: faiss.Index,
        chunks: List[CodeChunk]
    ) -> None:
        """Save FAISS index and metadata to disk.

        Args:
            index: FAISS index to save
            chunks: List of chunks corresponding to vectors

        Raises:
            StorageBackendError: If save fails
        """
        try:
            # Create directory if needed
            self.index_path.mkdir(parents=True, exist_ok=True)

            index_file, metadata_file, info_file = self._get_file_paths()

            # Save FAISS index
            faiss.write_index(index, str(index_file))
            logger.info(f"Saved FAISS index to {index_file}")

            # Save chunk metadata
            chunks_data = [
                {
                    "content": chunk.content,
                    "file_path": str(chunk.file_path),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                    "metadata": chunk.metadata or {}
                }
                for chunk in chunks
            ]

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(chunks)} chunk metadata")

            # Save index info
            info_data = {
                "index_type": "IndexFlatL2",
                "embedding_model": settings.EMBEDDING_MODEL,
                "embedding_dimension": get_embedding_dimension(),
                "total_vectors": index.ntotal,
                "total_chunks": len(chunks),
                "version": "1.0"
            }

            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info_data, f, indent=2)

            logger.info(f"Saved index info")

        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise StorageBackendError(f"Failed to save index: {e}") from e

    def build_index(
        self,
        chunks: List[CodeChunk],
        embeddings: List[List[float]],
        force_reindex: bool = False
    ) -> int:
        """Build FAISS index from chunks and embeddings.

        Args:
            chunks: List of CodeChunk objects
            embeddings: List of embedding vectors (384-dim floats)
            force_reindex: If True, overwrite existing index

        Returns:
            Number of chunks indexed

        Raises:
            IndexBuildError: If indexing fails
            ValueError: If chunks and embeddings lengths don't match
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        if not chunks:
            logger.warning("No chunks to index")
            return 0

        # Check if index exists and force_reindex is False
        if self.index_exists() and not force_reindex:
            raise IndexBuildError(
                f"Index already exists at {self.index_path}. "
                "Use force_reindex=True to overwrite or add_chunks() to append"
            )

        logger.info(f"Building FAISS index with {len(chunks)} chunks")

        try:
            # Get embedding dimension
            embedding_dim = get_embedding_dimension()

            # Create FAISS index (IndexFlatL2 for cosine similarity with normalized vectors)
            index = faiss.IndexFlatL2(embedding_dim)

            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Add embeddings to index
            index.add(embeddings_array)

            logger.info(f"FAISS index built with {index.ntotal} vectors")

            # Save to disk
            self._save_index_to_disk(index, chunks)

            # Update cached references
            self._index = index
            self._chunks_metadata = [
                {
                    "content": chunk.content,
                    "file_path": str(chunk.file_path),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                    "metadata": chunk.metadata or {}
                }
                for chunk in chunks
            ]

            return len(chunks)

        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise IndexBuildError(f"FAISS index building failed: {e}") from e

    def add_chunks(
        self,
        chunks: List[CodeChunk],
        embeddings: List[List[float]]
    ) -> int:
        """Add new chunks to existing FAISS index (incremental indexing).

        This loads the existing index, adds new vectors, and saves back to disk.

        Args:
            chunks: List of CodeChunk objects to add
            embeddings: List of embedding vectors

        Returns:
            Number of chunks added

        Raises:
            IndexBuildError: If adding fails
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length"
            )

        if not chunks:
            logger.warning("No chunks to add")
            return 0

        logger.info(f"Adding {len(chunks)} chunks to existing index")

        try:
            # Load existing index if it exists, otherwise create new
            if self.index_exists():
                index, existing_metadata = self._load_index_from_disk()
                logger.info(f"Loaded existing index with {index.ntotal} vectors")
            else:
                logger.info("No existing index found, creating new one")
                embedding_dim = get_embedding_dimension()
                index = faiss.IndexFlatL2(embedding_dim)
                existing_metadata = []

            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)

            # Add new embeddings to index
            index.add(embeddings_array)

            logger.info(f"Index now has {index.ntotal} vectors")

            # Combine existing and new metadata
            new_metadata = [
                {
                    "content": chunk.content,
                    "file_path": str(chunk.file_path),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                    "metadata": chunk.metadata or {}
                }
                for chunk in chunks
            ]

            all_metadata = existing_metadata + new_metadata

            # Convert metadata back to CodeChunk objects for saving
            all_chunks = []
            for meta in all_metadata:
                all_chunks.append(CodeChunk(
                    content=meta["content"],
                    file_path=Path(meta["file_path"]),
                    start_line=meta["start_line"],
                    end_line=meta["end_line"],
                    chunk_type=meta.get("chunk_type", "text"),
                    language=meta.get("language", "unknown"),
                    metadata=meta.get("metadata")
                ))

            # Save updated index
            self._save_index_to_disk(index, all_chunks)

            # Update cache
            self._index = index
            self._chunks_metadata = all_metadata

            return len(chunks)

        except Exception as e:
            logger.error(f"Failed to add chunks: {e}")
            raise IndexBuildError(f"Failed to add chunks: {e}") from e

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        **kwargs
    ) -> List[SearchResult]:
        """Perform semantic similarity search using FAISS.

        Args:
            query_embedding: Query vector (384-dim floats)
            top_k: Maximum number of results
            similarity_threshold: Minimum similarity score (0.0-1.0)
            **kwargs: Optional filters (file_path, chunk_type, language)

        Returns:
            List of SearchResult objects, sorted by similarity

        Raises:
            IndexNotFoundError: If no index exists
            SearchError: If search fails
        """
        if not self.index_exists():
            raise IndexNotFoundError(
                f"No index found at {self.index_path}. Build index first with build_index()"
            )

        logger.debug(f"Searching for top {top_k} results (threshold: {similarity_threshold})")

        try:
            # Load index if not cached
            if self._index is None or self._chunks_metadata is None:
                self._index, self._chunks_metadata = self._load_index_from_disk()

            # Convert query to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)

            # FAISS search (returns L2 distances and indices)
            distances, indices = self._index.search(query_vector, top_k)

            # Convert to SearchResult objects
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]

                # Validate index
                if idx < 0 or idx >= len(self._chunks_metadata):
                    logger.warning(f"Invalid index {idx} from FAISS search")
                    continue

                metadata = self._chunks_metadata[idx]

                # Convert L2 distance to similarity score (0-1 range)
                similarity = 1.0 / (1.0 + dist)

                # Filter by threshold
                if similarity < similarity_threshold:
                    continue

                # Apply optional filters
                if 'file_path' in kwargs and metadata.get('file_path') != kwargs['file_path']:
                    continue
                if 'chunk_type' in kwargs and metadata.get('chunk_type') != kwargs['chunk_type']:
                    continue
                if 'language' in kwargs and metadata.get('language') != kwargs['language']:
                    continue

                results.append(SearchResult(
                    file_path=metadata.get("file_path"),
                    start_line=metadata.get("start_line"),
                    end_line=metadata.get("end_line"),
                    content=metadata.get("content"),
                    score=float(similarity),
                    chunk_type=metadata.get("chunk_type", "text"),
                    language=metadata.get("language", "unknown"),
                    metadata=metadata.get("metadata", {})
                ))

            logger.debug(f"Found {len(results)} results")
            return results

        except IndexNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"FAISS search failed: {e}") from e

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the FAISS index.

        Returns:
            Dictionary with index statistics

        Raises:
            IndexNotFoundError: If no index exists
        """
        if not self.index_exists():
            raise IndexNotFoundError(f"No index found at {self.index_path}")

        try:
            # Load index if not cached
            if self._index is None or self._chunks_metadata is None:
                self._index, self._chunks_metadata = self._load_index_from_disk()

            # Calculate file sizes
            index_file, metadata_file, info_file = self._get_file_paths()
            index_size_mb = index_file.stat().st_size / (1024 * 1024) if index_file.exists() else 0
            metadata_size_mb = metadata_file.stat().st_size / (1024 * 1024) if metadata_file.exists() else 0

            # Count unique files and languages
            unique_files = len(set(m.get("file_path") for m in self._chunks_metadata))
            unique_languages = len(set(m.get("language", "unknown") for m in self._chunks_metadata))

            return {
                "total_chunks": len(self._chunks_metadata),
                "total_vectors": self._index.ntotal,
                "unique_files": unique_files,
                "unique_languages": unique_languages,
                "backend": "faiss",
                "index_size_mb": round(index_size_mb, 2),
                "metadata_size_mb": round(metadata_size_mb, 2),
                "total_size_mb": round(index_size_mb + metadata_size_mb, 2),
                "index_path": str(self.index_path)
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            raise IndexNotFoundError(f"Cannot get stats: {e}") from e

    def clear(self) -> None:
        """Clear all data from the index (delete files).

        Warning:
            This deletes the index files from disk. Cannot be undone.
        """
        logger.warning(f"Clearing FAISS index at {self.index_path}")

        try:
            index_file, metadata_file, info_file = self._get_file_paths()

            # Delete files if they exist
            if index_file.exists():
                index_file.unlink()
                logger.info(f"Deleted {index_file}")

            if metadata_file.exists():
                metadata_file.unlink()
                logger.info(f"Deleted {metadata_file}")

            if info_file.exists():
                info_file.unlink()
                logger.info(f"Deleted {info_file}")

            # Clear cache
            self._index = None
            self._chunks_metadata = None

            logger.info("Index cleared successfully")

        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
            raise StorageBackendError(f"Failed to clear index: {e}") from e

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks from a specific file.

        This requires rebuilding the index without the specified file's chunks.

        Args:
            file_path: Path to the file whose chunks should be deleted

        Returns:
            Number of chunks deleted
        """
        if not self.index_exists():
            logger.warning("No index exists, nothing to delete")
            return 0

        logger.info(f"Deleting chunks for file: {file_path}")

        try:
            # Load index
            index, chunks_metadata = self._load_index_from_disk()

            # Filter out chunks from the specified file
            remaining_metadata = [
                m for m in chunks_metadata if m.get("file_path") != file_path
            ]

            deleted_count = len(chunks_metadata) - len(remaining_metadata)

            if deleted_count == 0:
                logger.info(f"No chunks found for {file_path}")
                return 0

            # Rebuild index with remaining chunks
            remaining_chunks = []
            remaining_embeddings = []

            for i, meta in enumerate(chunks_metadata):
                if meta.get("file_path") != file_path:
                    # Reconstruct embedding from FAISS index
                    embedding = index.reconstruct(i).tolist()
                    remaining_embeddings.append(embedding)

                    # Reconstruct CodeChunk
                    remaining_chunks.append(CodeChunk(
                        content=meta["content"],
                        file_path=Path(meta["file_path"]),
                        start_line=meta["start_line"],
                        end_line=meta["end_line"],
                        chunk_type=meta.get("chunk_type", "text"),
                        language=meta.get("language", "unknown"),
                        metadata=meta.get("metadata")
                    ))

            # Rebuild index
            self.build_index(remaining_chunks, remaining_embeddings, force_reindex=True)

            logger.info(f"Deleted {deleted_count} chunks from {file_path}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            raise StorageBackendError(f"Failed to delete chunks: {e}") from e
