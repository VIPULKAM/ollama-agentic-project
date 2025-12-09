"""
Semantic Retriever for RAG System

This module provides the SemanticRetriever class, which performs semantic searches
using the configured storage backend (FAISS or PostgreSQL+pgvector).

The backend is selected automatically based on the ENABLE_POSTGRES_STORAGE setting.
"""

import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config.settings import Settings
from src.rag.embeddings import get_embeddings
from src.rag.backend_factory import get_storage_backend
from src.rag.storage_backend import IndexNotFoundError
from src.utils.logging import rag_logger as logger

class SemanticRetriever:
    """
    Handles semantic searches using the configured storage backend.

    This class now uses the backend factory to support both FAISS (local)
    and PostgreSQL+pgvector (production) backends transparently.
    """

    def __init__(self, settings: Settings):
        """
        Initializes the SemanticRetriever with the configured backend.

        Args:
            settings: The application settings object.
        """
        self.settings = settings

        # Get backend - if settings has a custom FAISS path, create a fresh backend
        from pathlib import Path
        backend = get_storage_backend()

        # Check if we need a custom backend for FAISS
        if hasattr(backend, 'index_path') and hasattr(settings, 'FAISS_INDEX_PATH'):
            custom_path = Path(settings.FAISS_INDEX_PATH).expanduser()
            if backend.index_path != custom_path:
                # Create a fresh backend with custom path
                logger.info(f"Creating fresh FAISS backend for custom path: {custom_path}")
                from .faiss_backend import FaissBackend
                backend = FaissBackend(index_path=str(custom_path))

        self.backend = backend
        logger.info(f"SemanticRetriever initialized with backend: {self.backend.__class__.__name__}")

        # Check if index exists
        if not self.backend.index_exists():
            raise IndexNotFoundError(
                f"No index found. Please build the index first using 'python main.py index' "
                f"or the indexing functions."
            )

    @property
    def index(self):
        """Backward compatibility: Return FAISS index for FAISS backend."""
        if hasattr(self.backend, '_index'):
            # Ensure index is loaded
            if self.backend._index is None or self.backend._chunks_metadata is None:
                self.backend._index, self.backend._chunks_metadata = self.backend._load_index_from_disk()
            return self.backend._index
        return None

    @property
    def chunks_metadata(self):
        """Backward compatibility: Return chunks metadata for FAISS backend."""
        if hasattr(self.backend, '_chunks_metadata'):
            # Ensure metadata is loaded
            if self.backend._index is None or self.backend._chunks_metadata is None:
                self.backend._index, self.backend._chunks_metadata = self.backend._load_index_from_disk()
            return self.backend._chunks_metadata
        return []

    @property
    def index_info(self):
        """Backward compatibility: Return index info."""
        import json
        from pathlib import Path
        if hasattr(self.backend, 'index_path'):
            info_file = self.backend.index_path / "index_info.json"
            if info_file.exists():
                with open(info_file, 'r') as f:
                    return json.load(f)
        return {}

    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search for the given query using the configured backend.

        Args:
            query: The natural language query to search for.
            top_k: The number of top results to return.
            similarity_threshold: The minimum similarity score for a result to be included.

        Returns:
            A list of dictionaries, where each dictionary represents a
            search result and contains the file path, content, line numbers,
            and a relevance score.
        """
        logger.debug(f"Generating embedding for query: '{query}'")
        query_embedding = get_embeddings([query], normalize=True)[0]

        logger.debug(f"Searching index for top {top_k} results (threshold: {similarity_threshold})")

        # Use backend for search
        search_results = self.backend.search(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # Convert SearchResult objects to dictionaries for backward compatibility
        results = []
        for result in search_results:
            results.append({
                "file_path": result.file_path,
                "start_line": result.start_line,
                "end_line": result.end_line,
                "content": result.content,
                "score": result.score,
                "chunk_type": result.chunk_type,
                "language": result.language,
                "metadata": result.metadata or {}
            })
            
        return results

def get_retriever(settings: Optional[Settings] = None) -> SemanticRetriever:
    """
    Factory function to get an instance of the SemanticRetriever.
    """
    if settings is None:
        settings = Settings()
    return SemanticRetriever(settings)
