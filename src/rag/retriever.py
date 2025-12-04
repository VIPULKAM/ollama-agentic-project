"""
Semantic Retriever for RAG System

This module provides the SemanticRetriever class, which is responsible for loading a 
FAISS index and performing semantic searches on it.
"""

import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.config.settings import Settings
from src.rag.embeddings import get_embeddings
from src.utils.logging import rag_logger as logger

class IndexNotFoundError(Exception):
    """Custom exception for when the FAISS index or metadata is not found."""
    pass

class SemanticRetriever:
    """
    Handles loading a FAISS index and performing semantic searches.
    """

    def __init__(self, settings: Settings):
        """
        Initializes the SemanticRetriever.

        Args:
            settings: The application settings object.
        """
        self.settings = settings
        self.index_path = Path(self.settings.FAISS_INDEX_PATH)
        self.index: Optional[faiss.Index] = None
        self.chunks_metadata: List[Dict[str, Any]] = []
        self.index_info: Dict[str, Any] = {}
        
        self.load_index()

    def load_index(self):
        """
        Loads the FAISS index and associated metadata from disk.

        Raises:
            IndexNotFoundError: If the index or metadata files do not exist.
        """
        faiss_file = self.index_path / "index.faiss"
        metadata_file = self.index_path / "chunks_metadata.json"
        info_file = self.index_path / "index_info.json"

        if not all([faiss_file.exists(), metadata_file.exists(), info_file.exists()]):
            raise IndexNotFoundError(
                f"Index files not found in {self.index_path}. "
                f"Please build the index first using 'python main.py index'."
            )

        logger.info(f"Loading FAISS index from {faiss_file}")
        self.index = faiss.read_index(str(faiss_file))

        logger.info(f"Loading chunks metadata from {metadata_file}")
        with open(metadata_file, "r", encoding="utf-8") as f:
            self.chunks_metadata = json.load(f)

        logger.info(f"Loading index info from {info_file}")
        with open(info_file, "r", encoding="utf-8") as f:
            self.index_info = json.load(f)

        logger.info(
            f"Successfully loaded index with {self.index.ntotal} vectors "
            f"and {len(self.chunks_metadata)} chunk metadata records."
        )

    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Performs a semantic search for the given query.

        Args:
            query: The natural language query to search for.
            top_k: The number of top results to return.
            similarity_threshold: The minimum similarity score for a result to be included.

        Returns:
            A list of dictionaries, where each dictionary represents a
            search result and contains the file path, content, line numbers,
            and a relevance score.
        """
        if not self.index:
            raise IndexNotFoundError("Index is not loaded.")

        logger.debug(f"Generating embedding for query: '{query}'")
        query_embedding = get_embeddings([query], normalize=True)[0]
        
        query_vector = np.array([query_embedding], dtype=np.float32)

        logger.debug(f"Searching index for top {top_k} results.")
        # FAISS returns distances (L2) and indices (I)
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]

            if idx < 0 or idx >= len(self.chunks_metadata):
                logger.warning(f"Invalid index {idx} returned from FAISS search. Skipping.")
                continue

            metadata = self.chunks_metadata[idx]
            
            # Convert L2 distance to a similarity score (0-1 range).
            similarity = 1.0 / (1.0 + dist)

            if similarity >= similarity_threshold:
                results.append({
                    "file_path": metadata.get("file_path"),
                    "start_line": metadata.get("start_line"),
                    "end_line": metadata.get("end_line"),
                    "content": metadata.get("content"),
                    "score": similarity,
                })
            
        return results

def get_retriever(settings: Optional[Settings] = None) -> SemanticRetriever:
    """
    Factory function to get an instance of the SemanticRetriever.
    """
    if settings is None:
        settings = Settings()
    return SemanticRetriever(settings)
