"""Embedding model management for the RAG system.

This module provides a singleton-loaded SentenceTransformer model for converting
text chunks into vector embeddings. The model is loaded once and cached for
efficient reuse across indexing and retrieval operations.

Model: sentence-transformers/all-MiniLM-L6-v2
- Embedding dimension: 384
- Max sequence length: 256 tokens
- Local execution (no API calls)
- Small (80MB) and fast
"""

import functools
import logging
from typing import List, Optional

from ..config.settings import settings

# Configure logging
logger = logging.getLogger("ai_agent.embeddings")


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass


class ModelLoadError(EmbeddingError):
    """Exception raised when model loading fails."""
    pass


class EmbeddingGenerationError(EmbeddingError):
    """Exception raised when embedding generation fails."""
    pass


@functools.lru_cache(maxsize=1)
def _load_model():
    """Load the SentenceTransformer model (singleton pattern).

    This function is cached using lru_cache, ensuring the model is loaded
    only once during the application lifetime. Subsequent calls return the
    cached model instance.

    Returns:
        SentenceTransformer: Loaded embedding model

    Raises:
        ModelLoadError: If model loading fails due to:
            - Missing dependencies (torch, sentence-transformers)
            - Network issues (first-time download)
            - Insufficient disk space
            - Corrupted model files
    """
    try:
        # Import here to provide better error messages if missing
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ModelLoadError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            ) from e

        try:
            import torch
        except ImportError as e:
            raise ModelLoadError(
                "torch package not installed. "
                "Install with: pip install torch"
            ) from e

        # Load the model
        model_name = settings.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {model_name}")

        try:
            model = SentenceTransformer(model_name)
            logger.info(
                f"Model loaded successfully. "
                f"Embedding dimension: {model.get_sentence_embedding_dimension()}, "
                f"Max sequence length: {model.max_seq_length}"
            )
            return model

        except Exception as e:
            # Catch model-specific errors (network, file system, etc.)
            raise ModelLoadError(
                f"Failed to load model '{model_name}'. "
                f"Error: {str(e)}. "
                f"If this is the first run, ensure you have internet connection "
                f"to download the model (~80MB). The model will be cached locally "
                f"for future use."
            ) from e

    except ModelLoadError:
        # Re-raise ModelLoadError as-is
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise ModelLoadError(
            f"Unexpected error loading embedding model: {str(e)}"
        ) from e


def get_model():
    """Get the singleton embedding model instance.

    This is a convenience wrapper around _load_model() for external use.

    Returns:
        SentenceTransformer: Loaded embedding model

    Raises:
        ModelLoadError: If model loading fails
    """
    return _load_model()


def get_embeddings(
    texts: List[str],
    batch_size: Optional[int] = None,
    show_progress: bool = False,
    normalize: bool = True
) -> List[List[float]]:
    """Convert text chunks into vector embeddings.

    This function provides a clean interface for generating embeddings from text.
    It handles batching, normalization, and error handling automatically.

    Args:
        texts: List of text strings to embed. Empty strings are allowed but
            will produce zero vectors.
        batch_size: Number of texts to process at once. If None, uses the
            model's default (32). Larger batches are faster but use more memory.
        show_progress: If True, display a progress bar during encoding.
            Useful for large batches during indexing.
        normalize: If True (default), normalize embeddings to unit length.
            This is required for cosine similarity with FAISS.

    Returns:
        List[List[float]]: List of embedding vectors, one per input text.
            Each vector has 384 dimensions for all-MiniLM-L6-v2.

    Raises:
        EmbeddingGenerationError: If embedding generation fails
        ModelLoadError: If model loading fails
        ValueError: If texts is empty

    Example:
        >>> texts = ["Hello world", "Python programming"]
        >>> embeddings = get_embeddings(texts)
        >>> len(embeddings)
        2
        >>> len(embeddings[0])
        384
    """
    # Input validation
    if not texts:
        raise ValueError("Input texts list cannot be empty")

    if not isinstance(texts, list):
        raise ValueError("Input must be a list of strings")

    # Validate all items are strings
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(f"Item at index {i} is not a string: {type(text)}")

    try:
        # Load model (cached after first call)
        model = get_model()

        # Set batch size
        if batch_size is None:
            batch_size = 32  # Default batch size

        logger.debug(
            f"Generating embeddings for {len(texts)} texts "
            f"(batch_size={batch_size}, normalize={normalize})"
        )

        # Generate embeddings
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True  # Return numpy arrays (faster)
        )

        # Convert numpy arrays to Python lists for JSON serialization
        embeddings_list = embeddings.tolist()

        logger.debug(
            f"Generated {len(embeddings_list)} embeddings "
            f"(dimension: {len(embeddings_list[0])})"
        )

        return embeddings_list

    except ModelLoadError:
        # Re-raise model loading errors
        raise
    except Exception as e:
        # Catch and wrap any other errors
        raise EmbeddingGenerationError(
            f"Failed to generate embeddings: {str(e)}"
        ) from e


def get_embedding_dimension() -> int:
    """Get the dimension of the embedding vectors.

    Returns:
        int: Embedding dimension (384 for all-MiniLM-L6-v2)

    Raises:
        ModelLoadError: If model loading fails
    """
    model = get_model()
    return model.get_sentence_embedding_dimension()


def get_max_sequence_length() -> int:
    """Get the maximum sequence length supported by the model.

    Text longer than this will be truncated.

    Returns:
        int: Maximum sequence length in tokens (256 for all-MiniLM-L6-v2)

    Raises:
        ModelLoadError: If model loading fails
    """
    model = get_model()
    return model.max_seq_length


def clear_model_cache():
    """Clear the cached model from memory.

    This is useful for testing or if you need to free up memory.
    The model will be reloaded on the next call to get_embeddings().
    """
    _load_model.cache_clear()
    logger.info("Embedding model cache cleared")


# Expose key information as module-level constants (after first model load)
def _get_model_info():
    """Get model information without triggering full load in case of errors."""
    try:
        return {
            "model_name": settings.EMBEDDING_MODEL,
            "dimension": 384,  # Known value for all-MiniLM-L6-v2
            "max_seq_length": 256,  # Known value for all-MiniLM-L6-v2
        }
    except Exception:
        return {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "max_seq_length": 256,
        }


MODEL_INFO = _get_model_info()
