"""Factory for creating storage backends based on configuration.

This module provides a singleton factory that creates and manages storage
backend instances based on the application configuration. It supports both
FAISS (local/offline) and PostgreSQL+pgvector (production) backends.

Typical usage:
    from src.rag.backend_factory import get_storage_backend, reset_backend

    # Get configured backend (singleton)
    backend = get_storage_backend()
    backend.build_index(chunks, embeddings)

    # For testing: reset backend to force re-creation
    reset_backend()
"""

import logging
from typing import Optional

from ..config.settings import settings
from .storage_backend import StorageBackend

logger = logging.getLogger("ai_agent.backend_factory")

# Singleton instance
_backend_instance: Optional[StorageBackend] = None


def get_storage_backend(force_recreate: bool = False) -> StorageBackend:
    """Get the configured storage backend instance (singleton pattern).

    This function returns a singleton instance of the configured storage
    backend based on the ENABLE_POSTGRES_STORAGE setting. The backend is
    created once and reused for subsequent calls.

    The backend selection logic:
    - If ENABLE_POSTGRES_STORAGE=True: Use PostgreSQL+pgvector backend
    - If ENABLE_POSTGRES_STORAGE=False: Use FAISS backend (default)

    Args:
        force_recreate: If True, recreate the backend even if one exists.
                       Useful for testing or configuration changes.

    Returns:
        StorageBackend instance (either FaissBackend or PostgresBackend)

    Raises:
        ImportError: If required dependencies are missing
        ValueError: If configuration is invalid (e.g., missing DATABASE_URL for PostgreSQL)

    Example:
        >>> backend = get_storage_backend()
        >>> backend.build_index(chunks, embeddings)
        >>> results = backend.search(query_embedding, top_k=5)
    """
    global _backend_instance

    # Return existing instance if available and not forcing recreation
    if _backend_instance is not None and not force_recreate:
        return _backend_instance

    # Log which backend we're creating
    if settings.ENABLE_POSTGRES_STORAGE:
        logger.info("Creating PostgreSQL + pgvector storage backend")
        _backend_instance = _create_postgres_backend()
    else:
        logger.info("Creating FAISS storage backend (local)")
        _backend_instance = _create_faiss_backend()

    logger.info(f"Storage backend initialized: {_backend_instance.__class__.__name__}")
    return _backend_instance


def _create_faiss_backend() -> StorageBackend:
    """Create a FAISS storage backend instance.

    Returns:
        FaissBackend instance

    Raises:
        ImportError: If FAISS or dependencies are not installed
    """
    try:
        from .faiss_backend import FaissBackend
    except ImportError as e:
        logger.error("Failed to import FaissBackend. Is faiss-cpu installed?")
        raise ImportError(
            "FAISS backend requires faiss-cpu. Install with: pip install faiss-cpu"
        ) from e

    return FaissBackend(
        index_path=settings.FAISS_INDEX_PATH
    )


def _create_postgres_backend() -> StorageBackend:
    """Create a PostgreSQL + pgvector storage backend instance.

    Returns:
        PostgresBackend instance

    Raises:
        ImportError: If psycopg2 or dependencies are not installed
        ValueError: If DATABASE_URL is not configured
    """
    # Validate configuration
    if not settings.DATABASE_URL:
        raise ValueError(
            "DATABASE_URL must be set in .env when ENABLE_POSTGRES_STORAGE=True. "
            "Example: postgresql://user:password@localhost:5432/ai_agent_rag"
        )

    # Try to import PostgreSQL backend
    try:
        from .postgres_backend import PostgresBackend
    except ImportError as e:
        logger.error("Failed to import PostgresBackend. Is psycopg2-binary installed?")
        raise ImportError(
            "PostgreSQL backend requires psycopg2-binary. "
            "Install with: pip install psycopg2-binary"
        ) from e

    return PostgresBackend(
        database_url=settings.DATABASE_URL,
        pool_size=settings.DB_POOL_SIZE,
        connection_timeout=settings.DB_CONNECTION_TIMEOUT
    )


def reset_backend() -> None:
    """Reset the backend singleton (useful for testing).

    This function clears the cached backend instance, forcing the next call
    to get_storage_backend() to create a new instance. Useful for:
    - Testing with different configurations
    - Switching between FAISS and PostgreSQL at runtime
    - Cleanup after integration tests

    Example:
        >>> # Test with FAISS
        >>> backend = get_storage_backend()
        >>> assert isinstance(backend, FaissBackend)
        >>>
        >>> # Switch to PostgreSQL
        >>> settings.ENABLE_POSTGRES_STORAGE = True
        >>> reset_backend()
        >>> backend = get_storage_backend()
        >>> assert isinstance(backend, PostgresBackend)
    """
    global _backend_instance

    if _backend_instance is not None:
        logger.debug(f"Resetting backend: {_backend_instance.__class__.__name__}")
        _backend_instance = None
    else:
        logger.debug("Backend already reset or never initialized")


def get_backend_type() -> str:
    """Get the type of the currently configured backend.

    Returns:
        "faiss" or "postgresql" depending on configuration

    Example:
        >>> get_backend_type()
        'faiss'
    """
    return "postgresql" if settings.ENABLE_POSTGRES_STORAGE else "faiss"


def is_backend_initialized() -> bool:
    """Check if a backend instance has been created.

    Returns:
        True if backend instance exists, False otherwise
    """
    return _backend_instance is not None
