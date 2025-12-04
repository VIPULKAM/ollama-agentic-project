"""RAG (Retrieval-Augmented Generation) system for codebase understanding."""

# Embeddings module
from .embeddings import (
    get_embeddings,
    get_model,
    get_embedding_dimension,
    get_max_sequence_length,
    clear_model_cache,
    EmbeddingError,
    ModelLoadError,
    EmbeddingGenerationError,
    MODEL_INFO,
)

# Indexer module
from .indexer import (
    build_index,
    load_index,
    save_index,
    index_exists,
    IndexError,
    IndexBuildError,
    IndexSaveError,
)

from .indexer_utils import (
    FileInfo,
    load_gitignore_patterns,
    file_discovery_generator,
)

from .chunker import (
    CodeChunk,
    CodeChunker,
    TextChunker,
    chunk_file,
)

# Components to be implemented
# from .retriever import search_codebase

__all__ = [
    # Embeddings
    "get_embeddings",
    "get_model",
    "get_embedding_dimension",
    "get_max_sequence_length",
    "clear_model_cache",
    "EmbeddingError",
    "ModelLoadError",
    "EmbeddingGenerationError",
    "MODEL_INFO",
    # Indexer
    "build_index",
    "load_index",
    "save_index",
    "index_exists",
    "IndexError",
    "IndexBuildError",
    "IndexSaveError",
    # Indexer Utils
    "FileInfo",
    "load_gitignore_patterns",
    "file_discovery_generator",
    # Chunker
    "CodeChunk",
    "CodeChunker",
    "TextChunker",
    "chunk_file",
]
