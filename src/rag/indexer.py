import json
import logging
from pathlib import Path
import time # Import time module
from typing import List, Optional, Tuple, Dict, Any

import faiss
import numpy as np
from tqdm import tqdm # Always import tqdm, it's conditionally used below

from ..config.settings import settings
from .embeddings import get_embeddings, get_embedding_dimension
from .chunker import chunk_file, CodeChunk
from .indexer_utils import file_discovery_generator, FileInfo

# Configure logging
logger = logging.getLogger("ai_agent.indexer")


class IndexError(Exception):
    """Base exception for indexing errors."""
    pass


class IndexBuildError(IndexError):
    """Exception raised when index building fails."""
    pass


class IndexSaveError(IndexError):
    """Exception raised when index saving fails."""
    pass


def build_index(
    force_reindex: bool = False,
    cwd: Optional[Path] = None,
    index_path: Optional[Path] = None,
    show_progress: bool = True,
    batch_size: int = 32,
) -> Tuple[faiss.Index, List[CodeChunk]]:
    """Build FAISS index from codebase files.

    This function orchestrates the entire indexing process:
    1. Discovers files.
    2. Chunks each file.
    3. Generates embeddings for all chunks in batches.
    4. Builds FAISS index.
    5. Saves index and metadata to disk.
    It also handles incremental indexing by checking for existing indices.

    Args:
        force_reindex: If True, forces a full rebuild of the index.
        cwd: Working directory to index (default: current directory).
        index_path: Path to save/load index (default: from settings).
        show_progress: Display progress bars (default: True).
        batch_size: Number of chunks to embed at once (default: 32).

    Returns:
        Tuple[faiss.Index, List[CodeChunk]]: The built index and list of chunks.

    Raises:
        IndexBuildError: If index building fails.
        IndexSaveError: If index saving fails.
    """
    start_time = time.time()

    if cwd is None:
        cwd = Path.cwd()

    if index_path is None:
        index_path = Path(settings.FAISS_INDEX_PATH).expanduser()

    if not force_reindex and index_exists(index_path):
        logger.info(f"Index already exists at {index_path}. Loading existing index.")
        try:
            index, chunks_metadata = load_index(index_path)
            logger.info(f"Loaded existing index with {len(chunks_metadata)} chunks")
            return index, chunks_metadata
        except IndexError as e:
            logger.warning(f"Failed to load existing index: {e}. Attempting to rebuild.")
            # Fall through to rebuild if loading fails

    logger.info(f"Starting index build for {cwd}")
    logger.info(f"Index will be saved to {index_path}")

    try:
        # Initialize statistics
        total_files = 0
        total_chunks = 0
        failed_files = 0

        # Collect all chunks from all files
        all_chunks: List[CodeChunk] = []

        # Step 1: Discover files
        logger.info("Discovering files...")
        files_to_process = list(file_discovery_generator(
            cwd=cwd,
            show_progress=show_progress
        ))
        logger.info(f"Discovered {len(files_to_process)} files to index")

        if not files_to_process:
            logger.warning("No files found to index!")
            raise IndexBuildError("No files found to index. Check your file filters and exclusions.")

        # Step 2: Chunk all files
        logger.info("Chunking files...")
        file_iterator = tqdm(files_to_process, desc="Processing files", unit="file", disable=not show_progress)

        for file_info in file_iterator:
            total_files += 1

            try:
                # Chunk the file
                chunks = chunk_file(file_info.path)

                if chunks:
                    all_chunks.extend(chunks)
                    total_chunks += len(chunks)
                    logger.debug(f"Chunked {file_info.relative_path}: {len(chunks)} chunks")
                else:
                    logger.warning(f"No chunks extracted from {file_info.relative_path}")

            except Exception as e:
                # Per-file error handling (medium priority: don't crash on one bad file)
                logger.error(f"Failed to chunk {file_info.relative_path}: {e}")
                failed_files += 1
                continue

        logger.info(
            f"Chunking complete: {total_chunks} chunks from {total_files} files "
            f"({failed_files} files failed)"
        )

        if not all_chunks:
            raise IndexBuildError("No chunks extracted from any files!")

        # Step 3: Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")

        # Extract chunk content for embedding
        chunk_texts = [chunk.content for chunk in all_chunks]

        # Generate embeddings in a single batch (or use batching internally)
        embeddings = get_embeddings(
            chunk_texts,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=True  # Required for cosine similarity with FAISS
        )

        logger.info(f"Generated {len(embeddings)} embeddings")

        # Step 4: Build FAISS index
        logger.info("Building FAISS index...")

        # Get embedding dimension
        embedding_dim = get_embedding_dimension()
        logger.debug(f"Embedding dimension: {embedding_dim}")

        # Create FAISS index (using IndexFlatL2 for now)
        # For cosine similarity with normalized embeddings, L2 distance is equivalent
        index = faiss.IndexFlatL2(embedding_dim)

        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Add embeddings to index
        index.add(embeddings_array)

        logger.info(f"FAISS index built with {index.ntotal} vectors")

        # Step 5: Save index and metadata
        logger.info("Saving index and metadata...")
        save_index(index, all_chunks, index_path)

        end_time = time.time()
        time_taken = end_time - start_time

        logger.info(
            f"Index build complete! "
            f"{total_files} files, {total_chunks} chunks, {failed_files} failures, "
            f"time: {time_taken:.2f}s"
        )

        return index, all_chunks

    except IndexBuildError:
        # Re-raise IndexBuildError
        raise
    except Exception as e:
        logger.exception("Unexpected error during index building")
        raise IndexBuildError(f"Index build failed: {str(e)}") from e


def save_index(
    index: faiss.Index,
    chunks: List[CodeChunk],
    index_path: Path
) -> None:
    """Save FAISS index and chunk metadata to disk.

    Args:
        index: FAISS index to save
        chunks: List of chunks corresponding to index vectors
        index_path: Directory path to save index

    Raises:
        IndexSaveError: If saving fails
    """
    try:
        # Create index directory if it doesn't exist
        index_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = index_path / "index.faiss"
        faiss.write_index(index, str(index_file))
        logger.info(f"Saved FAISS index to {index_file}")

        # Save chunk metadata
        metadata_file = index_path / "chunks_metadata.json"

        # Convert chunks to JSON-serializable format
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

        logger.info(f"Saved {len(chunks)} chunk metadata to {metadata_file}")

        # Save index info
        info_file = index_path / "index_info.json"
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

        logger.info(f"Saved index info to {info_file}")

    except Exception as e:
        logger.exception("Failed to save index")
        raise IndexSaveError(f"Failed to save index: {str(e)}") from e


def load_index(index_path: Optional[Path] = None) -> Tuple[faiss.Index, List[dict]]:
    """Load FAISS index and chunk metadata from disk.

    Args:
        index_path: Directory path where index is stored (default: from settings)

    Returns:
        Tuple[faiss.Index, List[dict]]: FAISS index and chunk metadata

    Raises:
        FileNotFoundError: If index files don't exist
        IndexError: If loading fails
    """
    if index_path is None:
        index_path = Path(settings.FAISS_INDEX_PATH).expanduser()

    try:
        # Load FAISS index
        index_file = index_path / "index.faiss"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")

        index = faiss.read_index(str(index_file))
        logger.info(f"Loaded FAISS index from {index_file} ({index.ntotal} vectors)")

        # Load chunk metadata
        metadata_file = index_path / "chunks_metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'r', encoding='utf-8') as f:
            chunks_metadata = json.load(f)

        logger.info(f"Loaded {len(chunks_metadata)} chunk metadata from {metadata_file}")

        return index, chunks_metadata

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.exception("Failed to load index")
        raise IndexError(f"Failed to load index: {str(e)}") from e


def index_exists(index_path: Optional[Path] = None) -> bool:
    """Check if index exists at the specified path.

    Args:
        index_path: Directory path to check (default: from settings)

    Returns:
        bool: True if index exists, False otherwise
    """
    if index_path is None:
        index_path = Path(settings.FAISS_INDEX_PATH).expanduser()

    index_file = index_path / "index.faiss"
    metadata_file = index_path / "chunks_metadata.json"

    return index_file.exists() and metadata_file.exists()
