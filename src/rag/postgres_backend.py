"""PostgreSQL + pgvector storage backend for RAG system.

This module implements the StorageBackend interface using PostgreSQL with the
pgvector extension for efficient vector similarity search. It provides production-grade
persistence, transaction safety, and concurrent access support.

Features:
- Connection pooling for multi-threaded applications
- Bulk insert operations for high throughput
- HNSW indexes for fast approximate nearest neighbor search
- Transactional integrity (ACID guarantees)
- Concurrent crawling and indexing support

Prerequisites:
- PostgreSQL 12+ with pgvector extension installed
- Database initialized with scripts/init_postgres_schema.sql

Typical usage:
    backend = PostgresBackend(
        database_url="postgresql://user:pass@localhost:5432/ai_agent_rag",
        pool_size=10
    )
    backend.build_index(chunks, embeddings)
    results = backend.search(query_embedding, top_k=5)
"""

import logging
import json
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2.pool import ThreadedConnectionPool
    from psycopg2.extras import execute_batch, RealDictCursor
    from psycopg2 import sql
except ImportError:
    raise ImportError(
        "PostgreSQL backend requires psycopg2-binary. "
        "Install with: pip install psycopg2-binary"
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
from ..config.settings import settings

logger = logging.getLogger("ai_agent.postgres_backend")


class PostgresBackend(StorageBackend):
    """PostgreSQL + pgvector storage backend implementation.

    This backend uses PostgreSQL with the pgvector extension to store code chunks
    and their vector embeddings. It provides production-ready features including:
    - Connection pooling for efficient resource usage
    - Bulk operations for high throughput indexing
    - Vector similarity search using HNSW or IVFFlat indexes
    - Transaction support for data consistency
    - Concurrent access from multiple processes/threads

    Args:
        database_url: PostgreSQL connection string
                     Format: postgresql://user:password@host:port/dbname
        pool_size: Maximum number of connections in the pool (default: 10)
        connection_timeout: Timeout for acquiring connections in seconds (default: 30)
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        connection_timeout: int = 30
    ):
        """Initialize PostgreSQL backend with connection pooling."""
        self.database_url = database_url
        self.pool_size = pool_size
        self.connection_timeout = connection_timeout

        # Create connection pool
        try:
            self.pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=pool_size,
                dsn=database_url,
                connect_timeout=connection_timeout
            )
            logger.info(f"PostgreSQL connection pool created (size: {pool_size})")
        except psycopg2.Error as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise StorageBackendError(
                f"Cannot connect to PostgreSQL: {e}\n"
                f"Check DATABASE_URL and ensure PostgreSQL is running"
            ) from e

        # Ensure schema is initialized
        self._ensure_schema()

    @contextmanager
    def _get_connection(self):
        """Get a connection from the pool (context manager).

        Usage:
            with backend._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT ...")
        """
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    def _ensure_schema(self):
        """Ensure database schema exists and is initialized.

        Creates tables if they don't exist. Uses the SQL schema from
        scripts/init_postgres_schema.sql for initial setup.
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Check if code_chunks table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'code_chunks'
                    );
                """)
                table_exists = cur.fetchone()[0]

                if not table_exists:
                    logger.warning(
                        "Database tables not found. Please initialize with: "
                        "psql -f scripts/init_postgres_schema.sql"
                    )
                    # For now, we'll just warn. In production, might want to auto-create
                    # or raise an error
                else:
                    logger.debug("Database schema verified")

    def index_exists(self) -> bool:
        """Check if index exists (has any data).

        Returns:
            True if code_chunks table has data, False otherwise
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT EXISTS(SELECT 1 FROM code_chunks LIMIT 1);")
                    return cur.fetchone()[0]
        except psycopg2.Error as e:
            logger.error(f"Failed to check index existence: {e}")
            return False

    def build_index(
        self,
        chunks: List[CodeChunk],
        embeddings: List[List[float]],
        force_reindex: bool = False
    ) -> int:
        """Build or rebuild index from chunks and embeddings.

        Uses bulk insert (execute_batch) for maximum throughput. All operations
        are done in a single transaction for data consistency.

        Args:
            chunks: List of CodeChunk objects
            embeddings: List of embedding vectors (384-dim floats)
            force_reindex: If True, truncate existing data before inserting

        Returns:
            Number of chunks successfully indexed

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

        logger.info(f"Building index with {len(chunks)} chunks (force_reindex={force_reindex})")

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Truncate if force reindexing
                    if force_reindex:
                        logger.info("Force reindex: truncating existing data")
                        cur.execute("TRUNCATE TABLE code_chunks RESTART IDENTITY CASCADE;")

                    # Prepare data for bulk insert
                    data = [
                        (
                            chunk.content,
                            str(chunk.file_path),
                            chunk.start_line,
                            chunk.end_line,
                            chunk.chunk_type,
                            chunk.language,
                            json.dumps(chunk.metadata or {}),
                            embeddings[i]  # pgvector handles list conversion
                        )
                        for i, chunk in enumerate(chunks)
                    ]

                    # Bulk insert using execute_batch (fast)
                    execute_batch(cur, """
                        INSERT INTO code_chunks
                        (content, file_path, start_line, end_line, chunk_type,
                         language, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, data, page_size=100)

                    # Commit transaction
                    conn.commit()

                    logger.info(f"Successfully indexed {len(chunks)} chunks")
                    return len(chunks)

        except psycopg2.Error as e:
            logger.error(f"Failed to build index: {e}")
            raise IndexBuildError(f"PostgreSQL indexing failed: {e}") from e

    def add_chunks(
        self,
        chunks: List[CodeChunk],
        embeddings: List[List[float]]
    ) -> int:
        """Add new chunks to existing index (incremental indexing).

        This is the same as build_index with force_reindex=False.

        Args:
            chunks: List of CodeChunk objects to add
            embeddings: List of embedding vectors

        Returns:
            Number of chunks added
        """
        return self.build_index(chunks, embeddings, force_reindex=False)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        **kwargs
    ) -> List[SearchResult]:
        """Perform semantic similarity search using pgvector.

        Uses the <-> operator for L2 distance (equivalent to cosine similarity
        with normalized vectors). Results are filtered by similarity threshold
        and sorted by relevance.

        Args:
            query_embedding: Query vector (384-dim floats)
            top_k: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0.0-1.0)
            **kwargs: Additional filters (e.g., file_path, chunk_type, language)

        Returns:
            List of SearchResult objects, sorted by similarity (highest first)

        Raises:
            IndexNotFoundError: If no data in index
            SearchError: If search fails
        """
        if not self.index_exists():
            raise IndexNotFoundError(
                "No index data found. Build index first with build_index()"
            )

        logger.debug(f"Searching for top {top_k} results (threshold: {similarity_threshold})")

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Build WHERE clause for optional filters
                    where_conditions = []
                    params = [query_embedding, query_embedding, similarity_threshold,
                             query_embedding, top_k]

                    # Optional filters from kwargs
                    if 'file_path' in kwargs:
                        where_conditions.append("file_path = %s")
                        params.insert(-2, kwargs['file_path'])
                    if 'chunk_type' in kwargs:
                        where_conditions.append("chunk_type = %s")
                        params.insert(-2, kwargs['chunk_type'])
                    if 'language' in kwargs:
                        where_conditions.append("language = %s")
                        params.insert(-2, kwargs['language'])

                    where_clause = " AND ".join(where_conditions) if where_conditions else "TRUE"

                    # Vector similarity search using <-> (L2 distance)
                    query = f"""
                        SELECT
                            file_path,
                            start_line,
                            end_line,
                            content,
                            chunk_type,
                            language,
                            metadata,
                            (1.0 / (1.0 + (embedding <-> %s::vector))) AS similarity
                        FROM code_chunks
                        WHERE
                            (1.0 / (1.0 + (embedding <-> %s::vector))) >= %s
                            AND {where_clause}
                        ORDER BY embedding <-> %s::vector
                        LIMIT %s
                    """

                    cur.execute(query, params)
                    rows = cur.fetchall()

                    # Convert to SearchResult objects
                    results = []
                    for row in rows:
                        results.append(SearchResult(
                            file_path=row['file_path'],
                            start_line=row['start_line'],
                            end_line=row['end_line'],
                            content=row['content'],
                            score=float(row['similarity']),
                            chunk_type=row['chunk_type'],
                            language=row['language'],
                            metadata=row['metadata']
                        ))

                    logger.debug(f"Found {len(results)} results")
                    return results

        except psycopg2.Error as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"PostgreSQL search failed: {e}") from e

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index.

        Returns:
            Dictionary with index statistics including:
                - total_chunks: Number of indexed chunks
                - total_vectors: Number of vectors (same as chunks)
                - unique_files: Number of unique source files
                - unique_languages: Number of programming languages
                - backend: "postgresql"
                - index_size_mb: Approximate database size in MB
                - table_size_mb: Table size in MB
                - index_physical_size_mb: Index size in MB
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get counts
                    cur.execute("""
                        SELECT
                            COUNT(*) AS total_chunks,
                            COUNT(DISTINCT file_path) AS unique_files,
                            COUNT(DISTINCT language) AS unique_languages,
                            pg_size_pretty(pg_total_relation_size('code_chunks')) AS total_size,
                            pg_total_relation_size('code_chunks') / (1024.0 * 1024.0) AS total_size_mb,
                            pg_relation_size('code_chunks') / (1024.0 * 1024.0) AS table_size_mb,
                            pg_indexes_size('code_chunks') / (1024.0 * 1024.0) AS index_size_mb
                        FROM code_chunks;
                    """)

                    row = cur.fetchone()

                    return {
                        "total_chunks": int(row['total_chunks']),
                        "total_vectors": int(row['total_chunks']),
                        "unique_files": int(row['unique_files']),
                        "unique_languages": int(row['unique_languages']),
                        "backend": "postgresql",
                        "index_size_mb": round(float(row['total_size_mb']), 2),
                        "table_size_mb": round(float(row['table_size_mb']), 2),
                        "index_physical_size_mb": round(float(row['index_size_mb']), 2),
                        "total_size_pretty": row['total_size']
                    }

        except psycopg2.Error as e:
            logger.error(f"Failed to get stats: {e}")
            raise IndexNotFoundError(f"Cannot get stats: {e}") from e

    def clear(self) -> None:
        """Clear all data from the index (TRUNCATE tables).

        Warning:
            This is a destructive operation that cannot be undone.
            All chunks, embeddings, and metadata will be permanently deleted.
        """
        logger.warning("Clearing all index data (TRUNCATE)")

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        TRUNCATE TABLE code_chunks, crawled_urls, crawl_sessions
                        RESTART IDENTITY CASCADE;
                    """)
                    conn.commit()
                    logger.info("Index cleared successfully")

        except psycopg2.Error as e:
            logger.error(f"Failed to clear index: {e}")
            raise StorageBackendError(f"Failed to clear index: {e}") from e

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks from a specific file.

        Args:
            file_path: Path to the file whose chunks should be deleted

        Returns:
            Number of chunks deleted
        """
        logger.info(f"Deleting chunks for file: {file_path}")

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM code_chunks WHERE file_path = %s",
                        (file_path,)
                    )
                    deleted = cur.rowcount
                    conn.commit()

                    logger.info(f"Deleted {deleted} chunks from {file_path}")
                    return deleted

        except psycopg2.Error as e:
            logger.error(f"Failed to delete chunks: {e}")
            raise StorageBackendError(f"Failed to delete chunks: {e}") from e

    def close(self):
        """Close all connections in the pool.

        Call this when shutting down the application to cleanly close
        all database connections.
        """
        if self.pool:
            self.pool.closeall()
            logger.info("PostgreSQL connection pool closed")

    def __del__(self):
        """Destructor: close connections on garbage collection."""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup
