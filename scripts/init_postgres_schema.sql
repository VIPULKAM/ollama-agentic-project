-- PostgreSQL + pgvector Schema for AI Agent RAG System
-- This script creates all necessary tables, indexes, and functions for
-- storing code chunks with vector embeddings.
--
-- Prerequisites:
--   - PostgreSQL 12+ installed
--   - pgvector extension available
--
-- Usage:
--   psql -U ai_agent -d ai_agent_rag -f scripts/init_postgres_schema.sql
--
-- Or from Python:
--   cursor.execute(open('scripts/init_postgres_schema.sql').read())

-- Enable pgvector extension (requires superuser or rds_superuser role)
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation for crawl sessions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Core Tables
-- =============================================================================

-- Main table for storing code chunks with vector embeddings
CREATE TABLE IF NOT EXISTS code_chunks (
    id BIGSERIAL PRIMARY KEY,

    -- Chunk content and location
    content TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER NOT NULL CHECK (start_line > 0),
    end_line INTEGER NOT NULL CHECK (end_line >= start_line),

    -- Chunk metadata
    chunk_type VARCHAR(50) NOT NULL DEFAULT 'text',
    language VARCHAR(50) NOT NULL DEFAULT 'unknown',
    metadata JSONB DEFAULT '{}',

    -- Vector embedding (384 dimensions for sentence-transformers/all-MiniLM-L6-v2)
    embedding vector(384) NOT NULL,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Comment on table
COMMENT ON TABLE code_chunks IS 'Stores code/text chunks with vector embeddings for semantic search';
COMMENT ON COLUMN code_chunks.embedding IS '384-dimensional vector from sentence-transformers/all-MiniLM-L6-v2';
COMMENT ON COLUMN code_chunks.metadata IS 'JSONB field for flexible metadata (function_name, class_name, etc.)';


-- URL tracking table (replaces crawl_tracker.json)
CREATE TABLE IF NOT EXISTS crawled_urls (
    id BIGSERIAL PRIMARY KEY,

    -- URL and content tracking
    url TEXT UNIQUE NOT NULL,
    content_hash VARCHAR(32) NOT NULL,  -- MD5 hash for change detection

    -- Crawl metadata
    chunk_count INTEGER NOT NULL DEFAULT 0,
    file_path TEXT NOT NULL,
    title TEXT DEFAULT '',
    content_length INTEGER DEFAULT 0,

    -- Timestamps
    crawl_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE crawled_urls IS 'Tracks crawled URLs with content hashes for deduplication';
COMMENT ON COLUMN crawled_urls.content_hash IS 'MD5 hash of content for detecting changes';


-- Index metadata table (replaces index_info.json)
CREATE TABLE IF NOT EXISTS index_metadata (
    id SERIAL PRIMARY KEY,

    -- Index configuration
    index_type VARCHAR(50) NOT NULL DEFAULT 'pgvector',
    embedding_model TEXT NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    embedding_dimension INTEGER NOT NULL DEFAULT 384,

    -- Statistics
    total_vectors INTEGER NOT NULL DEFAULT 0,
    total_chunks INTEGER NOT NULL DEFAULT 0,
    schema_version INTEGER NOT NULL DEFAULT 1,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

COMMENT ON TABLE index_metadata IS 'Stores index configuration and statistics';


-- Batch crawl sessions table (for transaction tracking)
CREATE TABLE IF NOT EXISTS crawl_sessions (
    id BIGSERIAL PRIMARY KEY,

    -- Session identification
    session_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),

    -- Session status
    status VARCHAR(20) NOT NULL DEFAULT 'in_progress'
        CHECK (status IN ('in_progress', 'completed', 'failed')),

    -- Statistics
    urls_crawled INTEGER DEFAULT 0,
    chunks_indexed INTEGER DEFAULT 0,

    -- Timestamps
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,

    -- Error tracking
    error_message TEXT
);

COMMENT ON TABLE crawl_sessions IS 'Tracks batch crawling sessions for monitoring and debugging';


-- =============================================================================
-- Indexes for Performance
-- =============================================================================

-- Vector similarity search index (HNSW for fast approximate nearest neighbor search)
-- HNSW (Hierarchical Navigable Small World) provides:
--   - Sub-linear search time
--   - Good recall even with large datasets
--   - Configurable trade-offs (m, ef_construction)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON code_chunks
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

COMMENT ON INDEX idx_chunks_embedding_hnsw IS 'HNSW index for fast vector similarity search using L2 distance';

-- Alternative: IVFFlat index (less memory, slightly slower)
-- Uncomment if you prefer IVFFlat over HNSW:
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding_ivfflat ON code_chunks
-- USING ivfflat (embedding vector_l2_ops)
-- WITH (lists = 100);

-- B-tree indexes for metadata lookups
CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON code_chunks (file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON code_chunks (chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_language ON code_chunks (language);
CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON code_chunks (created_at DESC);

-- Full-text search index on content (optional, for hybrid search)
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts ON code_chunks
USING gin(to_tsvector('english', content));

COMMENT ON INDEX idx_chunks_content_fts IS 'Full-text search index for keyword-based search (hybrid with vector search)';

-- Indexes for crawled_urls table
CREATE INDEX IF NOT EXISTS idx_crawled_urls_url ON crawled_urls (url);
CREATE INDEX IF NOT EXISTS idx_crawled_urls_content_hash ON crawled_urls (content_hash);
CREATE INDEX IF NOT EXISTS idx_crawled_urls_crawl_date ON crawled_urls (crawl_date DESC);

-- Indexes for crawl_sessions table
CREATE INDEX IF NOT EXISTS idx_crawl_sessions_status ON crawl_sessions (status);
CREATE INDEX IF NOT EXISTS idx_crawl_sessions_started ON crawl_sessions (started_at DESC);


-- =============================================================================
-- Functions and Triggers
-- =============================================================================

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_updated_at_column() IS 'Automatically updates the updated_at column on row modifications';

-- Triggers for updated_at columns
CREATE TRIGGER update_code_chunks_updated_at
    BEFORE UPDATE ON code_chunks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_crawled_urls_updated_at
    BEFORE UPDATE ON crawled_urls
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_index_metadata_updated_at
    BEFORE UPDATE ON index_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();


-- Function to update index metadata statistics
CREATE OR REPLACE FUNCTION update_index_stats()
RETURNS VOID AS $$
DECLARE
    chunk_count INTEGER;
    vector_count INTEGER;
BEGIN
    -- Count chunks and vectors
    SELECT COUNT(*) INTO chunk_count FROM code_chunks;
    vector_count := chunk_count;  -- One vector per chunk

    -- Update or insert index metadata
    INSERT INTO index_metadata (total_vectors, total_chunks)
    VALUES (vector_count, chunk_count)
    ON CONFLICT (id) DO UPDATE
    SET total_vectors = EXCLUDED.total_vectors,
        total_chunks = EXCLUDED.total_chunks,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_index_stats() IS 'Updates index metadata with current counts (call after bulk operations)';


-- =============================================================================
-- Views for Convenience
-- =============================================================================

-- View for quick statistics
CREATE OR REPLACE VIEW v_index_statistics AS
SELECT
    COUNT(*) AS total_chunks,
    COUNT(DISTINCT file_path) AS unique_files,
    COUNT(DISTINCT language) AS unique_languages,
    pg_size_pretty(pg_total_relation_size('code_chunks')) AS table_size,
    pg_size_pretty(pg_indexes_size('code_chunks')) AS index_size,
    MAX(created_at) AS last_indexed
FROM code_chunks;

COMMENT ON VIEW v_index_statistics IS 'Quick overview of index statistics';


-- View for crawl statistics
CREATE OR REPLACE VIEW v_crawl_statistics AS
SELECT
    COUNT(*) AS total_urls_crawled,
    SUM(chunk_count) AS total_chunks_from_crawls,
    AVG(chunk_count) AS avg_chunks_per_url,
    MAX(crawl_date) AS last_crawl_date
FROM crawled_urls;

COMMENT ON VIEW v_crawl_statistics IS 'Statistics about crawled URLs';


-- =============================================================================
-- Helper Functions for Common Operations
-- =============================================================================

-- Function to search for similar chunks (wrapper around vector search)
CREATE OR REPLACE FUNCTION search_similar_chunks(
    query_embedding vector(384),
    match_count INTEGER DEFAULT 5,
    similarity_threshold FLOAT DEFAULT 0.5
)
RETURNS TABLE (
    id BIGINT,
    file_path TEXT,
    start_line INTEGER,
    end_line INTEGER,
    content TEXT,
    chunk_type VARCHAR(50),
    language VARCHAR(50),
    metadata JSONB,
    similarity_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.file_path,
        c.start_line,
        c.end_line,
        c.content,
        c.chunk_type,
        c.language,
        c.metadata,
        (1.0 / (1.0 + (c.embedding <-> query_embedding))) AS similarity_score
    FROM code_chunks c
    WHERE (1.0 / (1.0 + (c.embedding <-> query_embedding))) >= similarity_threshold
    ORDER BY c.embedding <-> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION search_similar_chunks IS 'Search for semantically similar code chunks';


-- =============================================================================
-- Permissions (Optional - adjust based on your setup)
-- =============================================================================

-- Grant permissions to ai_agent user (if it exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_agent') THEN
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ai_agent;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO ai_agent;
        GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO ai_agent;
    END IF;
END $$;


-- =============================================================================
-- Initial Data (Optional)
-- =============================================================================

-- Insert initial index metadata record
INSERT INTO index_metadata (
    index_type,
    embedding_model,
    embedding_dimension,
    schema_version
) VALUES (
    'pgvector',
    'sentence-transformers/all-MiniLM-L6-v2',
    384,
    1
) ON CONFLICT DO NOTHING;


-- =============================================================================
-- Cleanup Functions (Use with caution!)
-- =============================================================================

-- Function to truncate all tables (for testing/development)
CREATE OR REPLACE FUNCTION truncate_all_tables()
RETURNS VOID AS $$
BEGIN
    TRUNCATE TABLE code_chunks, crawled_urls, crawl_sessions RESTART IDENTITY CASCADE;
    UPDATE index_metadata SET total_vectors = 0, total_chunks = 0;
    RAISE NOTICE 'All tables truncated successfully';
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION truncate_all_tables() IS 'DANGER: Truncates all data tables (use only for testing)';


-- =============================================================================
-- Schema Version Check
-- =============================================================================

-- Display schema version and statistics
DO $$
DECLARE
    version_info RECORD;
BEGIN
    SELECT * INTO version_info FROM index_metadata WHERE id = 1;

    RAISE NOTICE '==========================================================';
    RAISE NOTICE 'AI Agent RAG Database Schema Initialized';
    RAISE NOTICE '==========================================================';
    RAISE NOTICE 'Schema Version: %', COALESCE(version_info.schema_version, 1);
    RAISE NOTICE 'Embedding Model: %', COALESCE(version_info.embedding_model, 'sentence-transformers/all-MiniLM-L6-v2');
    RAISE NOTICE 'Vector Dimension: %', COALESCE(version_info.embedding_dimension, 384);
    RAISE NOTICE '==========================================================';
END $$;
