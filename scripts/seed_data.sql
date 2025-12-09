-- Seed data for AI Agent PostgreSQL database
-- This file is automatically loaded when initializing the database

-- Note: This is optional sample data for development/testing
-- Comment out or remove sections you don't need

-- Insert sample index metadata
INSERT INTO index_metadata (
    total_chunks,
    total_files,
    total_size_bytes,
    embedding_model,
    last_updated
) VALUES (
    0,
    0,
    0,
    'sentence-transformers/all-MiniLM-L6-v2',
    NOW()
) ON CONFLICT (id) DO NOTHING;

-- Log the initialization
DO $$
BEGIN
    RAISE NOTICE '✓ Database initialized successfully';
    RAISE NOTICE '✓ pgvector extension enabled';
    RAISE NOTICE '✓ Tables created: code_chunks, crawled_urls, index_metadata, crawl_sessions';
    RAISE NOTICE '✓ HNSW indexes created for vector similarity search';
    RAISE NOTICE '✓ Helper functions and views available';
    RAISE NOTICE '';
    RAISE NOTICE 'Database ready for use!';
    RAISE NOTICE 'Connection string: postgresql://ai_agent_user:dev_password_change_in_production@localhost:5432/ai_agent';
END $$;
