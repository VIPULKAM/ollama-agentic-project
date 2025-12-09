"""Configuration settings for AI Coding Agent."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Literal
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with cloud migration support."""

    # LLM Provider Selection
    # Options: "ollama", "claude", "gemini", "hybrid"
    LLM_PROVIDER: Literal["ollama", "claude", "gemini", "hybrid"] = "ollama"

    # Ollama Configuration
    # Local: http://localhost:11434
    # Cloud: https://ai-server.yourcompany.com
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "qwen2.5-coder:1.5b"

    # Claude/Anthropic Configuration
    ANTHROPIC_API_KEY: Optional[str] = None
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"  # Latest Claude 3.5 Sonnet with tool calling

    # Gemini/Google Configuration
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"

    # Model Parameters
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 8192  # Increased for Gemini 2.5 Flash (supports up to 8192 output tokens)

    # Memory Configuration
    MAX_HISTORY_LENGTH: int = 10

    # CLI Configuration
    CLI_THEME: str = "monokai"
    STREAM_OUTPUT: bool = True

    # Hybrid Mode Settings
    # If LLM_PROVIDER="hybrid", use these keywords to route to Claude
    CLAUDE_KEYWORDS: list = [
        "architecture", "design pattern", "refactor", "optimize",
        "security", "best practice", "review", "compare"
    ]

    # Tool Configuration (NEW)
    ENABLE_TOOLS: bool = True  # Feature flag - start disabled for gradual rollout
    ENABLE_FILE_OPS: bool = True
    ENABLE_RAG: bool = True
    ENABLE_GIT_TOOLS: bool = True  # Enable Git integration tools
    ENABLE_PLANNING: bool = False  # Explicitly disable planning due to missing modules

    # Agent Configuration
    AGENT_RECURSION_LIMIT: int = 50  # LangGraph recursion limit (default: 25)

    # RAG Configuration (NEW)
    VECTOR_DB: str = "faiss"  # Using FAISS instead of ChromaDB (better Python 3.13 support)
    FAISS_INDEX_PATH: str = str(Path.home() / ".ai-agent" / "faiss_index")
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    RAG_TOP_K: int = 5
    RAG_CHUNK_SIZE: int = 500
    RAG_CHUNK_OVERLAP: int = 50
    INDEX_SCHEMA_VERSION: int = 1  # Increment to force reindex

    # PostgreSQL + pgvector Configuration (NEW)
    ENABLE_POSTGRES_STORAGE: bool = False  # Default to FAISS for backward compatibility
    DATABASE_URL: Optional[str] = None  # Format: postgresql://user:password@host:port/dbname
    DB_POOL_SIZE: int = 10  # Connection pool size
    DB_POOL_MAX_OVERFLOW: int = 20  # Max overflow connections
    DB_CONNECTION_TIMEOUT: int = 30  # Connection timeout in seconds
    DB_ECHO: bool = False  # Echo SQL queries (for debugging)

    # pgvector Index Configuration
    PGVECTOR_INDEX_TYPE: str = "hnsw"  # "hnsw" or "ivfflat"
    PGVECTOR_HNSW_M: int = 16  # HNSW index parameter (links per node)
    PGVECTOR_HNSW_EF_CONSTRUCTION: int = 64  # HNSW construction parameter
    PGVECTOR_IVFFLAT_LISTS: int = 100  # IVFFlat lists parameter

    # Batch Crawling Configuration (for PostgreSQL backend)
    BATCH_CRAWL_MAX_CONCURRENT: int = 5  # Max concurrent crawl operations
    BATCH_CRAWL_SKIP_DUPLICATES: bool = True  # Skip URLs with unchanged content
    BATCH_CRAWL_TRANSACTION_SIZE: int = 100  # Commit every N chunks

    # File Operations Configuration (NEW)
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: list = [".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml", ".txt"]
    FILE_BACKUP_ENABLED: bool = True
    FILE_BACKUP_SUFFIX: str = ".bak"
    MAX_FILE_OPS_PER_MINUTE: int = 50  # Rate limiting

    # Indexing Configuration (NEW)
    AUTO_INDEX_ON_START: bool = False
    INDEX_BATCH_SIZE: int = 32
    INDEX_EXCLUDE_PATTERNS: list = [
        ".venv/*", "venv/*", "node_modules/*",
        "__pycache__/*", "*.pyc", ".git/*",
        "*.egg-info/*", "dist/*", "build/*"
    ]
    INDEXER_EXCLUDE_PATTERNS: list = []  # Additional user-defined exclusions
    USE_GITIGNORE: bool = True  # Respect .gitignore patterns

    # Allowed file extensions for indexing
    ALLOWED_FILE_EXTENSIONS: list = [
        ".py", ".js", ".ts", ".tsx", ".jsx",
        ".md", ".txt", ".json", ".yaml", ".yml",
        ".java", ".go", ".rs", ".cpp", ".c", ".h"
    ]

    # Chunking Configuration (NEW)
    CHUNK_SIZE: int = 500  # Characters per chunk (for text chunking fallback)
    CHUNK_OVERLAP: int = 50  # Overlap between chunks
    CHUNK_MIN_SIZE: int = 50  # Minimum chunk size to avoid tiny chunks

    # Logging Configuration (NEW)
    LOG_FILE_OPERATIONS: bool = True
    LOG_LEVEL: str = "INFO"

    # Web Crawler Configuration (NEW - CrawlAI)
    ENABLE_WEB_CRAWLING: bool = True
    CRAWLER_HEADLESS: bool = True  # Run browser in headless mode
    CRAWLER_VERBOSE: bool = False
    CRAWLED_DOCS_PATH: str = str(Path.home() / ".ai-agent" / "crawled_docs")
    CRAWLER_USER_AGENT: str = "Mozilla/5.0 (compatible; AI-Agent/1.0; +https://github.com/yourusername/ai-agent)"

    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Ignore extra fields in .env
    )


# Global settings instance
settings = Settings()
