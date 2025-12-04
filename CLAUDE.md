# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Coding Agent - A LangChain-based coding assistant that supports multiple LLM providers (Ollama for self-hosted models, Claude API, or hybrid mode). Designed for local development with easy cloud migration to AWS.

**Key Concept**: This is a CLI tool that helps developers with coding questions, specialized in Python, TypeScript, and various databases (PostgreSQL, MySQL, MongoDB, Snowflake, ClickHouse, etc.).

**Current Development Phase**: ✅ Tools & RAG complete (195 tests passing). LangGraph integration **verified working** - requires API key with credits to test tool execution (see Known Issues #8).

## Architecture

### Multi-Provider LLM System

The agent uses LangChain's provider abstraction to support three modes:

1. **Ollama mode** (`LLM_PROVIDER=ollama`): Self-hosted models (local or cloud-deployed)
2. **Claude mode** (`LLM_PROVIDER=claude`): Anthropic API (Haiku model for cost-effectiveness)
3. **Hybrid mode** (`LLM_PROVIDER=hybrid`): Smart routing between providers based on query keywords

**Current Model**: `qwen2.5-coder:1.5b` - Chosen due to local machine limitations. **Note**: This model does NOT support native tool calling required by LangGraph. Tool support requires larger models with function calling capabilities (see Configuration section).

**Smart Routing Logic** (src/agent/agent.py:115-125): In hybrid mode, queries containing keywords like "architecture", "design pattern", "refactor", "optimize", "security", "best practice", "review", "compare" are automatically routed to Claude for better quality. Other queries use Ollama.

### Core Components

- **CodingAgent** (src/agent/agent.py): Main agent class that wraps LangChain's RunnableWithMessageHistory
- **Prompt System** (src/agent/prompts.py): System prompt with database expertise
- **CLI Interface** (src/cli/main.py): Rich terminal UI with markdown rendering
- **Settings** (src/config/settings.py): Pydantic-based configuration with .env support
- **Tools System** (src/agent/tools/): 5 tools ready for agent integration ✅
  - File operations: ReadFileTool, WriteFileTool, ListDirectoryTool, SearchCodeTool
  - RAG search: RagSearchTool for semantic codebase search
- **RAG System** (src/rag/): Complete FAISS-based semantic search pipeline ✅
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2 with singleton caching
  - Indexing: File discovery with gitignore, AST-based chunking, FAISS IndexFlatL2
  - Retrieval: Semantic search with similarity threshold filtering and ranking

### Current Architecture Evolution

**Phase 1 (Complete) - Simple Chain**:
```python
chain = ChatPromptTemplate | LLM
chain_with_history = RunnableWithMessageHistory(chain, get_session_history, ...)
```

**Phase 2 (Complete) - ReAct Agent**:
```python
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent, tools, memory=memory)
```

**Phase 3 (Current) - LangGraph Agent**:
```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

agent = create_react_agent(llm, tools, checkpointer=MemorySaver())
```

The agent has evolved through three phases:
1. **Simple ConversationChain**: Basic Q&A with memory
2. **ReAct Agent**: Added tool use with reasoning
3. **LangGraph Agent**: Advanced workflow orchestration with state management

**Current Capabilities**:
- Reading files in current directory
- Writing/modifying files with backups and diff preview
- Listing directory contents
- Searching code with regex
- Semantic codebase search using RAG
- Multi-step reasoning with tool composition
- State persistence across sessions (via LangGraph checkpointing)

### LangChain Integration

Conversation history is stored in memory using `InMemoryChatMessageHistory` keyed by session ID.

### Provider Switching

When using hybrid mode, the agent rebuilds the chain with the selected LLM on each request (src/agent/agent.py:165-172). This allows dynamic provider selection while maintaining conversation history.

## Development Commands

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development (includes testing/linting tools)
pip install -r requirements-dev.txt

# Copy environment template
cp .env.example .env
# Edit .env to configure providers
```

### Running the Application

```bash
# Run the CLI
python main.py

# Or directly
python -m src.cli.main
```

### Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_agent.py

# Run specific test suites
pytest tests/test_tools/           # All file operation tools tests
pytest tests/test_rag/             # All RAG system tests

# Run tests by category
pytest tests/test_tools/test_read_file.py          # Read file tool
pytest tests/test_tools/test_write_file.py         # Write file tool
pytest tests/test_tools/test_list_directory.py     # List directory tool
pytest tests/test_tools/test_search_code.py        # Search code tool
pytest tests/test_tools/test_file_ops_security.py  # Security tests
pytest tests/test_rag/test_embeddings.py           # Embeddings tests
pytest tests/test_rag/test_indexer_integration.py  # Indexer tests
pytest tests/test_rag/test_retriever.py            # Retriever tests

# Run with coverage report
pytest --cov=src --cov-report=html

# Run agent validation tests (includes Ollama pre-flight checks)
pytest tests/test_agent.py -v
```

### Development Tools

```bash
# Code formatting
black src/

# Linting
ruff check src/

# Type checking
mypy src/

# Check Python version
python --version  # Should be 3.10+

# Verify virtual environment
which python  # Should point to venv/bin/python
```

### Ollama Setup (for local development)

```bash
# Install Ollama from https://ollama.ai

# Start Ollama service
ollama serve

# Pull the qwen2.5-coder model (current default)
ollama pull qwen2.5-coder:1.5b

# Check available models
ollama list

# Test model locally
ollama run qwen2.5-coder:1.5b "Write a hello world in Python"
```

### Debugging and Troubleshooting

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Test Ollama model directly
ollama run qwen2.5-coder:1.5b "print hello world in python"

# Check installed dependencies
pip list | grep langchain
pip list | grep faiss
pip list | grep sentence-transformers

# Verify environment variables
python -c "from src.config.settings import Settings; s = Settings(); print(f'Provider: {s.LLM_PROVIDER}, Model: {s.MODEL_NAME}')"

# Check FAISS index status
ls -lh ~/.ai-agent/faiss_index/

# View logs (if logging to file)
tail -f ~/.ai-agent/logs/agent.log
```

### RAG System Commands

```bash
# Index current codebase (if implemented in CLI)
python main.py index

# Force reindex
python main.py index --force

# Index specific directory
python main.py index --path /path/to/code

# Check index info
python main.py info --index
```

## Configuration

All configuration is managed through `.env` file (see `.env.example` for template).

**Critical Settings**:
- `LLM_PROVIDER`: Controls which provider(s) to use ("ollama", "claude", "hybrid")
- `MODEL_NAME`: Default is `qwen2.5-coder:1.5b`
- `OLLAMA_BASE_URL`: Change to `https://your-server.com` for cloud deployment
- `ANTHROPIC_API_KEY`: Required for Claude or hybrid mode
- `TEMPERATURE`: Set to 0.1 for focused code generation (default: 0.1)

**New Tool & RAG Settings** (src/config/settings.py:47-51):
- `ENABLE_TOOLS`: Feature flag for tool-based agent (default: True, **requires compatible model**)
- `ENABLE_FILE_OPS`: Enable file operation tools (default: True)
- `ENABLE_RAG`: Enable RAG semantic search (default: True)
- `VECTOR_DB`: "faiss" (using FAISS instead of ChromaDB for Python 3.13 compatibility)
- `FAISS_INDEX_PATH`: Path to FAISS index storage (~/.ai-agent/faiss_index)
- `EMBEDDING_MODEL`: "sentence-transformers/all-MiniLM-L6-v2" (local embeddings)
- `MAX_FILE_SIZE_MB`: File size limit for operations (default: 10MB)
- `FILE_BACKUP_ENABLED`: Automatic backups before writes (default: True)

**Cloud Migration**: To migrate from local to cloud Ollama, only `OLLAMA_BASE_URL` needs to be changed. All code remains identical.

### Model Requirements for Tool Use

**IMPORTANT**: The LangGraph tool-based agent requires models with **native function/tool calling support**. Not all models support this feature!

**✅ Compatible Models (Tool Mode Works)**:
- **Claude**: claude-3-haiku-20240307, claude-3-5-sonnet (recommended for quality)
- **Gemini**: gemini-1.5-flash, gemini-1.5-pro
- **GPT**: gpt-4, gpt-4-turbo, gpt-3.5-turbo
- **Ollama**: Check model card - must explicitly support function calling

**❌ Incompatible Models (Tool Mode Fails)**:
- **qwen2.5-coder:1.5b** (current default) - No tool calling support
- **llama3.2:1b** - Too small, no tool calling
- **codellama:7b** - No native tool calling support
- Most Ollama models < 13B parameters

**Workarounds**:
1. **Use Claude or Gemini for tool mode**: Set `LLM_PROVIDER=claude` and add `ANTHROPIC_API_KEY` to .env
2. **Use hybrid mode**: Set `LLM_PROVIDER=hybrid` to route complex queries to Claude
3. **Disable tools**: Set `ENABLE_TOOLS=False` to use simple conversational mode
4. **Future**: Wait for prompt-based ReAct implementation (doesn't require native tool calling)

**Testing Tool Support**:
```bash
# Test if your model supports tools
python -c "
from src.agent.agent import CodingAgent
agent = CodingAgent(provider='your-provider', temperature=0.1)
response = agent.ask('What is 2+2?')
print('SUCCESS' if '4' in response.content else 'FAILED - check model compatibility')
"
```

## Code Patterns

### Adding New CLI Commands

CLI commands are handled in the main loop (src/cli/main.py:95-137). To add a command:

1. Add command check in the main loop
2. Implement command logic
3. Update help text in `print_welcome()` and `print_help()`

### Extending System Prompts

Database-specific knowledge is in `src/agent/prompts.py`. The `DATABASE_TIPS` dictionary contains specialized knowledge that could be used for RAG in the future.

### Error Handling Pattern

The agent's `ask()` method (src/agent/agent.py:127-187) catches exceptions and provides context-specific error messages:
- Ollama errors suggest checking if `ollama serve` is running
- API errors suggest checking `ANTHROPIC_API_KEY`

### Provider Selection Logic

Force a specific provider for a query:
```python
response = agent.ask("your query", force_provider="claude")
```

This overrides hybrid mode routing.

### Tool Implementation Pattern (In Development)

Tools use Pydantic models for type-safe input validation:
```python
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

class ReadFileInput(BaseModel):
    path: str = Field(description="Path to file to read")

class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read contents of a file"
    args_schema = ReadFileInput

    def _run(self, path: str) -> str:
        # Implementation with path validation
        ...
```

## Important Implementation Details

### Memory Management

- Conversation history is stored per-session in `self.store` dictionary (src/agent/agent.py:91)
- Default session ID is "default" (src/agent/agent.py:92)
- Clear history with `agent.clear_history()` which empties the session's InMemoryChatMessageHistory
- History is NOT persisted; it's lost when the application exits

### Chain Rebuilding in Hybrid Mode

In hybrid mode, the chain is rebuilt on each request to swap the LLM (src/agent/agent.py:166-172). This is necessary because LangChain chains are immutable. The session history store is shared across chain instances, maintaining conversation context.

### Model Info Display

The CLI's `info` command calls `agent.get_model_info()` which returns different fields based on the provider mode. Note: The CLI implementation (src/cli/main.py:36-49) expects specific keys like "model", "base_url", "deployment" which don't match the hybrid mode output structure - this is a bug that should be fixed.

### RAG System Architecture (In Development)

**Vector Database**: FAISS (Facebook AI Similarity Search)
- **Why FAISS over ChromaDB**: Python 3.13 compatibility issues with ChromaDB's chroma-hnswlib dependency
- **Trade-offs**: Manual save/load (1-2 lines of code) vs ChromaDB's automatic persistence
- **Benefits**: Pre-built wheels for Python 3.13, faster search, production-proven, lower memory footprint

**Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- Local execution (no API costs)
- Small (80MB) and fast
- Good semantic similarity for code
- Aligns with "local-first" philosophy

**Chunking Strategy**:
- Function-level chunks for Python/JS/TS using AST parsing
- Include context (imports, class definitions, decorators, docstrings)
- Fallback: Sliding window (500 chars, 50 overlap) for non-parseable files
- Metadata: file path, language, function/class name, line numbers, timestamps, file hash

**Indexing**:
- Storage: `~/.ai-agent/faiss_index/`
- What: All `.py`, `.js`, `.ts`, `.md`, `.json`, `.yaml`, `.yml`, `.txt` files
- Exclude: `.venv/`, `node_modules/`, `__pycache__/`, `.git/`
- Incremental updates using file hashes
- Index versioning for schema changes
- Batch processing (32 files at a time)
- Respect .gitignore patterns

### ReAct Pattern Validation

**Critical Pre-Implementation Test** (scripts/test_react_format.py):
- Validated that qwen2.5-coder:1.5b can follow ReAct format
- Result: PASSED ✅ (3/3 format compliance, 2/3 tool identification)
- This confirms the model can use the Thought/Action/Action Input/Observation pattern

## Testing Strategy

Tests are in the `tests/` directory with comprehensive coverage:

### Unit Tests (tests/test_agent.py)
- **Pre-flight checks**: Validates Ollama service is running and model is available
- **Agent initialization**: Tests agent creation with different providers
- **Configuration validation**: Tests settings and environment variables
- **Basic functionality**: Tests simple query responses
- **Conversation memory**: Tests history retention across queries
- **Error handling**: Tests Ollama connection failures, API key issues
- **Integration tests**: Tests with real Ollama instance

### Tool Tests (tests/test_tools/) - ✅ COMPLETE
- **test_read_file.py**: 40+ tests for read functionality, encoding, error handling
- **test_list_directory.py**: 50+ tests for directory listing, formatting, permissions
- **test_write_file.py**: 30+ tests for write functionality, backups, diffs, confirmations
- **test_search_code.py**: 40+ tests for regex search, file filtering, exclusions
- **test_file_ops_security.py**: 30+ security tests (path traversal, symlinks, size limits)
- **Total**: 147 tests, all passing

### RAG Tests (tests/test_rag/) - ✅ COMPLETE
- **test_embeddings.py**: 31 tests for model loading, caching, embedding generation ✅ COMPLETE
- **test_indexer_integration.py**: 9 integration tests for file discovery, chunking, index building ✅ COMPLETE
- **test_retriever.py**: 8 tests for semantic search, ranking, tool integration ✅ COMPLETE
  - Retriever initialization and index loading
  - IndexNotFoundError handling
  - Semantic search with relevant/irrelevant queries
  - top_k parameter behavior and threshold filtering
  - RagSearchTool formatted output and error handling
- **Total**: 48 tests, all passing

### Future Tests (Planned - Agent Integration Phase)
- Agent tool execution with ReAct pattern
- Tool selection and routing logic
- End-to-end agent workflows with all 5 tools
- Performance benchmarks for RAG search at scale
- Multi-turn conversations with tool use

### Testing with LangChain

Use `FakeListLLM` for unit tests (avoids real LLM calls):
```python
from langchain_community.llms.fake import FakeListLLM

def test_agent():
    fake_llm = FakeListLLM(responses=["Test response"])
    agent = CodingAgent(llm=fake_llm)
    assert agent.ask("test") == "Test response"
```

### Test Count Verification

The project currently has:
- **Tool Tests**: 5 test files in `tests/test_tools/` (~147 assertions)
- **RAG Tests**: 3 test files in `tests/test_rag/` (~48 assertions)
- **Agent Tests**: Multiple files in `tests/` for agent, CLI, providers
- **Total**: 16 test files with comprehensive coverage

Note: Some tests require Ollama to be running (`ollama serve`) and the model to be available (`ollama pull qwen2.5-coder:1.5b`).

## Known Issues & Gotchas

1. **CLI info command bug**: The `print_model_info()` function expects keys that don't exist in hybrid/claude mode. Needs refactoring to handle all provider modes.

2. **Temperature default mismatch**: `.env.example` shows TEMPERATURE=0.1, but settings.py defaults to 1.0. This is confusing. (Note: Now standardized to 0.1)

3. **Hybrid mode prints to stdout**: Provider selection messages use `print()` instead of the Rich console (src/agent/agent.py:160, 162), which breaks the UI styling.

4. **Python 3.13 ChromaDB Incompatibility**: ChromaDB's chroma-hnswlib dependency fails to build on Python 3.13 due to compilation issues. **Solution**: Using FAISS instead, which has pre-built wheels and better Python 3.13 support.

5. **Large Dependencies**: PyTorch and sentence-transformers add ~3GB of dependencies. This is expected for local embeddings and is a trade-off for avoiding API costs.

6. **RAG Retriever Similarity Score** (src/rag/retriever.py:113): The similarity calculation `1.0 / (1.0 + dist)` works but could be refined. For normalized vectors with IndexFlatL2, the formula `1 - (dist² / 2)` would be more mathematically accurate for cosine similarity. Current implementation is functional but produces slightly different score ranges.

7. **RAG Retriever Caching**: Unlike embeddings.py which uses `@lru_cache`, the retriever creates new instances on each call. This means repeated searches reload the index from disk. Consider adding singleton pattern for production use.

8. **LangGraph Tool Calling Limitations** (**CRITICAL**): The LangGraph agent implementation using `create_react_agent` requires models with **native tool/function calling support**. The current default model `qwen2.5-coder:1.5b` does NOT support this and will fail to properly execute tools.
   - **Working Models**: Claude (claude-3-haiku+), GPT-4, Gemini, larger Ollama models with tool support
   - **Not Working**: qwen2.5-coder:1.5b, llama3.2:1b, most small Ollama models
   - **Workaround**: Set `ENABLE_TOOLS=False` in .env to use simple conversational mode
   - **Future Fix**: Implement prompt-based ReAct agent for small models (doesn't require native tool calling)

## Current Implementation Status

### ✅ Completed (Step 0-2)

1. **ReAct Pattern Validation** (scripts/test_react_format.py)
   - Validated qwen2.5-coder:1.5b can follow Thought/Action/Observation format
   - Result: 3/3 format compliance, 2/3 tool identification
   - Decision: Proceed with ReAct agent implementation

2. **Foundation Setup**
   - Created directory structure:
     - `src/agent/tools/` - Tool implementations
     - `src/rag/` - RAG system components
     - `tests/test_tools/` - Tool tests
     - `tests/test_rag/` - RAG tests
   - Created `src/utils/logging.py` with structured logging for file ops and RAG
   - Updated `src/config/settings.py` with comprehensive RAG and file operations configuration

3. **Dependencies Installed**
   - `faiss-cpu==1.9.0.post1` (vector database)
   - `sentence-transformers==2.3.1` (embeddings)
   - `torch>=2.0.0` (required by sentence-transformers)
   - `pathspec==0.12.1` (gitignore patterns)
   - `tqdm==4.66.1` (progress indicators)

4. **Configuration Updates**
   - Model changed from `codellama:7b` to `qwen2.5-coder:1.5b` throughout codebase
   - Added tool configuration (ENABLE_TOOLS, ENABLE_FILE_OPS, ENABLE_RAG)
   - Added RAG configuration (FAISS paths, embedding model, chunking params)
   - Added file operations configuration (size limits, allowed extensions, backups)
   - Added indexing configuration (batch size, exclude patterns, gitignore support)
   - Added logging configuration (file operations logging, log levels)

5. **Testing Infrastructure**
   - Created comprehensive `tests/test_agent.py` with Ollama pre-flight checks
   - All tests passing with qwen2.5-coder:1.5b model

6. **Step 2: File Operation Tools** (src/agent/tools/file_ops.py) ✅ COMPLETE
   - ✅ `ReadFileTool` - Read file with path validation, size limits, encoding fallback
   - ✅ `WriteFileTool` - Write with backup & diff preview, confirmation flag
   - ✅ `ListDirectoryTool` - List files and folders with metadata
   - ✅ `SearchCodeTool` - Regex code search with file filtering and exclude patterns
   - ✅ Security: Path validation (prevent traversal), size limits, backups, comprehensive logging
   - ✅ Tests: Created comprehensive test suites:
     - `tests/test_tools/test_read_file.py` - 40+ tests for read functionality
     - `tests/test_tools/test_list_directory.py` - 50+ tests for directory listing
     - `tests/test_tools/test_write_file.py` - 30+ tests for write functionality
     - `tests/test_tools/test_search_code.py` - 40+ tests for code search
     - `tests/test_tools/test_file_ops_security.py` - 30+ security tests
   - ✅ Factory functions: `get_read_file_tool()`, `get_list_directory_tool()`, `get_write_file_tool()`, `get_search_code_tool()`

7. **Step 3: RAG System - Embeddings** (src/rag/embeddings.py) ✅ COMPLETE
   - ✅ Singleton model loading using `@functools.lru_cache` decorator
   - ✅ `get_embeddings()` - Convert text chunks to 384-dim vectors with batching support
   - ✅ `get_model()` - Access singleton SentenceTransformer instance
   - ✅ Error handling: `ModelLoadError`, `EmbeddingGenerationError` with detailed messages
   - ✅ Features:
     - Automatic model caching (loaded once, reused)
     - Batch processing with configurable batch size
     - Normalization control (required for FAISS cosine similarity)
     - Progress bar support for large batches
     - Unicode and code snippet support
     - Returns JSON-serializable Python lists
   - ✅ Tests: `tests/test_rag/test_embeddings.py` - 31 tests covering:
     - Model loading and singleton pattern
     - Embedding generation and consistency
     - Input validation and error handling
     - Batch processing (1 to 1000 texts)
     - Integration and typical workflows
   - ✅ All 31 tests passing

8. **Step 3: RAG System - Indexing** (src/rag/) ✅ COMPLETE
   - ✅ **Part A: Setup and Filtering** (`indexer_utils.py`) - COMPLETE
     - `FileInfo` dataclass with path, size, mtime, content hash
     - `load_gitignore_patterns()` - Load .gitignore + hardcoded exclusions (.venv, node_modules, etc.)
     - `file_discovery_generator()` - Iterator with security checks, gitignore filtering, extension filtering
     - Leverages existing `validate_path()` for security
   - ✅ **Part B: Chunking** (`chunker.py`) - COMPLETE
     - `CodeChunker` - AST-based chunking for Python using tree-sitter
       - Extracts functions and classes with context (imports, class signatures)
       - Handles decorators, docstrings, method nesting
       - Graceful fallback if tree-sitter unavailable
     - `TextChunker` - Sliding window fallback for non-code files
       - Configurable chunk size (500 chars) and overlap (50 chars)
       - Line-accurate tracking for metadata
       - Minimum chunk size to avoid tiny chunks
     - `chunk_file()` - Orchestration function that selects appropriate chunker
       - Uses CodeChunker for .py files (with fallback)
       - Uses TextChunker for .md, .txt, and other files
       - Singleton pattern for chunker instances (cached)
     - `CodeChunk` dataclass with content, file_path, start_line, end_line, chunk_type, language, metadata
   - ✅ **Part C: Indexing Orchestration** (`indexer.py`) - COMPLETE
     - `build_index()` - Main indexing pipeline with 5 steps:
       1. File discovery using `file_discovery_generator()`
       2. Chunking all files with per-file error handling
       3. Batch embedding generation with progress bars
       4. FAISS index creation (IndexFlatL2 for cosine similarity)
       5. Index persistence (index.faiss + chunks_metadata.json + index_info.json)
     - `save_index()` - Persist FAISS index and metadata
     - `load_index()` - Load FAISS index and metadata from disk
     - `index_exists()` - Check if index files exist
     - Error handling: `IndexBuildError`, `IndexSaveError`, `IndexError`
     - Features:
       - Progress bars with tqdm integration
       - Per-file error handling (don't crash on one bad file)
       - Batch processing (configurable batch size, default 32)
       - Comprehensive logging with statistics
   - ✅ Tests: `tests/test_rag/test_indexer_integration.py` - 9 integration tests covering:
     - File discovery with gitignore patterns
     - Python code chunking (with tree-sitter fallback)
     - Text file chunking (markdown)
     - Index building and saving
     - Index loading from disk
     - Index existence checking
     - End-to-end pipeline with test codebase
   - ✅ All 9 integration tests passing

### ✅ Completed (Step 3: RAG System - Retrieval)

9. **Step 3: RAG System - Retrieval** ✅ COMPLETE
   - ✅ **Retriever** (`retriever.py`) - COMPLETE
     - `SemanticRetriever` class - Load FAISS index and perform semantic searches
     - `search()` method - Convert query to embedding, search index, return ranked results
     - Similarity threshold filtering (default 0.5) to ensure relevance
     - Error handling: `IndexNotFoundError` with helpful messages
     - Logging: Debug and info level for search operations
     - Factory function: `get_retriever()` for dependency injection
     - **Note**: Similarity score calculation uses `1.0 / (1.0 + dist)` formula
   - ✅ **RAG Search Tool** (`rag_search.py`) - COMPLETE
     - `RagSearchTool` - LangChain BaseTool wrapper for retriever
     - Pydantic input validation with `RagSearchInput` schema
     - Formatted output for LLM consumption (file path, line numbers, score, content)
     - Error handling: IndexNotFoundError, generic exceptions
     - Async support (delegates to sync)
     - Factory function: `get_rag_search_tool()` for dependency injection
     - **Note**: Hardcoded top_k=5, could be made configurable
   - ✅ Tests: `tests/test_rag/test_retriever.py` - 8 tests covering:
     - Retriever initialization and index loading
     - IndexNotFoundError when index missing
     - Semantic search with relevant/irrelevant queries
     - No results for irrelevant query (threshold filtering)
     - top_k parameter behavior
     - RagSearchTool formatted output
     - RagSearchTool error handling
   - ✅ All 8 tests passing
   - **Technical Review**: Grade B+ - Functional and well-tested, minor improvements recommended:
     - Consider caching retriever instances to avoid repeated disk I/O
     - Similarity score calculation could be refined for normalized vectors
     - top_k could be made configurable via tool input

### ✅ Phase 2: Tools & RAG System Complete

**Summary**: All file operation tools and RAG components are implemented, tested, and ready for agent integration.
- **Total Tests**: 195 tests passing (147 file ops + 48 RAG)
- **Test Coverage**: Unit tests + integration tests + security tests
- **Components**: 4 file operation tools + 1 RAG search tool + embeddings + indexing + retrieval
- **Status**: ✅ Ready for agent integration (Step 4)

---

### ⚠️ Completed with Limitations: Agent Integration

10. **Step 4: Agent Refactoring** ⚠️ COMPLETE (with model limitations)
    - ✅ Upgraded from ConversationChain to LangGraph ReAct agent using `create_react_agent`
    - ✅ Registered all 5 tools (4 file ops + 1 RAG search)
    - ✅ Implemented MemorySaver for state persistence via checkpointing
    - ✅ Cleaned up unused imports and fixed missing methods
    - ✅ Fixed dependency conflicts (langgraph 0.2.76 + langchain-core 0.3.63)
    - ⚠️ **Model Compatibility Issue**: qwen2.5-coder:1.5b lacks native tool calling support
    - ✅ Agent works in conversational mode (`ENABLE_TOOLS=False`)
    - ⚠️ Agent requires Claude/GPT/Gemini for tool mode (`ENABLE_TOOLS=True`)
    - ✅ Added tests: `test_agent_react.py` and `test_langgraph_agent.py`

11. **Step 5: CLI Integration** (PARTIAL - IN PROGRESS)
    - ⏳ Add `index` command for codebase indexing (TODO)
    - ⏳ Add `info` subcommand to show index statistics (TODO)
    - ⏳ Add tool execution indicators (show which tool is running) (TODO)
    - ⏳ Update help messages with new commands (TODO)
    - ⏳ Add progress indicators for indexing (TODO)

12. **Step 6: Testing & Validation** (IN PROGRESS)
    - ✅ Unit tests for file operations (COMPLETE - 5 test files)
    - ✅ Unit tests for embeddings (COMPLETE)
    - ✅ Integration tests for indexer (COMPLETE)
    - ✅ Unit tests for retriever (COMPLETE)
    - ✅ Security tests for path traversal, symlinks (COMPLETE)
    - ✅ LangGraph agent tests (COMPLETE - test_langgraph_agent.py)
    - ⏳ End-to-end workflow tests with all tools (IN PROGRESS)
    - ⏳ Performance benchmarks for RAG search (TODO)
    - ⏳ Multi-turn conversation tests (TODO)

## Future Enhancements (as documented)

From ARCHITECTURE.md roadmap:

### Phase 2 (Current Development)
- 🚧 **File Operations**: Read, write, list, search files with safety features
- 🚧 **RAG System**: FAISS-based semantic codebase search with function-level chunking
- 🚧 **Tool-Based Agent**: ReAct pattern for autonomous tool use

### Phase 3 (Post-Tool Implementation)
- [ ] **Conversation Persistence**: Redis or Postgres-backed message history
- [ ] **Streaming**: LangChain streaming callbacks for real-time output
- [ ] **Web UI**: FastAPI + LangServe for API deployment
- [ ] **Multi-Agent**: LangGraph for complex workflows
- [ ] **SQL Agent**: LangChain SQL agent for direct database queries

### Phase 4 (Advanced)
- [ ] **Code Execution**: LangChain code execution tool
- [ ] **Git Integration**: Custom LangChain tool for version control
- [ ] **Analytics**: LangSmith for tracing and monitoring

## Project Structure

```
ollama-agentic-project/
├── src/
│   ├── agent/
│   │   ├── agent.py              # Main agent with LangGraph integration
│   │   ├── prompts.py            # Database knowledge & tool prompts
│   │   └── tools/
│   │       ├── __init__.py       # Tool registry
│   │       ├── file_ops.py       # File operation tools (✅ COMPLETE)
│   │       └── rag_search.py     # RAG search tool (✅ COMPLETE)
│   ├── cli/
│   │   └── main.py               # Terminal interface
│   ├── config/
│   │   └── settings.py           # Configuration (updated with RAG & tools)
│   ├── rag/
│   │   ├── __init__.py           # RAG module exports (✅ COMPLETE)
│   │   ├── embeddings.py         # Embedding model setup (✅ COMPLETE)
│   │   ├── indexer_utils.py      # File discovery & gitignore (✅ COMPLETE)
│   │   ├── chunker.py            # Code & text chunking (✅ COMPLETE)
│   │   ├── indexer.py            # FAISS index building (✅ COMPLETE)
│   │   └── retriever.py          # Semantic search (✅ COMPLETE)
│   └── utils/
│       ├── __init__.py
│       └── logging.py            # Structured logging (✅ COMPLETE)
├── tests/
│   ├── test_agent.py             # Basic agent tests
│   ├── test_agent_react.py       # ReAct agent tests
│   ├── test_langgraph_agent.py   # LangGraph agent tests ✅ NEW
│   ├── test_cli_info.py          # CLI info command tests
│   ├── test_gemini_provider.py   # Gemini provider tests
│   ├── test_tools/               # Tool tests (5 test files)
│   │   ├── test_read_file.py
│   │   ├── test_list_directory.py
│   │   ├── test_write_file.py
│   │   ├── test_search_code.py
│   │   └── test_file_ops_security.py
│   └── test_rag/                 # RAG tests (3 test files)
│       ├── test_embeddings.py
│       ├── test_indexer_integration.py
│       └── test_retriever.py
├── scripts/
│   ├── benchmark_models.py       # Model comparison
│   └── test_react_format.py      # ReAct validation (✅ PASSED)
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT_ANALYSIS.md
│   └── [other documentation]
├── main.py                       # Entry point
├── requirements.txt              # Includes LangGraph, FAISS, embeddings
├── requirements-dev.txt          # Testing and development tools
├── .env.example                  # Configuration template
└── CLAUDE.md                     # This file
```

## Documentation Files

- **README.md**: User-facing documentation, quick start, deployment guide
- **ARCHITECTURE.md**: Detailed technical architecture, LangChain integration patterns
- **DEPLOYMENT_ANALYSIS.md**: Cloud deployment options (CPU & GPU instances)
- **GPU_DEPLOYMENT_GUIDE.md**: GPU deployment strategy & ROI analysis
- **MULTI_PROVIDER_GUIDE.md**: Guide for multi-provider setup
- **DESIGN_SUMMARY.md**: Quick reference guide
- **EXECUTIVE_DECISION_BRIEF.md**: Management-focused ROI analysis
- **MODEL_COMPARISON_REPORT.md**: Model selection rationale
- **CLAUDE.md**: This file - guidance for Claude Code when working with the codebase

## Key Decisions Made

### Model Selection
- **Decision**: Use qwen2.5-coder:1.5b
- **Rationale**: Local machine limitations, smaller model size, validated ReAct compatibility
- **Impact**: Updated throughout codebase (.env.example, settings.py, tests, scripts)

### Vector Database
- **Decision**: Use FAISS instead of ChromaDB
- **Rationale**: Python 3.13 compatibility (ChromaDB's chroma-hnswlib fails to build)
- **Trade-offs**: Manual save/load (1-2 lines) vs automatic persistence
- **Benefits**: Pre-built wheels, faster, lower memory, production-proven

### Agent Pattern
- **Decision**: LangGraph-based ReAct Agent (upgraded from basic ReAct)
- **Rationale**: LangGraph provides better state management, checkpointing, and workflow control
- **Trade-offs**: Slightly more complex than basic ReAct, but offers production-ready features
- **Future**: Can extend to multi-agent workflows (researcher + coder + reviewer)

### Implementation Approach
- **Decision**: Gradual rollout with feature flags (ENABLE_TOOLS now defaults to True)
- **Rationale**: Tool system complete and tested (195 tests passing), ready for integration
- **Migration**: Phase 1 (ConversationChain) → Phase 2 (ReAct with tools) → Phase 3 (LangGraph multi-agent)

## Next Steps

**✅ Phase 2 Complete**: All RAG components implemented and tested
- ✅ Embeddings: sentence-transformers with singleton caching (31 tests passing)
- ✅ Indexing: File discovery, AST chunking, FAISS index building (9 tests passing)
- ✅ Retrieval: Semantic search with ranking and threshold filtering (8 tests passing)
- ✅ Tools: 4 file operations + 1 RAG search tool (147 file ops + 8 retrieval = 155 tool tests)
- **Total**: 195 tests passing (147 file ops + 48 RAG)

**⏳ Phase 3 (CURRENT)**: Model Compatibility & CLI Enhancement

1. **Model Compatibility Resolution** (Step 5 - HIGH PRIORITY)
   - ⏳ **Option A**: Implement prompt-based ReAct agent for small models (recommended)
     - Create custom ReAct loop using prompts instead of `bind_tools()`
     - Compatible with qwen2.5-coder:1.5b and other small models
     - Uses text-based tool descriptions and output parsing
   - ⏳ **Option B**: Document Claude/Gemini as required for tool mode
     - Update .env.example to default to `ENABLE_TOOLS=False`
     - Add clear documentation about model requirements
     - Provide instructions for using with Claude API
   - ⏳ Test tool functionality with compatible models (Claude Haiku, Gemini Flash)

2. **CLI Integration** (Step 6 - IN PROGRESS)
   - ⏳ Add `index` command to build/rebuild codebase index
   - ⏳ Add `info` subcommand to show index statistics (size, files, last updated)
   - ⏳ Add tool execution indicators (show which tool is running)
   - ⏳ Update help messages to document new commands
   - ⏳ Add progress indicators for indexing operations
   - ⏳ Improve error messages for common issues

3. **Testing & Validation** (Step 7 - IN PROGRESS)
   - ✅ LangGraph agent structure tests (COMPLETE)
   - ⚠️ End-to-end tool tests (blocked by model compatibility)
   - ✅ Multi-turn conversation tests (COMPLETE - simple mode)
   - ⏳ Performance benchmarks for RAG search
   - ⏳ Load testing with large codebases (10k+ files)
   - ⏳ Error recovery and edge case testing

4. **Production Readiness** (Step 8 - TODO)
   - Performance optimization (caching, lazy loading)
   - Logging and monitoring improvements
   - Documentation updates (user guide, troubleshooting)
   - Cloud deployment guide updates
   - Team rollout plan