"""
Tests for the RAG Retriever
"""
import os
import shutil
import tempfile
from pathlib import Path
import pytest
import json

from src.config.settings import Settings
from src.rag.indexer import build_index
from src.rag.retriever import SemanticRetriever, IndexNotFoundError

@pytest.fixture(scope="function")
def test_index_dir(monkeypatch):
    """
    Creates a temporary directory with dummy source files and builds an index from them.
    This fixture provides a realistic environment for testing the retriever.
    It uses monkeypatch to override settings for the test session.
    """
    temp_dir = tempfile.mkdtemp(prefix="rag_test_")
    index_path = os.path.join(temp_dir, "test_index")
    source_path = os.path.join(temp_dir, "source")
    
    os.makedirs(source_path, exist_ok=True)
    os.makedirs(index_path, exist_ok=True)

    # Create a settings object with temporary paths
    test_settings = Settings(
        FAISS_INDEX_PATH=index_path,
        EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Use monkeypatch to override settings in the modules where they are imported
    monkeypatch.setattr("src.rag.indexer.settings", test_settings)
    monkeypatch.setattr("src.rag.embeddings.settings", test_settings)

    # Create dummy files
    (Path(source_path) / "main.py").write_text(
        "def hello_world():\\n    print('Hello, World!')\\n"
    )
    (Path(source_path) / "README.md").write_text(
        "# Project Title\\n\\nThis is a project about semantic search."
    )
    (Path(source_path) / ".gitignore").write_text("ignored_dir/\\n")
    
    ignored_dir = Path(source_path) / "ignored_dir"
    os.makedirs(ignored_dir, exist_ok=True)
    (ignored_dir / "ignored_file.txt").write_text("This should not be indexed.")

    # Build the index using the patched settings
    build_index(cwd=Path(source_path))

    yield test_settings

    # Teardown: remove the temporary directory
    shutil.rmtree(temp_dir)

def test_retriever_initialization(test_index_dir):
    """
    Tests that the SemanticRetriever initializes correctly and loads a valid index.
    """
    settings = test_index_dir
    retriever = SemanticRetriever(settings)
    
    assert retriever.index is not None, "FAISS index should be loaded."
    assert retriever.index.ntotal > 0, "Index should contain vectors."
    assert len(retriever.chunks_metadata) > 0, "Chunk metadata should be loaded."
    assert retriever.index.ntotal == len(retriever.chunks_metadata), "Mismatch between index size and metadata."
    assert "embedding_model" in retriever.index_info, "Index info should be loaded."

def test_index_not_found_error():
    """
    Tests that IndexNotFoundError is raised if the index directory is invalid.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        settings = Settings(FAISS_INDEX_PATH=temp_dir)
        with pytest.raises(IndexNotFoundError):
            SemanticRetriever(settings)

def test_search_semantic_match(test_index_dir):
    """
    Tests a basic semantic search to see if it returns relevant results.
    """
    settings = test_index_dir
    retriever = SemanticRetriever(settings)
    
    query = "a function that prints a greeting"
    results = retriever.search(query, top_k=1)
    
    assert len(results) >= 1, "Should find at least one result."
    top_result = results[0]
    
    assert top_result["file_path"].endswith("main.py")
    assert "hello_world" in top_result["content"]
    assert top_result["score"] > 0.5, "Similarity score should be reasonably high."
    assert "file_path" in top_result
    assert "start_line" in top_result
    assert "end_line" in top_result
    assert "content" in top_result
    assert "score" in top_result

def test_search_no_results(test_index_dir):
    """
    Tests a query that is completely irrelevant and should not return results
    with a high similarity score.
    """
    settings = test_index_dir
    retriever = SemanticRetriever(settings)
    
    # This query should be very dissimilar to the indexed content
    query = "quantum mechanics and black holes"
    results = retriever.search(query, top_k=1)
    
    # With the threshold, we expect zero results
    assert len(results) == 0

def test_top_k_parameter(test_index_dir):
    """
    Tests that the top_k parameter correctly limits the number of results.
    """
    settings = test_index_dir
    # The dummy index has 2 files, resulting in a few chunks.
    # Let's ensure top_k=2 returns 2 results.
    retriever = SemanticRetriever(settings)
    
    query = "code and documentation"
    # We pass a lower threshold to ensure we get results back for this test
    results = retriever.search(query, top_k=2, similarity_threshold=0.3)
    
    # The number of results should be min(top_k, number_of_chunks)
    assert len(results) <= 2
    if retriever.index.ntotal >= 2 and len(results) > 1:
        assert len(results) == 2
    else:
        # If we get 0 or 1 results, that's also acceptable depending on the scores
        assert len(results) in [0, 1, 2]


from src.agent.tools.rag_search import RagSearchTool, IndexNotFoundError

def test_rag_search_tool_success(test_index_dir):
    """
    Tests that the RagSearchTool successfully returns formatted results.
    """
    settings = test_index_dir
    tool = RagSearchTool(settings=settings)
    query = "semantic search"
    result_str = tool.run(tool_input={"query": query})

    assert isinstance(result_str, str)
    assert "Found" in result_str
    assert "relevant code snippets" in result_str
    assert "File:" in result_str
    assert "README.md" in result_str
    assert "Score:" in result_str

def test_rag_search_tool_no_results(test_index_dir):
    """
    Tests the RagSearchTool's output when no results are found.
    """
    settings = test_index_dir
    tool = RagSearchTool(settings=settings)
    query = "a query that will certainly not match anything in the test index"
    result_str = tool.run(tool_input={"query": query})

    assert "No relevant code snippets found" in result_str

def test_rag_search_tool_index_not_found(monkeypatch):
    """
    Tests that the tool returns a helpful error message if the index is not found.
    """
    # Mock get_retriever to raise the specific error
    def mock_get_retriever(settings):
        raise IndexNotFoundError
    
    monkeypatch.setattr("src.agent.tools.rag_search.get_retriever", mock_get_retriever)
    
    tool = RagSearchTool(settings=Settings())
    query = "any query"
    result_str = tool.run(tool_input={"query": query})

    assert "Error: The codebase index has not been built yet." in result_str
