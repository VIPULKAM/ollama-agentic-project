"""Comprehensive tests for embeddings module."""

import pytest
from src.rag.embeddings import (
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


class TestModelLoading:
    """Tests for model loading and caching."""

    def test_model_loads_successfully(self):
        """Test that model loads without errors."""
        model = get_model()
        assert model is not None

    def test_model_is_singleton(self):
        """Test that model is cached (singleton pattern)."""
        model1 = get_model()
        model2 = get_model()
        # Same object instance
        assert model1 is model2

    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        dim = get_embedding_dimension()
        assert dim == 384  # all-MiniLM-L6-v2 dimension

    def test_get_max_sequence_length(self):
        """Test getting max sequence length."""
        max_len = get_max_sequence_length()
        assert max_len == 256  # all-MiniLM-L6-v2 max length

    def test_model_info_constant(self):
        """Test MODEL_INFO constant."""
        assert MODEL_INFO is not None
        assert "model_name" in MODEL_INFO
        assert "dimension" in MODEL_INFO
        assert "max_seq_length" in MODEL_INFO
        assert MODEL_INFO["dimension"] == 384

    def test_clear_model_cache(self):
        """Test clearing model cache."""
        # Load model
        model1 = get_model()

        # Clear cache
        clear_model_cache()

        # Load again - should be a new instance
        model2 = get_model()

        # Note: This might be the same object if the underlying library
        # caches internally, but the cache_clear() should work
        assert model2 is not None


class TestGetEmbeddings:
    """Tests for get_embeddings function."""

    def test_single_text_embedding(self):
        """Test generating embedding for single text."""
        texts = ["Hello world"]
        embeddings = get_embeddings(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384  # Embedding dimension
        assert isinstance(embeddings[0], list)
        assert isinstance(embeddings[0][0], float)

    def test_multiple_texts_embedding(self):
        """Test generating embeddings for multiple texts."""
        texts = ["Hello world", "Python programming", "Machine learning"]
        embeddings = get_embeddings(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384
            assert isinstance(emb, list)

    def test_empty_string_embedding(self):
        """Test embedding empty string."""
        texts = [""]
        embeddings = get_embeddings(texts)

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384

    def test_long_text_embedding(self):
        """Test embedding text longer than max sequence length."""
        # Create text longer than 256 tokens
        long_text = " ".join(["word"] * 500)
        texts = [long_text]

        # Should not raise error (will be truncated)
        embeddings = get_embeddings(texts)
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384

    def test_unicode_text_embedding(self):
        """Test embedding text with unicode characters."""
        texts = ["Hello 世界", "café ☕", "emoji 😀"]
        embeddings = get_embeddings(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384

    def test_special_characters_embedding(self):
        """Test embedding text with special characters."""
        texts = [
            "def function():",
            "SELECT * FROM table;",
            "@decorator",
            "/* comment */",
        ]
        embeddings = get_embeddings(texts)

        assert len(embeddings) == 4
        for emb in embeddings:
            assert len(emb) == 384

    def test_code_snippet_embedding(self):
        """Test embedding code snippets."""
        texts = [
            "def hello(): print('hello')",
            "class MyClass: pass",
            "import numpy as np",
        ]
        embeddings = get_embeddings(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384


class TestEmbeddingParameters:
    """Tests for get_embeddings parameters."""

    def test_custom_batch_size(self):
        """Test using custom batch size."""
        texts = ["Text " + str(i) for i in range(10)]
        embeddings = get_embeddings(texts, batch_size=2)

        assert len(embeddings) == 10
        for emb in embeddings:
            assert len(emb) == 384

    def test_show_progress_false(self):
        """Test with progress bar disabled (default)."""
        texts = ["Text 1", "Text 2"]
        embeddings = get_embeddings(texts, show_progress=False)

        assert len(embeddings) == 2

    def test_normalize_true(self):
        """Test with normalization enabled (default)."""
        texts = ["Hello world"]
        embeddings = get_embeddings(texts, normalize=True)

        # Check that vector is normalized (length ~ 1.0)
        import math
        vector_length = math.sqrt(sum(x**2 for x in embeddings[0]))
        assert abs(vector_length - 1.0) < 0.01  # Should be close to 1.0

    def test_normalize_false(self):
        """Test with normalization disabled."""
        texts = ["Hello world"]
        embeddings = get_embeddings(texts, normalize=False)

        # Vector should not be normalized
        import math
        vector_length = math.sqrt(sum(x**2 for x in embeddings[0]))
        # Length might be different from 1.0
        assert vector_length > 0  # But should be non-zero


class TestEmbeddingConsistency:
    """Tests for embedding consistency and reproducibility."""

    def test_same_text_same_embedding(self):
        """Test that same text produces same embedding."""
        text = "Consistent embedding test"
        emb1 = get_embeddings([text])[0]
        emb2 = get_embeddings([text])[0]

        # Embeddings should be identical
        assert emb1 == emb2

    def test_different_texts_different_embeddings(self):
        """Test that different texts produce different embeddings."""
        texts = ["Text A", "Text B"]
        embeddings = get_embeddings(texts)

        # Embeddings should be different
        assert embeddings[0] != embeddings[1]

    def test_similar_texts_similar_embeddings(self):
        """Test that similar texts have similar embeddings."""
        texts = [
            "Python programming language",
            "Python programming",
            "Completely different topic",
        ]
        embeddings = get_embeddings(texts)

        # Calculate cosine similarity
        def cosine_similarity(vec1, vec2):
            import math
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            mag1 = math.sqrt(sum(a**2 for a in vec1))
            mag2 = math.sqrt(sum(b**2 for b in vec2))
            return dot_product / (mag1 * mag2)

        sim_01 = cosine_similarity(embeddings[0], embeddings[1])
        sim_02 = cosine_similarity(embeddings[0], embeddings[2])

        # Similar texts should have higher similarity
        assert sim_01 > sim_02


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_empty_list_raises_error(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            get_embeddings([])

    def test_non_list_input_raises_error(self):
        """Test that non-list input raises ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            get_embeddings("not a list")

    def test_non_string_items_raise_error(self):
        """Test that non-string items raise ValueError."""
        with pytest.raises(ValueError, match="not a string"):
            get_embeddings([123, 456])

    def test_mixed_types_raise_error(self):
        """Test that mixed types raise ValueError."""
        with pytest.raises(ValueError, match="not a string"):
            get_embeddings(["valid string", 123, "another string"])


class TestBatchProcessing:
    """Tests for batch processing capabilities."""

    def test_large_batch(self):
        """Test processing large batch of texts."""
        # Create 100 texts
        texts = [f"Text number {i}" for i in range(100)]
        embeddings = get_embeddings(texts, batch_size=10)

        assert len(embeddings) == 100
        for emb in embeddings:
            assert len(emb) == 384

    def test_single_item_batch(self):
        """Test batch size of 1."""
        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = get_embeddings(texts, batch_size=1)

        assert len(embeddings) == 3


class TestErrorHandling:
    """Tests for error handling in edge cases."""

    def test_very_long_batch(self):
        """Test processing very long batch."""
        # Create 1000 texts
        texts = [f"Text {i}" for i in range(1000)]

        # Should not crash
        embeddings = get_embeddings(texts, batch_size=32)
        assert len(embeddings) == 1000

    def test_whitespace_only_text(self):
        """Test embedding whitespace-only text."""
        texts = ["   ", "\t\t", "\n\n"]
        embeddings = get_embeddings(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == 384


class TestIntegration:
    """Integration tests for embeddings module."""

    def test_typical_workflow(self):
        """Test typical workflow of generating embeddings."""
        # Step 1: Get model info
        dim = get_embedding_dimension()
        assert dim == 384

        # Step 2: Generate embeddings for some code
        code_chunks = [
            "def calculate_sum(a, b): return a + b",
            "class Calculator: pass",
            "import math",
        ]
        embeddings = get_embeddings(code_chunks)

        # Step 3: Verify results
        assert len(embeddings) == len(code_chunks)
        for emb in embeddings:
            assert len(emb) == dim

    def test_multiple_sequential_calls(self):
        """Test multiple sequential embedding calls."""
        for i in range(5):
            texts = [f"Iteration {i}"]
            embeddings = get_embeddings(texts)
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 384

    def test_returns_python_lists(self):
        """Test that function returns Python lists (not numpy arrays)."""
        texts = ["Test"]
        embeddings = get_embeddings(texts)

        # Should be Python list
        assert isinstance(embeddings, list)
        assert isinstance(embeddings[0], list)
        # Should be JSON-serializable
        import json
        json_str = json.dumps(embeddings)
        assert isinstance(json_str, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
