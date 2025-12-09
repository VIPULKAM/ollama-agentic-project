# Testing Guide

This guide explains how to run tests for the AI Coding Agent, including parallel execution.

## Quick Start

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# Run tests in parallel (recommended)
pytest tests/ -n auto -v

# Or use the test runner script
./run_tests.sh parallel
```

## Test Organization

```
tests/
├── test_agent.py                   # Agent core tests
├── test_langgraph_agent.py        # LangGraph integration
├── test_quick_wins_regression.py  # Regression tests
├── test_tools/                     # Tool tests (6 files, 159 tests)
│   ├── test_crawl_and_index.py   # Crawl tool tests
│   ├── test_file_ops_security.py # Security tests
│   ├── test_list_directory.py
│   ├── test_read_file.py
│   ├── test_search_code.py
│   └── test_write_file.py
└── test_rag/                       # RAG system tests (5 files, 48+ tests)
    ├── test_crawl_tracker.py      # URL tracking (NEW - 40+ tests)
    ├── test_crawl_integration.py  # Integration (NEW - 10+ tests)
    ├── test_embeddings.py
    ├── test_indexer_integration.py
    └── test_retriever.py
```

## Running Tests

### Basic Commands

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_rag/test_crawl_tracker.py

# Specific test class
pytest tests/test_rag/test_crawl_tracker.py::TestCrawlTrackerInit

# Specific test function
pytest tests/test_rag/test_crawl_tracker.py::test_compute_content_hash
```

### Parallel Execution 🚀

```bash
# Auto-detect CPU count (recommended)
pytest tests/ -n auto

# Use specific number of workers
pytest tests/ -n 4

# Crawler tests in parallel
pytest tests/test_rag/test_crawl_tracker.py tests/test_rag/test_crawl_integration.py -n auto
```

### Using the Test Runner Script

```bash
# Show all options
./run_tests.sh

# Run all tests in parallel
./run_tests.sh parallel

# Run crawler tests only (parallel)
./run_tests.sh crawler-parallel

# Quick crawler tests
./run_tests.sh quick

# With coverage report
./run_tests.sh coverage-parallel
```

## Test Categories

### By Type

```bash
# Unit tests only
pytest tests/ -m unit

# Integration tests only
pytest tests/ -m integration

# Fast tests (exclude slow)
pytest tests/ -m "not slow"
```

### By Component

```bash
# Crawler tests
pytest tests/test_rag/test_crawl_tracker.py tests/test_rag/test_crawl_integration.py -v

# Tool tests
pytest tests/test_tools/ -v

# RAG tests
pytest tests/test_rag/ -v

# Agent tests
pytest tests/test_agent.py tests/test_langgraph_agent.py -v
```

## Coverage Reports

```bash
# Generate HTML coverage report
pytest tests/ --cov=src --cov-report=html

# View in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Coverage with parallel execution
pytest tests/ -n auto --cov=src --cov-report=html
```

## Parallel Testing Guidelines

### Benefits

- **Speed**: 3-5x faster on multi-core systems
- **Efficiency**: Utilize all CPU cores
- **Best for**: Large test suites, integration tests

### When to Use Parallel

✅ **Good candidates:**
- Unit tests with isolated fixtures
- Tests using `tmp_path` fixture
- Parameterized tests
- Independent integration tests

❌ **Avoid for:**
- Tests that modify global state
- Tests that use shared resources (databases, files)
- Tests that depend on execution order

### Our Tests Are Parallel-Ready

All crawler tests use:
- `tmp_path` fixture (isolated temp directories)
- No shared state between tests
- Independent test data
- Proper cleanup in fixtures

## Test Performance

### Benchmarks (on 8-core machine)

```bash
# Serial execution
pytest tests/test_rag/test_crawl_tracker.py
# ~15 seconds (40+ tests)

# Parallel execution (-n auto)
pytest tests/test_rag/test_crawl_tracker.py -n auto
# ~4 seconds (40+ tests) - 3.75x faster!
```

## CI/CD Integration

```yaml
# Example GitHub Actions workflow
- name: Run tests in parallel
  run: pytest tests/ -n auto --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Debugging Failed Tests

```bash
# Verbose output
pytest tests/test_crawl_tracker.py -vv

# Show full traceback
pytest tests/test_crawl_tracker.py --tb=long

# Stop on first failure
pytest tests/ -x

# Drop into debugger on failure
pytest tests/ --pdb

# Run last failed tests
pytest --lf
```

## Parameterized Tests

Our tests use parameterization for parallel-friendly testing:

```python
@pytest.mark.parametrize("url,content,chunks", [
    ("https://docs.python.org", "Python docs", 10),
    ("https://fastapi.tiangolo.com", "FastAPI docs", 25),
])
def test_add_multiple_urls(tracker, url, content, chunks):
    # Each parameter set runs in parallel
    ...
```

## Common Issues

### Issue: Tests pass individually but fail in parallel

**Solution**: Check for shared state or global variables
```bash
# Run with 1 worker to confirm
pytest tests/ -n 1
```

### Issue: Temp files conflict

**Solution**: Always use `tmp_path` fixture
```python
def test_something(tmp_path):
    test_file = tmp_path / "test.txt"
    # Each test gets isolated temp directory
```

### Issue: Database/network conflicts

**Solution**: Use mocks for external resources
```python
with patch('module.external_call'):
    # Test logic here
```

## Test Metrics

Current test suite:
- **Total Tests**: ~260+ tests
- **Crawler Tests**: 50+ tests (unit + integration)
- **Parallel Speedup**: 3-4x on 8-core CPU
- **Coverage**: >85% (target: 90%)

## Best Practices

1. **Isolation**: Each test should be independent
2. **Fixtures**: Use fixtures for common setup
3. **Cleanup**: Always clean up resources
4. **Naming**: Use descriptive test names
5. **Assertions**: One logical assertion per test
6. **Speed**: Keep tests fast (<100ms if possible)
7. **Mocking**: Mock external dependencies

## Contributing Tests

When adding new tests:

1. Use `tmp_path` for file operations
2. Add appropriate markers (`@pytest.mark.integration`)
3. Ensure parallel compatibility
4. Update this guide if needed

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-xdist](https://pytest-xdist.readthedocs.io/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
