# File Operations Tools - Test Suite

Comprehensive test suite for `read_file` and `list_directory` tools with security focus.

## Test Files

### 1. `test_read_file.py`
**Tests:** 25+ test cases for ReadFile tool

**Coverage:**
- ✅ Path validation (valid paths in CWD)
- ✅ Relative and absolute paths
- ✅ File size checking (large files rejected)
- ✅ Non-existent file handling
- ✅ Directory vs file validation
- ✅ Path traversal prevention
- ✅ Encoding fallback (UTF-8 → latin-1)
- ✅ Empty/whitespace path handling
- ✅ Special characters in filenames
- ✅ LangChain interface compatibility

**Security Tests:**
- Path traversal attacks (`../../etc/passwd`)
- Absolute paths outside CWD (`/etc/passwd`)
- Empty and whitespace paths
- Missing keys handling

---

### 2. `test_list_directory.py`
**Tests:** 40+ test cases for ListDirectory tool

**Coverage:**
- ✅ List current directory (default and explicit)
- ✅ List subdirectories and nested directories
- ✅ Non-existent directory handling
- ✅ File vs directory validation
- ✅ Path traversal prevention
- ✅ Output formatting (headers, separators, summary)
- ✅ Directory/file type indicators (DIR/FILE)
- ✅ Human-readable file sizes (B, KB, MB, GB)
- ✅ Sorted output (directories first, alphabetically)
- ✅ Empty directory handling
- ✅ Hidden files visibility
- ✅ Directories with spaces in names
- ✅ Deeply nested directories
- ✅ LangChain interface compatibility

**Helper Method Tests:**
- `_format_size()` - bytes to human-readable
- `_format_time()` - timestamp to readable format

---

### 3. `test_file_ops_security.py`
**Tests:** 40+ security-focused test cases for BOTH tools

**Coverage:**

#### Path Traversal Attack Prevention
- Simple parent directory access (`../`)
- Multiple levels (`../../`, `../../../`)
- Mixed traversal (`src/../../etc`)
- Absolute paths outside CWD (`/etc/passwd`)
- Home directory access (`~/.bashrc`)
- Root directory access (`/`)

#### Symlink Attack Prevention
- Symlinks pointing outside CWD
- Symlinks pointing to sensitive files (`/etc/passwd`)
- Symlinks within CWD (allowed)
- Symlink validation consistency

#### Input Validation
- Empty strings
- Whitespace-only strings
- Null byte injection (`\x00`)
- Newline injection (`\n`)

#### Permission Handling
- Unreadable files (permission denied)
- Inaccessible directories

#### Cross-Tool Security
- Consistent validation between read_file and list_directory
- Multiple attack vectors tested on both tools
- Valid operations not blocked

---

## Running Tests

### Run All Tool Tests
```bash
# Using Python script (cross-platform)
python scripts/run_tool_tests.py

# Using bash script (Linux/Mac)
chmod +x scripts/run_tool_tests.sh
./scripts/run_tool_tests.sh
```

### Run Individual Test Files
```bash
# ReadFile tests only
pytest tests/test_tools/test_read_file.py -v

# ListDirectory tests only
pytest tests/test_tools/test_list_directory.py -v

# Security tests only
pytest tests/test_tools/test_file_ops_security.py -v
```

### Run Specific Test Classes
```bash
# Path validation tests
pytest tests/test_tools/test_read_file.py::TestPathValidation -v

# Output formatting tests
pytest tests/test_tools/test_list_directory.py::TestListDirectoryOutput -v

# Path traversal attack tests
pytest tests/test_tools/test_file_ops_security.py::TestPathTraversalAttacks -v
```

### Run Specific Test Methods
```bash
# Single test
pytest tests/test_tools/test_read_file.py::TestReadFileTool::test_path_traversal_blocked -v

# Security test
pytest tests/test_tools/test_file_ops_security.py::TestSymlinkAttacks::test_symlink_to_outside_cwd_read -v
```

### Run with Coverage
```bash
# Generate coverage report
pytest tests/test_tools/ --cov=src.agent.tools.file_ops --cov-report=html

# View coverage
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
```

---

## Test Statistics

| Test File | Test Cases | Security Tests | Coverage |
|-----------|------------|----------------|----------|
| `test_read_file.py` | 25+ | 8+ | Path validation, file operations |
| `test_list_directory.py` | 40+ | 6+ | Directory listing, formatting |
| `test_file_ops_security.py` | 40+ | 40+ | All security vectors |
| **Total** | **105+** | **54+** | **Comprehensive** |

---

## Security Test Coverage

### Attack Vectors Tested

| Attack Type | Blocked | Test Coverage |
|-------------|---------|---------------|
| Path traversal (`../`) | ✅ | 12+ tests |
| Absolute paths outside CWD | ✅ | 6+ tests |
| Symlink to outside CWD | ✅ | 6+ tests |
| Null byte injection | ✅ | 2+ tests |
| Newline injection | ✅ | 2+ tests |
| Home directory access | ✅ | 2+ tests |
| Root directory access | ✅ | 2+ tests |

### Valid Operations Tested

| Operation | Allowed | Test Coverage |
|-----------|---------|---------------|
| Read files in CWD | ✅ | 10+ tests |
| Read files in subdirectories | ✅ | 8+ tests |
| List current directory | ✅ | 8+ tests |
| List subdirectories | ✅ | 6+ tests |
| Absolute paths within CWD | ✅ | 4+ tests |
| Symlinks within CWD | ✅ | 2+ tests |

---

## Test Philosophy

1. **Security First**: Every tool is tested against known attack vectors
2. **Comprehensive Coverage**: Both happy paths and error conditions
3. **Real-World Scenarios**: Integration tests simulate actual usage
4. **Cross-Tool Consistency**: Security policies are consistent across tools
5. **Defensive Programming**: Tests ensure errors are handled gracefully

---

## Adding New Tests

When adding new file operation tools:

1. Create `test_<tool_name>.py` with tool-specific tests
2. Add security tests to `test_file_ops_security.py`
3. Update this README
4. Run full test suite: `python scripts/run_tool_tests.py`

Example test structure:
```python
class TestYourTool:
    @pytest.fixture
    def tool(self):
        return get_your_tool()

    def test_basic_functionality(self, tool):
        result = tool._run("valid_input")
        assert "expected" in result

    def test_security_validation(self, tool):
        result = tool._run("../../etc/passwd")
        assert "Access denied" in result
```

---

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run Tool Tests
  run: |
    pytest tests/test_tools/ -v --cov=src.agent.tools.file_ops
```

---

## Test Maintenance

- **Run tests before commits**: `pytest tests/test_tools/ -v`
- **Update tests when tools change**: Keep tests in sync with implementation
- **Add tests for new features**: Every new feature needs tests
- **Fix failing tests immediately**: Never commit with failing tests

---

## Contact

For questions about these tests, refer to the implementation in:
- `src/agent/tools/file_ops.py` - Tool implementations
- `CLAUDE.md` - Project documentation
