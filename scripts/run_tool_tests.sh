#!/bin/bash
# Run all file operation tool tests

echo "========================================================================"
echo "Running File Operation Tools Test Suite"
echo "========================================================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "\n${BLUE}Test Suite 1: ReadFile Tool${NC}"
echo "------------------------------------------------------------------------"
pytest tests/test_tools/test_read_file.py -v
READ_RESULT=$?

echo -e "\n${BLUE}Test Suite 2: ListDirectory Tool${NC}"
echo "------------------------------------------------------------------------"
pytest tests/test_tools/test_list_directory.py -v
LIST_RESULT=$?

echo -e "\n${BLUE}Test Suite 3: Security Tests (Both Tools)${NC}"
echo "------------------------------------------------------------------------"
pytest tests/test_tools/test_file_ops_security.py -v
SECURITY_RESULT=$?

echo -e "\n========================================================================"
echo "Test Results Summary"
echo "========================================================================"

if [ $READ_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} ReadFile Tool Tests: PASSED"
else
    echo -e "${RED}✗${NC} ReadFile Tool Tests: FAILED"
fi

if [ $LIST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} ListDirectory Tool Tests: PASSED"
else
    echo -e "${RED}✗${NC} ListDirectory Tool Tests: FAILED"
fi

if [ $SECURITY_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Security Tests: PASSED"
else
    echo -e "${RED}✗${NC} Security Tests: FAILED"
fi

echo "========================================================================"

# Exit with failure if any test suite failed
if [ $READ_RESULT -ne 0 ] || [ $LIST_RESULT -ne 0 ] || [ $SECURITY_RESULT -ne 0 ]; then
    echo -e "\n${RED}Some tests failed. Please review the output above.${NC}"
    exit 1
else
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
fi
