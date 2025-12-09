#!/bin/bash
# Test runner script with various execution modes

set -e  # Exit on error

echo "🧪 AI Coding Agent Test Runner"
echo "=============================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
MODE=${1:-"all"}

case $MODE in
    "all")
        echo -e "${BLUE}Running all tests...${NC}"
        pytest tests/ -v
        ;;

    "parallel")
        echo -e "${BLUE}Running tests in parallel (auto-detect CPUs)...${NC}"
        pytest tests/ -n auto -v
        ;;

    "parallel-4")
        echo -e "${BLUE}Running tests in parallel (4 workers)...${NC}"
        pytest tests/ -n 4 -v
        ;;

    "crawler")
        echo -e "${BLUE}Running crawler tests only...${NC}"
        pytest tests/test_rag/test_crawl_tracker.py tests/test_rag/test_crawl_integration.py -v
        ;;

    "crawler-parallel")
        echo -e "${BLUE}Running crawler tests in parallel...${NC}"
        pytest tests/test_rag/test_crawl_tracker.py tests/test_rag/test_crawl_integration.py -n auto -v
        ;;

    "unit")
        echo -e "${BLUE}Running unit tests only...${NC}"
        pytest tests/ -m unit -v
        ;;

    "integration")
        echo -e "${BLUE}Running integration tests only...${NC}"
        pytest tests/ -m integration -v
        ;;

    "fast")
        echo -e "${BLUE}Running fast tests (excluding slow)...${NC}"
        pytest tests/ -m "not slow" -v
        ;;

    "coverage")
        echo -e "${BLUE}Running tests with coverage report...${NC}"
        pytest tests/ --cov=src --cov-report=html --cov-report=term
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;

    "coverage-parallel")
        echo -e "${BLUE}Running tests with coverage in parallel...${NC}"
        pytest tests/ -n auto --cov=src --cov-report=html --cov-report=term
        echo -e "${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;

    "watch")
        echo -e "${YELLOW}Watching for changes and re-running tests...${NC}"
        pytest-watch tests/ -v
        ;;

    "verbose")
        echo -e "${BLUE}Running tests with verbose output...${NC}"
        pytest tests/ -vv --tb=long
        ;;

    "quick")
        echo -e "${BLUE}Running quick crawler tests (parallel)...${NC}"
        pytest tests/test_rag/test_crawl_tracker.py -n auto -v
        ;;

    *)
        echo -e "${YELLOW}Usage: ./run_tests.sh [mode]${NC}"
        echo ""
        echo "Available modes:"
        echo "  all                - Run all tests (default)"
        echo "  parallel           - Run all tests in parallel (auto CPUs)"
        echo "  parallel-4         - Run tests with 4 workers"
        echo "  crawler            - Run crawler tests only"
        echo "  crawler-parallel   - Run crawler tests in parallel"
        echo "  unit               - Run unit tests only"
        echo "  integration        - Run integration tests only"
        echo "  fast               - Run fast tests (exclude slow)"
        echo "  coverage           - Run with coverage report"
        echo "  coverage-parallel  - Run coverage in parallel"
        echo "  verbose            - Run with verbose output"
        echo "  quick              - Quick crawler tests (parallel)"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh                    # Run all tests"
        echo "  ./run_tests.sh parallel           # Parallel execution"
        echo "  ./run_tests.sh crawler-parallel   # Parallel crawler tests"
        echo "  ./run_tests.sh coverage           # With coverage"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✓ Tests completed${NC}"
