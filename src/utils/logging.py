"""Logging configuration for the AI Coding Agent."""

import logging
import sys
from pathlib import Path
from typing import Optional

from ..config.settings import settings


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: Optional[str] = None
) -> logging.Logger:
    """Setup a logger with console and optional file handlers.

    Args:
        name: Logger name (typically __name__)
        log_file: Optional file path for logging
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    # Get log level from settings or parameter
    log_level = getattr(logging, level or settings.LOG_LEVEL)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# Global logger for file operations
file_ops_logger = setup_logger(
    "ai_agent.file_ops",
    log_file=Path.home() / ".ai-agent" / "logs" / "file_operations.log" if settings.LOG_FILE_OPERATIONS else None
)

# Global logger for RAG operations
rag_logger = setup_logger(
    "ai_agent.rag",
    log_file=Path.home() / ".ai-agent" / "logs" / "rag.log" if settings.LOG_FILE_OPERATIONS else None
)

# Global logger for agent operations
agent_logger = setup_logger("ai_agent.core")
