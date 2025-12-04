#!/usr/bin/env python3
"""
AI Coding Agent - Entry Point

Local MVP with cloud migration support.
"""

import os

# Remove shell environment variable override to allow .env to take precedence
if "LLM_PROVIDER" in os.environ:
    del os.environ["LLM_PROVIDER"]

from src.cli.main import main

if __name__ == "__main__":
    main()
