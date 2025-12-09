"""Crawl URL tracking and deduplication for web documentation.

This module tracks crawled URLs to prevent duplicate crawling and
provide visibility into what documentation has been indexed.
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from ..config.settings import settings

logger = logging.getLogger("ai_agent.crawl_tracker")


@dataclass
class CrawlRecord:
    """Record of a crawled URL."""
    url: str
    crawl_date: str  # ISO format datetime
    content_hash: str  # MD5 hash of markdown content
    chunk_count: int
    file_path: str
    title: str = ""
    content_length: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'CrawlRecord':
        """Create from dictionary."""
        return cls(**data)


class CrawlTracker:
    """Track crawled URLs and prevent duplicates."""

    def __init__(self, tracker_file: Optional[Path] = None):
        """Initialize the crawl tracker.

        Args:
            tracker_file: Path to JSON file storing crawl records.
                         Defaults to ~/.ai-agent/crawled_urls.json
        """
        if tracker_file is None:
            base_path = Path(settings.CRAWLED_DOCS_PATH).expanduser()
            tracker_file = base_path / "crawled_urls.json"

        self.tracker_file = tracker_file
        self.records: Dict[str, CrawlRecord] = {}
        self._load_records()

    def _load_records(self) -> None:
        """Load crawl records from JSON file."""
        if not self.tracker_file.exists():
            logger.debug(f"No tracker file found at {self.tracker_file}")
            return

        try:
            with open(self.tracker_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.records = {
                url: CrawlRecord.from_dict(record)
                for url, record in data.items()
            }

            logger.info(f"Loaded {len(self.records)} crawl records from {self.tracker_file}")

        except Exception as e:
            logger.error(f"Failed to load tracker file: {e}")
            self.records = {}

    def _save_records(self) -> None:
        """Save crawl records to JSON file."""
        try:
            # Ensure directory exists
            self.tracker_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert records to dict
            data = {
                url: record.to_dict()
                for url, record in self.records.items()
            }

            # Save to file
            with open(self.tracker_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(self.records)} records to {self.tracker_file}")

        except Exception as e:
            logger.error(f"Failed to save tracker file: {e}")

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute MD5 hash of content.

        Args:
            content: Text content to hash

        Returns:
            MD5 hash as hex string
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def is_crawled(self, url: str) -> bool:
        """Check if URL has been crawled before.

        Args:
            url: URL to check

        Returns:
            True if URL has been crawled
        """
        return url in self.records

    def has_changed(self, url: str, content: str) -> bool:
        """Check if content has changed since last crawl.

        Args:
            url: URL to check
            content: Current content to compare

        Returns:
            True if content has changed or URL not crawled yet
        """
        if not self.is_crawled(url):
            return True

        current_hash = self.compute_content_hash(content)
        previous_hash = self.records[url].content_hash

        return current_hash != previous_hash

    def get_record(self, url: str) -> Optional[CrawlRecord]:
        """Get crawl record for a URL.

        Args:
            url: URL to get record for

        Returns:
            CrawlRecord if found, None otherwise
        """
        return self.records.get(url)

    def add_record(
        self,
        url: str,
        content: str,
        chunk_count: int,
        file_path: str,
        title: str = ""
    ) -> CrawlRecord:
        """Add or update a crawl record.

        Args:
            url: URL that was crawled
            content: Markdown content
            chunk_count: Number of chunks extracted
            file_path: Path to saved markdown file
            title: Page title

        Returns:
            The created/updated CrawlRecord
        """
        content_hash = self.compute_content_hash(content)
        crawl_date = datetime.utcnow().isoformat()

        record = CrawlRecord(
            url=url,
            crawl_date=crawl_date,
            content_hash=content_hash,
            chunk_count=chunk_count,
            file_path=file_path,
            title=title,
            content_length=len(content)
        )

        self.records[url] = record
        self._save_records()

        logger.info(f"Added crawl record for {url} ({chunk_count} chunks)")
        return record

    def get_all_records(self) -> List[CrawlRecord]:
        """Get all crawl records sorted by date (newest first).

        Returns:
            List of CrawlRecords
        """
        records = list(self.records.values())
        records.sort(key=lambda r: r.crawl_date, reverse=True)
        return records

    def remove_record(self, url: str) -> bool:
        """Remove a crawl record.

        Args:
            url: URL to remove

        Returns:
            True if record was removed, False if not found
        """
        if url in self.records:
            del self.records[url]
            self._save_records()
            logger.info(f"Removed crawl record for {url}")
            return True
        return False

    def clear_all(self) -> None:
        """Clear all crawl records."""
        self.records.clear()
        self._save_records()
        logger.info("Cleared all crawl records")

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about crawled URLs.

        Returns:
            Dictionary with statistics
        """
        total_chunks = sum(r.chunk_count for r in self.records.values())
        total_content = sum(r.content_length for r in self.records.values())

        return {
            "total_urls": len(self.records),
            "total_chunks": total_chunks,
            "total_content_bytes": total_content,
            "avg_chunks_per_url": total_chunks // len(self.records) if self.records else 0
        }


# Global tracker instance (singleton pattern)
_tracker_instance: Optional[CrawlTracker] = None


def get_crawl_tracker() -> CrawlTracker:
    """Get the global CrawlTracker instance.

    Returns:
        CrawlTracker singleton instance
    """
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = CrawlTracker()
    return _tracker_instance
