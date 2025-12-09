"""Unit tests for CrawlTracker module.

These tests are designed to run in parallel using pytest-xdist.
Each test is isolated and uses temporary directories.
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from src.rag.crawl_tracker import CrawlTracker, CrawlRecord


@pytest.fixture
def temp_tracker_file(tmp_path):
    """Provide a temporary tracker file for each test."""
    tracker_file = tmp_path / "crawled_urls.json"
    yield tracker_file
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def tracker(temp_tracker_file):
    """Provide an isolated CrawlTracker instance."""
    return CrawlTracker(tracker_file=temp_tracker_file)


class TestCrawlRecord:
    """Tests for CrawlRecord dataclass."""

    def test_create_record(self):
        """Test creating a CrawlRecord."""
        record = CrawlRecord(
            url="https://example.com",
            crawl_date="2024-12-08T10:00:00",
            content_hash="abc123",
            chunk_count=10,
            file_path="/path/to/file.md",
            title="Example Page"
        )

        assert record.url == "https://example.com"
        assert record.chunk_count == 10
        assert record.title == "Example Page"

    def test_to_dict(self):
        """Test converting record to dictionary."""
        record = CrawlRecord(
            url="https://example.com",
            crawl_date="2024-12-08T10:00:00",
            content_hash="abc123",
            chunk_count=10,
            file_path="/path/to/file.md",
            title="Example"
        )

        record_dict = record.to_dict()

        assert isinstance(record_dict, dict)
        assert record_dict["url"] == "https://example.com"
        assert record_dict["chunk_count"] == 10

    def test_from_dict(self):
        """Test creating record from dictionary."""
        data = {
            "url": "https://example.com",
            "crawl_date": "2024-12-08T10:00:00",
            "content_hash": "abc123",
            "chunk_count": 10,
            "file_path": "/path/to/file.md",
            "title": "Example",
            "content_length": 1000
        }

        record = CrawlRecord.from_dict(data)

        assert record.url == "https://example.com"
        assert record.chunk_count == 10


class TestCrawlTrackerInit:
    """Tests for CrawlTracker initialization."""

    def test_init_with_custom_file(self, temp_tracker_file):
        """Test initialization with custom tracker file."""
        tracker = CrawlTracker(tracker_file=temp_tracker_file)

        assert tracker.tracker_file == temp_tracker_file
        assert isinstance(tracker.records, dict)
        assert len(tracker.records) == 0

    def test_init_creates_parent_directory(self, tmp_path):
        """Test that parent directory is created when saving."""
        tracker_file = tmp_path / "subdir" / "tracker.json"
        tracker = CrawlTracker(tracker_file=tracker_file)

        # Add a record to trigger save
        tracker.add_record(
            url="https://example.com",
            content="test",
            chunk_count=1,
            file_path="/test.md"
        )

        assert tracker_file.parent.exists()
        assert tracker_file.exists()

    def test_load_existing_records(self, temp_tracker_file):
        """Test loading existing records from file."""
        # Create a tracker file with existing data
        existing_data = {
            "https://example.com": {
                "url": "https://example.com",
                "crawl_date": "2024-12-08T10:00:00",
                "content_hash": "abc123",
                "chunk_count": 5,
                "file_path": "/test.md",
                "title": "Test",
                "content_length": 100
            }
        }

        temp_tracker_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_tracker_file, 'w') as f:
            json.dump(existing_data, f)

        # Load tracker
        tracker = CrawlTracker(tracker_file=temp_tracker_file)

        assert len(tracker.records) == 1
        assert "https://example.com" in tracker.records
        assert tracker.records["https://example.com"].chunk_count == 5


class TestContentHashing:
    """Tests for content hashing."""

    def test_compute_content_hash(self):
        """Test computing MD5 hash of content."""
        content = "Hello, World!"
        hash1 = CrawlTracker.compute_content_hash(content)

        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hex length

    def test_same_content_same_hash(self):
        """Test that same content produces same hash."""
        content = "Test content"

        hash1 = CrawlTracker.compute_content_hash(content)
        hash2 = CrawlTracker.compute_content_hash(content)

        assert hash1 == hash2

    def test_different_content_different_hash(self):
        """Test that different content produces different hash."""
        hash1 = CrawlTracker.compute_content_hash("content1")
        hash2 = CrawlTracker.compute_content_hash("content2")

        assert hash1 != hash2


class TestCrawlDetection:
    """Tests for crawl detection methods."""

    def test_is_crawled_false_for_new_url(self, tracker):
        """Test that new URL is not marked as crawled."""
        assert not tracker.is_crawled("https://new.com")

    def test_is_crawled_true_after_adding(self, tracker):
        """Test that URL is marked as crawled after adding."""
        tracker.add_record(
            url="https://example.com",
            content="test",
            chunk_count=1,
            file_path="/test.md"
        )

        assert tracker.is_crawled("https://example.com")

    def test_has_changed_true_for_new_url(self, tracker):
        """Test that new URL is marked as changed."""
        assert tracker.has_changed("https://new.com", "any content")

    def test_has_changed_false_for_same_content(self, tracker):
        """Test that same content is not marked as changed."""
        content = "Original content"
        tracker.add_record(
            url="https://example.com",
            content=content,
            chunk_count=1,
            file_path="/test.md"
        )

        assert not tracker.has_changed("https://example.com", content)

    def test_has_changed_true_for_different_content(self, tracker):
        """Test that different content is marked as changed."""
        tracker.add_record(
            url="https://example.com",
            content="Original content",
            chunk_count=1,
            file_path="/test.md"
        )

        assert tracker.has_changed("https://example.com", "Updated content")


class TestRecordManagement:
    """Tests for adding, getting, and removing records."""

    def test_add_record(self, tracker):
        """Test adding a new record."""
        record = tracker.add_record(
            url="https://example.com",
            content="Test content",
            chunk_count=5,
            file_path="/test.md",
            title="Test Page"
        )

        assert isinstance(record, CrawlRecord)
        assert record.url == "https://example.com"
        assert record.chunk_count == 5
        assert record.title == "Test Page"

    def test_add_record_persists_to_file(self, tracker, temp_tracker_file):
        """Test that adding record saves to file."""
        tracker.add_record(
            url="https://example.com",
            content="test",
            chunk_count=1,
            file_path="/test.md"
        )

        assert temp_tracker_file.exists()

        # Verify file content
        with open(temp_tracker_file, 'r') as f:
            data = json.load(f)

        assert "https://example.com" in data

    def test_update_existing_record(self, tracker):
        """Test updating an existing record."""
        # Add initial record
        tracker.add_record(
            url="https://example.com",
            content="Original",
            chunk_count=5,
            file_path="/test1.md"
        )

        # Update record
        tracker.add_record(
            url="https://example.com",
            content="Updated",
            chunk_count=10,
            file_path="/test2.md"
        )

        record = tracker.get_record("https://example.com")
        assert record.chunk_count == 10
        assert record.file_path == "/test2.md"

    def test_get_record_existing(self, tracker):
        """Test getting an existing record."""
        tracker.add_record(
            url="https://example.com",
            content="test",
            chunk_count=3,
            file_path="/test.md"
        )

        record = tracker.get_record("https://example.com")

        assert record is not None
        assert record.url == "https://example.com"
        assert record.chunk_count == 3

    def test_get_record_nonexistent(self, tracker):
        """Test getting a nonexistent record."""
        record = tracker.get_record("https://nonexistent.com")

        assert record is None

    def test_remove_record_existing(self, tracker):
        """Test removing an existing record."""
        tracker.add_record(
            url="https://example.com",
            content="test",
            chunk_count=1,
            file_path="/test.md"
        )

        result = tracker.remove_record("https://example.com")

        assert result is True
        assert not tracker.is_crawled("https://example.com")

    def test_remove_record_nonexistent(self, tracker):
        """Test removing a nonexistent record."""
        result = tracker.remove_record("https://nonexistent.com")

        assert result is False

    def test_clear_all(self, tracker):
        """Test clearing all records."""
        # Add multiple records
        tracker.add_record("https://example1.com", "test1", 1, "/test1.md")
        tracker.add_record("https://example2.com", "test2", 2, "/test2.md")

        tracker.clear_all()

        assert len(tracker.records) == 0

    def test_get_all_records_empty(self, tracker):
        """Test getting all records when empty."""
        records = tracker.get_all_records()

        assert isinstance(records, list)
        assert len(records) == 0

    def test_get_all_records_sorted_by_date(self, tracker):
        """Test that records are sorted by date (newest first)."""
        # Add records with different dates
        tracker.records = {
            "url1": CrawlRecord("url1", "2024-12-08T10:00:00", "hash1", 1, "/1.md"),
            "url2": CrawlRecord("url2", "2024-12-08T12:00:00", "hash2", 2, "/2.md"),
            "url3": CrawlRecord("url3", "2024-12-08T11:00:00", "hash3", 3, "/3.md")
        }

        records = tracker.get_all_records()

        assert len(records) == 3
        assert records[0].crawl_date == "2024-12-08T12:00:00"  # Newest first
        assert records[1].crawl_date == "2024-12-08T11:00:00"
        assert records[2].crawl_date == "2024-12-08T10:00:00"


class TestStatistics:
    """Tests for statistics methods."""

    def test_get_stats_empty(self, tracker):
        """Test statistics for empty tracker."""
        stats = tracker.get_stats()

        assert stats["total_urls"] == 0
        assert stats["total_chunks"] == 0
        assert stats["total_content_bytes"] == 0
        assert stats["avg_chunks_per_url"] == 0

    def test_get_stats_with_records(self, tracker):
        """Test statistics with multiple records."""
        tracker.add_record("https://url1.com", "a" * 100, 5, "/1.md")
        tracker.add_record("https://url2.com", "b" * 200, 10, "/2.md")
        tracker.add_record("https://url3.com", "c" * 300, 15, "/3.md")

        stats = tracker.get_stats()

        assert stats["total_urls"] == 3
        assert stats["total_chunks"] == 30  # 5 + 10 + 15
        assert stats["total_content_bytes"] == 600  # 100 + 200 + 300
        assert stats["avg_chunks_per_url"] == 10  # 30 / 3


class TestPersistence:
    """Tests for file persistence."""

    def test_save_and_load_roundtrip(self, temp_tracker_file):
        """Test saving and loading records."""
        # Create tracker and add records
        tracker1 = CrawlTracker(tracker_file=temp_tracker_file)
        tracker1.add_record("https://example.com", "content", 5, "/test.md", "Title")

        # Create new tracker instance and load
        tracker2 = CrawlTracker(tracker_file=temp_tracker_file)

        assert len(tracker2.records) == 1
        record = tracker2.get_record("https://example.com")
        assert record.chunk_count == 5
        assert record.title == "Title"

    def test_handle_corrupted_file_gracefully(self, temp_tracker_file):
        """Test handling of corrupted tracker file."""
        # Write invalid JSON
        temp_tracker_file.parent.mkdir(parents=True, exist_ok=True)
        temp_tracker_file.write_text("invalid json {{{")

        # Should not crash, just start with empty records
        tracker = CrawlTracker(tracker_file=temp_tracker_file)

        assert len(tracker.records) == 0


@pytest.mark.parametrize("url,content,chunks", [
    ("https://docs.python.org", "Python documentation", 10),
    ("https://fastapi.tiangolo.com", "FastAPI docs", 25),
    ("https://github.com/example", "GitHub README", 5),
])
def test_add_multiple_different_urls(tracker, url, content, chunks):
    """Test adding different URLs (parameterized for parallel execution)."""
    record = tracker.add_record(url, content, chunks, "/test.md")

    assert record.url == url
    assert record.chunk_count == chunks
    assert tracker.is_crawled(url)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-n", "auto"])
