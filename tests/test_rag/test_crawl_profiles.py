"""Tests for documentation crawl profiles."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.rag.crawl_profiles import (
    CrawlProfile,
    ProfileManager,
    get_profile_manager
)


class TestCrawlProfile:
    """Tests for CrawlProfile dataclass."""

    def test_profile_with_urls(self):
        """Test profile with URL list."""
        profile = CrawlProfile(
            name="test",
            description="Test profile",
            urls=["https://example.com/doc1", "https://example.com/doc2"]
        )

        assert profile.name == "test"
        assert profile.description == "Test profile"
        assert len(profile.urls) == 2
        assert profile.sitemap_url is None
        assert profile.max_concurrent == 5

    def test_profile_with_sitemap(self):
        """Test profile with sitemap URL."""
        profile = CrawlProfile(
            name="test",
            description="Test profile",
            sitemap_url="https://example.com/sitemap.xml",
            url_filter="/docs/"
        )

        assert profile.name == "test"
        assert profile.sitemap_url == "https://example.com/sitemap.xml"
        assert profile.url_filter == "/docs/"
        assert profile.urls is None

    def test_profile_with_custom_concurrent(self):
        """Test profile with custom max_concurrent."""
        profile = CrawlProfile(
            name="test",
            description="Test profile",
            urls=["https://example.com"],
            max_concurrent=10
        )

        assert profile.max_concurrent == 10

    def test_profile_validation_error(self):
        """Test profile validation requires at least one URL source."""
        with pytest.raises(ValueError, match="must have either urls or sitemap_url"):
            CrawlProfile(
                name="invalid",
                description="Invalid profile"
            )


class TestProfileManager:
    """Tests for ProfileManager class."""

    def test_initialization_creates_default_profiles(self, tmp_path):
        """Test ProfileManager creates default profiles on first use."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        # Should create default profiles
        profiles = manager.list_profiles()
        assert len(profiles) > 0

        # Check some expected defaults
        profile_names = [p.name for p in profiles]
        assert "fastapi" in profile_names
        assert "django" in profile_names
        assert "react" in profile_names

        # Should save to file
        assert profiles_file.exists()

    def test_initialization_loads_existing_profiles(self, tmp_path):
        """Test ProfileManager loads profiles from existing file."""
        profiles_file = tmp_path / "profiles.json"

        # Create a custom profiles file
        custom_profiles = {
            "version": "1.0",
            "profiles": [
                {
                    "name": "custom",
                    "description": "Custom profile",
                    "urls": ["https://example.com"],
                    "sitemap_url": None,
                    "url_filter": None,
                    "max_concurrent": 5
                }
            ]
        }

        with open(profiles_file, 'w') as f:
            json.dump(custom_profiles, f)

        # Load manager
        manager = ProfileManager(profiles_file=profiles_file)

        # Should load custom profile
        profiles = manager.list_profiles()
        assert len(profiles) == 1
        assert profiles[0].name == "custom"

    def test_get_profile_found(self, tmp_path):
        """Test getting an existing profile."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        profile = manager.get_profile("fastapi")
        assert profile is not None
        assert profile.name == "fastapi"
        assert "FastAPI" in profile.description

    def test_get_profile_not_found(self, tmp_path):
        """Test getting a non-existent profile."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        profile = manager.get_profile("nonexistent")
        assert profile is None

    def test_list_profiles(self, tmp_path):
        """Test listing all profiles."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        profiles = manager.list_profiles()
        assert len(profiles) > 0
        # Should be sorted by name
        names = [p.name for p in profiles]
        assert names == sorted(names)

    def test_add_profile(self, tmp_path):
        """Test adding a new profile."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        new_profile = CrawlProfile(
            name="custom",
            description="Custom test profile",
            urls=["https://example.com/doc1"]
        )

        result = manager.add_profile(new_profile)
        assert result is True

        # Should be retrievable
        retrieved = manager.get_profile("custom")
        assert retrieved is not None
        assert retrieved.name == "custom"
        assert retrieved.description == "Custom test profile"

        # Should be saved to file
        assert profiles_file.exists()

    def test_update_existing_profile(self, tmp_path):
        """Test updating an existing profile."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        # Get existing profile
        original = manager.get_profile("fastapi")
        original_desc = original.description

        # Update it
        updated = CrawlProfile(
            name="fastapi",
            description="Updated description",
            sitemap_url=original.sitemap_url,
            url_filter=original.url_filter
        )

        result = manager.add_profile(updated)
        assert result is True

        # Should have new description
        retrieved = manager.get_profile("fastapi")
        assert retrieved.description == "Updated description"
        assert retrieved.description != original_desc

    def test_remove_profile(self, tmp_path):
        """Test removing a profile."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        # Ensure profile exists
        assert manager.get_profile("fastapi") is not None

        # Remove it
        result = manager.remove_profile("fastapi")
        assert result is True

        # Should be gone
        assert manager.get_profile("fastapi") is None

    def test_remove_nonexistent_profile(self, tmp_path):
        """Test removing a profile that doesn't exist."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        result = manager.remove_profile("nonexistent")
        assert result is False

    def test_get_stats(self, tmp_path):
        """Test getting profile statistics."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        stats = manager.get_stats()

        assert "total_profiles" in stats
        assert "sitemap_based" in stats
        assert "url_list_based" in stats
        assert "profile_names" in stats

        assert stats["total_profiles"] > 0
        assert isinstance(stats["profile_names"], list)
        assert stats["profile_names"] == sorted(stats["profile_names"])

    def test_default_profiles_have_required_fields(self, tmp_path):
        """Test that all default profiles have required fields."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        for profile in manager.list_profiles():
            assert profile.name
            assert profile.description
            assert profile.max_concurrent > 0
            # Must have either urls or sitemap_url
            assert profile.urls or profile.sitemap_url

    def test_sitemap_profiles_have_sitemap_url(self, tmp_path):
        """Test sitemap-based profiles."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        stats = manager.get_stats()
        sitemap_count = stats["sitemap_based"]

        assert sitemap_count > 0

        # Check FastAPI specifically (should be sitemap-based)
        fastapi = manager.get_profile("fastapi")
        assert fastapi.sitemap_url is not None
        assert "/tutorial/" in fastapi.url_filter

    def test_url_list_profiles_have_urls(self, tmp_path):
        """Test URL list-based profiles."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        stats = manager.get_stats()
        url_list_count = stats["url_list_based"]

        assert url_list_count > 0

        # Check Django specifically (should be URL list-based)
        django = manager.get_profile("django")
        assert django.urls is not None
        assert len(django.urls) > 0
        assert all("django" in url.lower() for url in django.urls)

    def test_save_and_reload(self, tmp_path):
        """Test saving and reloading profiles."""
        profiles_file = tmp_path / "profiles.json"

        # Create manager and add custom profile
        manager1 = ProfileManager(profiles_file=profiles_file)
        custom = CrawlProfile(
            name="test-reload",
            description="Test reload",
            urls=["https://example.com"]
        )
        manager1.add_profile(custom)

        # Create new manager instance (should load from file)
        manager2 = ProfileManager(profiles_file=profiles_file)

        # Should have the custom profile
        retrieved = manager2.get_profile("test-reload")
        assert retrieved is not None
        assert retrieved.description == "Test reload"

    def test_json_file_structure(self, tmp_path):
        """Test the JSON file has correct structure."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        # Load raw JSON
        with open(profiles_file, 'r') as f:
            data = json.load(f)

        assert "version" in data
        assert "profiles" in data
        assert isinstance(data["profiles"], list)
        assert len(data["profiles"]) > 0

        # Check profile structure
        for profile_dict in data["profiles"]:
            assert "name" in profile_dict
            assert "description" in profile_dict
            assert "max_concurrent" in profile_dict


class TestProfileManagerSingleton:
    """Tests for global ProfileManager singleton."""

    def test_get_profile_manager_singleton(self):
        """Test get_profile_manager returns singleton."""
        manager1 = get_profile_manager()
        manager2 = get_profile_manager()

        # Should be the same instance
        assert manager1 is manager2

    def test_singleton_persists_changes(self):
        """Test changes persist across get_profile_manager calls."""
        manager1 = get_profile_manager()

        # Add a test profile
        test_profile = CrawlProfile(
            name="test-singleton",
            description="Test singleton persistence",
            urls=["https://example.com"]
        )
        manager1.add_profile(test_profile)

        # Get manager again
        manager2 = get_profile_manager()

        # Should have the test profile
        retrieved = manager2.get_profile("test-singleton")
        assert retrieved is not None
        assert retrieved.description == "Test singleton persistence"

        # Cleanup
        manager2.remove_profile("test-singleton")


class TestSpecificProfiles:
    """Tests for specific default profiles."""

    def test_fastapi_profile(self, tmp_path):
        """Test FastAPI profile configuration."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        fastapi = manager.get_profile("fastapi")
        assert fastapi is not None
        assert fastapi.name == "fastapi"
        assert "FastAPI" in fastapi.description
        assert fastapi.sitemap_url == "https://fastapi.tiangolo.com/sitemap.xml"
        assert fastapi.url_filter == "/tutorial/"

    def test_django_profile(self, tmp_path):
        """Test Django profile configuration."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        django = manager.get_profile("django")
        assert django is not None
        assert django.name == "django"
        assert "Django" in django.description
        assert django.urls is not None
        assert len(django.urls) > 0
        # Check it has key docs
        url_paths = [url for url in django.urls]
        assert any("models" in url for url in url_paths)
        assert any("views" in url for url in url_paths)

    def test_python_stdlib_profile(self, tmp_path):
        """Test Python stdlib profile configuration."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        python = manager.get_profile("python-stdlib")
        assert python is not None
        assert python.name == "python-stdlib"
        assert "Python" in python.description
        assert python.sitemap_url == "https://docs.python.org/3/sitemap.xml"
        assert python.url_filter == "/library/"

    def test_langchain_profile(self, tmp_path):
        """Test LangChain profile configuration."""
        profiles_file = tmp_path / "profiles.json"
        manager = ProfileManager(profiles_file=profiles_file)

        langchain = manager.get_profile("langchain")
        assert langchain is not None
        assert langchain.name == "langchain"
        assert "LangChain" in langchain.description
        assert langchain.sitemap_url == "https://python.langchain.com/sitemap.xml"
        assert langchain.url_filter == "/docs/"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
