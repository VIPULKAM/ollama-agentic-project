"""Documentation crawling profiles for popular frameworks.

This module provides pre-configured profiles for crawling documentation
from popular frameworks and libraries. Profiles can be used via CLI
commands for quick, standardized documentation crawling.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from ..config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class CrawlProfile:
    """Profile for crawling a specific documentation site.

    Attributes:
        name: Unique profile identifier (e.g., "fastapi", "django")
        description: Human-readable description of what gets crawled
        urls: List of specific URLs to crawl (optional)
        sitemap_url: URL to sitemap.xml for auto-discovery (optional)
        url_filter: Filter pattern for sitemap URLs (optional)
        max_concurrent: Max concurrent crawls (default: 5)
    """
    name: str
    description: str
    urls: Optional[List[str]] = None
    sitemap_url: Optional[str] = None
    url_filter: Optional[str] = None
    max_concurrent: int = 5

    def __post_init__(self):
        """Validate that at least one URL source is provided."""
        if not self.urls and not self.sitemap_url:
            raise ValueError(f"Profile '{self.name}' must have either urls or sitemap_url")


class ProfileManager:
    """Manage documentation crawling profiles."""

    def __init__(self, profiles_file: Optional[Path] = None):
        """Initialize profile manager.

        Args:
            profiles_file: Path to profiles JSON file (default: ~/.ai-agent/crawl_profiles.json)
        """
        if profiles_file is None:
            # Store in crawled_docs directory alongside other crawl data
            crawled_docs_path = Path(settings.CRAWLED_DOCS_PATH)
            crawled_docs_path.mkdir(parents=True, exist_ok=True)
            profiles_file = crawled_docs_path / "crawl_profiles.json"

        self.profiles_file = profiles_file
        self._profiles: Dict[str, CrawlProfile] = {}

        # Load existing profiles or create default ones
        if self.profiles_file.exists():
            self._load_profiles()
        else:
            self._create_default_profiles()
            self._save_profiles()

    def _create_default_profiles(self):
        """Create default profiles for popular frameworks."""

        # FastAPI - Use sitemap with filter
        self._profiles["fastapi"] = CrawlProfile(
            name="fastapi",
            description="FastAPI web framework documentation",
            sitemap_url="https://fastapi.tiangolo.com/sitemap.xml",
            url_filter="/tutorial/",
            max_concurrent=5
        )

        # Django - Key documentation pages
        self._profiles["django"] = CrawlProfile(
            name="django",
            description="Django web framework core documentation",
            urls=[
                "https://docs.djangoproject.com/en/stable/topics/db/models/",
                "https://docs.djangoproject.com/en/stable/topics/db/queries/",
                "https://docs.djangoproject.com/en/stable/topics/http/views/",
                "https://docs.djangoproject.com/en/stable/topics/http/urls/",
                "https://docs.djangoproject.com/en/stable/topics/forms/",
                "https://docs.djangoproject.com/en/stable/topics/class-based-views/",
                "https://docs.djangoproject.com/en/stable/topics/auth/",
                "https://docs.djangoproject.com/en/stable/ref/models/fields/",
                "https://docs.djangoproject.com/en/stable/ref/settings/",
            ],
            max_concurrent=5
        )

        # React - Main documentation
        self._profiles["react"] = CrawlProfile(
            name="react",
            description="React library documentation",
            sitemap_url="https://react.dev/sitemap.xml",
            url_filter="/learn/",
            max_concurrent=5
        )

        # Vue - Main documentation
        self._profiles["vue"] = CrawlProfile(
            name="vue",
            description="Vue.js framework documentation",
            urls=[
                "https://vuejs.org/guide/introduction.html",
                "https://vuejs.org/guide/essentials/reactivity-fundamentals.html",
                "https://vuejs.org/guide/components/registration.html",
                "https://vuejs.org/guide/components/props.html",
                "https://vuejs.org/guide/components/events.html",
                "https://vuejs.org/guide/reusability/composables.html",
                "https://vuejs.org/guide/scaling-up/routing.html",
                "https://vuejs.org/guide/scaling-up/state-management.html",
            ],
            max_concurrent=5
        )

        # TypeScript - Handbook
        self._profiles["typescript"] = CrawlProfile(
            name="typescript",
            description="TypeScript language handbook",
            sitemap_url="https://www.typescriptlang.org/sitemap.xml",
            url_filter="/docs/handbook/",
            max_concurrent=5
        )

        # Python - Standard library (filtered sitemap)
        self._profiles["python-stdlib"] = CrawlProfile(
            name="python-stdlib",
            description="Python standard library documentation",
            sitemap_url="https://docs.python.org/3/sitemap.xml",
            url_filter="/library/",
            max_concurrent=5
        )

        # Flask - Web framework
        self._profiles["flask"] = CrawlProfile(
            name="flask",
            description="Flask web framework documentation",
            urls=[
                "https://flask.palletsprojects.com/en/stable/quickstart/",
                "https://flask.palletsprojects.com/en/stable/tutorial/",
                "https://flask.palletsprojects.com/en/stable/api/",
                "https://flask.palletsprojects.com/en/stable/blueprints/",
                "https://flask.palletsprojects.com/en/stable/patterns/",
            ],
            max_concurrent=5
        )

        # Express.js - Node.js web framework
        self._profiles["express"] = CrawlProfile(
            name="express",
            description="Express.js web framework documentation",
            urls=[
                "https://expressjs.com/en/starter/installing.html",
                "https://expressjs.com/en/guide/routing.html",
                "https://expressjs.com/en/guide/writing-middleware.html",
                "https://expressjs.com/en/guide/using-middleware.html",
                "https://expressjs.com/en/guide/error-handling.html",
                "https://expressjs.com/en/guide/database-integration.html",
            ],
            max_concurrent=5
        )

        # PostgreSQL - Database documentation
        self._profiles["postgresql"] = CrawlProfile(
            name="postgresql",
            description="PostgreSQL database documentation (SQL commands)",
            sitemap_url="https://www.postgresql.org/docs/current/sitemap.xml",
            url_filter="/sql-",
            max_concurrent=3  # Lower for large docs
        )

        # LangChain - AI framework
        self._profiles["langchain"] = CrawlProfile(
            name="langchain",
            description="LangChain framework documentation",
            sitemap_url="https://python.langchain.com/sitemap.xml",
            url_filter="/docs/",
            max_concurrent=5
        )

        logger.info(f"Created {len(self._profiles)} default profiles")

    def _load_profiles(self):
        """Load profiles from JSON file."""
        try:
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)

            for profile_dict in data.get("profiles", []):
                profile = CrawlProfile(**profile_dict)
                self._profiles[profile.name] = profile

            logger.info(f"Loaded {len(self._profiles)} profiles from {self.profiles_file}")

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Failed to load profiles: {e}")
            # Fall back to default profiles
            self._create_default_profiles()

    def _save_profiles(self):
        """Save profiles to JSON file."""
        try:
            data = {
                "version": "1.0",
                "profiles": [asdict(p) for p in self._profiles.values()]
            }

            with open(self.profiles_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self._profiles)} profiles to {self.profiles_file}")

        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")

    def get_profile(self, name: str) -> Optional[CrawlProfile]:
        """Get a profile by name.

        Args:
            name: Profile name

        Returns:
            CrawlProfile if found, None otherwise
        """
        return self._profiles.get(name)

    def list_profiles(self) -> List[CrawlProfile]:
        """Get all available profiles.

        Returns:
            List of all profiles, sorted by name
        """
        return sorted(self._profiles.values(), key=lambda p: p.name)

    def add_profile(self, profile: CrawlProfile) -> bool:
        """Add or update a profile.

        Args:
            profile: Profile to add

        Returns:
            True if profile was added/updated
        """
        try:
            self._profiles[profile.name] = profile
            self._save_profiles()
            logger.info(f"Added profile: {profile.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add profile {profile.name}: {e}")
            return False

    def remove_profile(self, name: str) -> bool:
        """Remove a profile by name.

        Args:
            name: Profile name to remove

        Returns:
            True if profile was removed, False if not found
        """
        if name in self._profiles:
            del self._profiles[name]
            self._save_profiles()
            logger.info(f"Removed profile: {name}")
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get profile statistics.

        Returns:
            Dict with profile counts and info
        """
        profiles = list(self._profiles.values())

        sitemap_profiles = [p for p in profiles if p.sitemap_url]
        url_list_profiles = [p for p in profiles if p.urls]

        return {
            "total_profiles": len(profiles),
            "sitemap_based": len(sitemap_profiles),
            "url_list_based": len(url_list_profiles),
            "profile_names": sorted(self._profiles.keys())
        }


# Global singleton instance
_profile_manager: Optional[ProfileManager] = None


def get_profile_manager() -> ProfileManager:
    """Get global ProfileManager singleton.

    Returns:
        ProfileManager instance
    """
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager()
    return _profile_manager
