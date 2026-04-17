"""Disk-based caching for API responses."""

import hashlib
import os
from typing import Any, Optional

import diskcache

from toolblind.utils.config import get_config
from toolblind.utils.logging import get_logger

logger = get_logger("cache")


class ResponseCache:
    """Disk-backed cache for API responses keyed by model + prompt hash."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the disk cache."""
        if cache_dir is None:
            cache_dir = get_config().cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._cache = diskcache.Cache(cache_dir, size_limit=10 * 1024 * 1024 * 1024)

    @staticmethod
    def _make_key(model: str, prompt_data: str) -> str:
        """Create a deterministic cache key from model name and prompt content."""
        raw = f"{model}::{prompt_data}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, model: str, prompt_data: str) -> Optional[Any]:
        """Retrieve a cached response, or None if not found."""
        key = self._make_key(model, prompt_data)
        result = self._cache.get(key)
        if result is not None:
            logger.debug(f"Cache hit for {model} key={key[:12]}...")
        return result

    def put(self, model: str, prompt_data: str, response: Any) -> None:
        """Store a response in the cache."""
        key = self._make_key(model, prompt_data)
        self._cache.set(key, response)
        logger.debug(f"Cached response for {model} key={key[:12]}...")

    def clear(self) -> None:
        """Clear all cached responses."""
        self._cache.clear()

    def close(self) -> None:
        """Close the cache."""
        self._cache.close()


_cache_instance: Optional[ResponseCache] = None


def get_cache() -> ResponseCache:
    """Get or create the global cache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ResponseCache()
    return _cache_instance
