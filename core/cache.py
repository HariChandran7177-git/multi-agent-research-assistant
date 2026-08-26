"""
Unified caching layer supporting both SQLite (local) and Redis (production).
Uses SQLite by default, falls back to Redis if enabled and available.
"""

import sqlite3
import os
import json
from typing import Optional, Dict, Any
from core.logger import get_logger

logger = get_logger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "cache.sqlite")


def _get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS query_cache (
            cache_key TEXT PRIMARY KEY,
            data TEXT
        )
    ''')
    conn.commit()
    return conn


class CacheManager:
    """SQLite-backed cache manager with optional Redis fallback."""

    def __init__(self):
        self.conn = _get_conn()
        self.redis_client = None
        self.enabled = True

        # Try to initialize Redis if enabled
        self._init_redis()

    def _init_redis(self):
        """Initialize Redis client if REDIS_ENABLED is true."""
        try:
            from core.config import REDIS_URL, REDIS_ENABLED
            if REDIS_ENABLED:
                import redis
                self.redis_client = redis.from_url(REDIS_URL)
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
        except ImportError:
            logger.debug("Redis not installed, using SQLite only")
        except Exception as e:
            logger.warning(f"Redis connection failed, using SQLite: {e}")

    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result for query (async wrapper)."""
        cache_key = self._get_key(query)

        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.debug(f"Redis get failed: {e}")

        cursor = self.conn.cursor()
        cursor.execute("SELECT data FROM query_cache WHERE cache_key = ?", (cache_key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    async def set(self, query: str, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache result for query (async wrapper)."""
        cache_key = self._get_key(query)

        if self.redis_client:
            try:
                self.redis_client.setex(cache_key, ttl, json.dumps(data))
            except Exception as e:
                logger.debug(f"Redis set failed: {e}")

        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO query_cache (cache_key, data) VALUES (?, ?)",
            (cache_key, json.dumps(data))
        )
        self.conn.commit()
        return True

    def _get_key(self, query: str) -> str:
        """Generate cache key from query."""
        normalized = " ".join(query.strip().lower().split())
        import hashlib
        hash_key = hashlib.sha256(normalized.encode()).hexdigest()[:16]
        return f"research:{hash_key}"

    async def clear(self):
        """Clear all cache entries."""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.debug(f"Redis flush failed: {e}")

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM query_cache")
        self.conn.commit()
        logger.info("Cache cleared")


# Global cache instance
cache = CacheManager()
