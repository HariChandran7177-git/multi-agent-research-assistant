"""
Unified caching layer supporting SQLite (local).
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
    """SQLite-backed cache manager."""

    def __init__(self):
        self.conn = _get_conn()
        self.enabled = True

    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result for query (async wrapper)."""
        cache_key = self._get_key(query)

        cursor = self.conn.cursor()
        cursor.execute("SELECT data FROM query_cache WHERE cache_key = ?", (cache_key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None

    async def set(self, query: str, data: Dict[str, Any], ttl: int = 3600) -> bool:
        """Cache result for query (async wrapper)."""
        cache_key = self._get_key(query)

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
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM query_cache")
        self.conn.commit()
        logger.info("Cache cleared")


# Global cache instance
cache = CacheManager()
