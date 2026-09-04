"""
Health check utilities for all external services.
"""

import asyncio
import time
from typing import Dict, Any
from core.logger import get_logger
from core.config import (
    GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY,
    TAVILY_MAX_RESULTS
)

logger = get_logger(__name__)


class HealthChecker:
    """Checks health of all external dependencies."""

    def __init__(self):
        self.services = {
            "redis": {"status": "unknown", "latency_ms": None},
            "groq": {"status": "unknown", "latency_ms": None},
            "qdrant": {"status": "unknown", "latency_ms": None},

            "tavily": {"status": "unknown", "latency_ms": None},
        }

    async def check_redis(self) -> Dict[str, Any]:
        """Check Redis connection."""
        start = time.perf_counter()
        try:
            import redis
            from core.config import REDIS_URL, REDIS_ENABLED
            if not REDIS_ENABLED:
                return {"status": "disabled", "latency_ms": 0}

            client = redis.from_url(REDIS_URL)
            await asyncio.get_event_loop().run_in_executor(None, client.ping)
            latency = (time.perf_counter() - start) * 1000
            self.services["redis"] = {"status": "healthy", "latency_ms": round(latency, 2)}
            return {"status": "healthy", "latency_ms": round(latency, 2)}
        except ImportError:
            return {"status": "not_installed", "latency_ms": None}
        except Exception as e:
            self.services["redis"] = {"status": "degraded", "error": str(e)}
            return {"status": "degraded", "error": str(e)}

    async def check_groq(self) -> Dict[str, Any]:
        """Check Groq API availability."""
        start = time.perf_counter()
        try:
            if not GROQ_API_KEY:
                return {"status": "no_config", "error": "GROQ_API_KEY not set"}

            from langchain_groq import ChatGroq
            llm = ChatGroq(model="groq/compound-mini", api_key=GROQ_API_KEY)
            future = asyncio.get_event_loop().run_in_executor(
                None, llm.invoke, "hi"
            )
            await asyncio.wait_for(future, timeout=10)
            latency = (time.perf_counter() - start) * 1000
            self.services["groq"] = {"status": "healthy", "latency_ms": round(latency, 2)}
            return {"status": "healthy", "latency_ms": round(latency, 2)}
        except Exception as e:
            self.services["groq"] = {"status": "degraded", "error": str(e)}
            return {"status": "degraded", "error": str(e)}

    async def check_qdrant(self) -> Dict[str, Any]:
        """Check Qdrant connection."""
        start = time.perf_counter()
        try:
            if not QDRANT_URL:
                return {"status": "no_config", "error": "QDRANT_URL not set"}

            from qdrant_client import QdrantClient
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            await asyncio.get_event_loop().run_in_executor(None, client.get_collections)
            latency = (time.perf_counter() - start) * 1000
            self.services["qdrant"] = {"status": "healthy", "latency_ms": round(latency, 2)}
            return {"status": "healthy", "latency_ms": round(latency, 2)}
        except Exception as e:
            self.services["qdrant"] = {"status": "degraded", "error": str(e)}
            return {"status": "degraded", "error": str(e)}

    async def check_tavily(self) -> Dict[str, Any]:
        """Check Tavily API availability."""
        start = time.perf_counter()
        try:
            import os
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return {"status": "no_config", "error": "TAVILY_API_KEY not set"}

            from tavily import TavilyClient
            client = TavilyClient(api_key=api_key)
            future = asyncio.get_event_loop().run_in_executor(
                None, client.search, "test", 1
            )
            await asyncio.wait_for(future, timeout=10)
            latency = (time.perf_counter() - start) * 1000
            self.services["tavily"] = {"status": "healthy", "latency_ms": round(latency, 2)}
            return {"status": "healthy", "latency_ms": round(latency, 2)}
        except Exception as e:
            self.services["tavily"] = {"status": "degraded", "error": str(e)}
            return {"status": "degraded", "error": str(e)}

    async def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        results["redis"] = await self.check_redis()
        results["groq"] = await self.check_groq()
        results["qdrant"] = await self.check_qdrant()

        results["tavily"] = await self.check_tavily()

        overall = all(s["status"] == "healthy" for s in results.values())
        return {
            "status": "healthy" if overall else "degraded",
            "timestamp": time.time(),
            "services": results,
        }


# Global health checker
health_checker = HealthChecker()
