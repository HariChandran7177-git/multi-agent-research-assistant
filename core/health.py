"""
Health check utilities for all external services: Groq, Qdrant, Tavily, and Google Gemini.
"""

import asyncio
import time
import os
from typing import Dict, Any
from core.logger import get_logger
from core.config import (
    GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY,
    GEMINI_ROUTER_MODEL, GROQ_CRITIC_MODEL
)

logger = get_logger(__name__)


class HealthChecker:
    """Checks health of all external dependencies honestly."""

    def __init__(self):
        self.services = {
            "groq": {"status": "unknown", "latency_ms": None},
            "gemini": {"status": "unknown", "latency_ms": None},
            "qdrant": {"status": "unknown", "latency_ms": None},
            "tavily": {"status": "unknown", "latency_ms": None},
        }

    def check_shallow(self) -> Dict[str, Any]:
        """Lightweight health check — no external calls.
        Verifies that required env vars are set and the app is responsive.
        Safe for load-balancer polling every 10–30 s.
        """
        checks = {
            "groq_key_set": bool(GROQ_API_KEY),
            "gemini_key_set": bool(GEMINI_API_KEY),
            "tavily_key_set": bool(os.getenv("TAVILY_API_KEY")),
            "qdrant_url_set": bool(QDRANT_URL),
        }

        all_set = all(checks.values())
        return {
            "status": "healthy" if all_set else "unhealthy",
            "timestamp": time.time(),
            "config": checks,
        }

    async def check_groq(self) -> Dict[str, Any]:
        """Check Groq API availability (makes a real LLM call)."""
        start = time.perf_counter()
        try:
            if not GROQ_API_KEY:
                return {"status": "unhealthy", "error": "GROQ_API_KEY not set"}

            from langchain_groq import ChatGroq
            llm = ChatGroq(model=GROQ_CRITIC_MODEL, api_key=GROQ_API_KEY, timeout=10)
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, llm.invoke, "ping")
            await asyncio.wait_for(future, timeout=10)
            latency = (time.perf_counter() - start) * 1000
            result = {"status": "healthy", "latency_ms": round(latency, 2)}
            self.services["groq"] = result
            return result
        except Exception as e:
            result = {"status": "unhealthy", "error": str(e)}
            self.services["groq"] = result
            return result

    async def check_gemini(self) -> Dict[str, Any]:
        """Check Google Gemini API availability (makes a real LLM call)."""
        start = time.perf_counter()
        try:
            if not GEMINI_API_KEY:
                return {"status": "unhealthy", "error": "GOOGLE_API_KEY / GEMINI_API_KEY not set"}

            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_ROUTER_MODEL,
                google_api_key=GEMINI_API_KEY,
                temperature=0.1,
                timeout=10
            )
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, llm.invoke, "ping")
            await asyncio.wait_for(future, timeout=10)
            latency = (time.perf_counter() - start) * 1000
            result = {"status": "healthy", "latency_ms": round(latency, 2)}
            self.services["gemini"] = result
            return result
        except Exception as e:
            result = {"status": "unhealthy", "error": str(e)}
            self.services["gemini"] = result
            return result

    async def check_qdrant(self) -> Dict[str, Any]:
        """Check Qdrant connection."""
        start = time.perf_counter()
        try:
            if not QDRANT_URL:
                return {"status": "unhealthy", "error": "QDRANT_URL not set"}

            from qdrant_client import QdrantClient
            client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5, check_compatibility=False)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, client.get_collections)
            latency = (time.perf_counter() - start) * 1000
            result = {"status": "healthy", "latency_ms": round(latency, 2)}
            self.services["qdrant"] = result
            return result
        except Exception as e:
            result = {"status": "unhealthy", "error": str(e)}
            self.services["qdrant"] = result
            return result

    async def check_tavily(self) -> Dict[str, Any]:
        """Check Tavily API availability."""
        start = time.perf_counter()
        try:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                return {"status": "unhealthy", "error": "TAVILY_API_KEY not set"}

            from tavily import TavilyClient
            import functools
            client = TavilyClient(api_key=api_key)
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                None, functools.partial(client.search, query="ping", max_results=1)
            )
            await asyncio.wait_for(future, timeout=10)
            latency = (time.perf_counter() - start) * 1000
            result = {"status": "healthy", "latency_ms": round(latency, 2)}
            self.services["tavily"] = result
            return result
        except Exception as e:
            result = {"status": "unhealthy", "error": str(e)}
            self.services["tavily"] = result
            return result

    async def check_all_deep(self) -> Dict[str, Any]:
        """Full deep check — verifies all external dependencies.
        Returns status 'healthy' only if all services pass, otherwise 'unhealthy'.
        """
        results = {}
        # Run all health checks in parallel
        groq_task = asyncio.create_task(self.check_groq())
        gemini_task = asyncio.create_task(self.check_gemini())
        qdrant_task = asyncio.create_task(self.check_qdrant())
        tavily_task = asyncio.create_task(self.check_tavily())

        results["groq"] = await groq_task
        results["gemini"] = await gemini_task
        results["qdrant"] = await qdrant_task
        results["tavily"] = await tavily_task

        overall = all(s.get("status") == "healthy" for s in results.values())
        return {
            "status": "healthy" if overall else "unhealthy",
            "timestamp": time.time(),
            "services": results,
        }


# Global health checker
health_checker = HealthChecker()
