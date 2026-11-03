# ruff: noqa: E402
"""
Multi-Agent Research Assistant — FastAPI Backend
Streams real-time agent progress via Server-Sent Events (SSE)
"""

import asyncio
import json
import sys
import os
import time
import traceback
import mimetypes
import uuid
from typing import AsyncGenerator
from collections import defaultdict

# Force correct MIME types for Windows registry issues
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import threading
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse as StarletteJSONResponse

# ── Add root to path so we can import core / agents ──────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import (
    APP_API_KEY, ALLOWED_ORIGINS, MAX_QUERY_LENGTH,
    RATE_LIMIT_PER_MINUTE, CONFIDENCE_THRESHOLD,
    MAX_ITERATIONS, AGENT_TIMEOUT
)
from core.logger import get_logger
from core.graph import build_graph
from core.cache import cache as cache_manager
from core.metrics import metrics
from core.health import health_checker
from core.report_history import save_report, list_reports, get_report, delete_report

logger = get_logger(__name__)

# ── Environment Validation ────────────────────────────────────────────────────
REQUIRED_ENV_VARS = [
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
    "QDRANT_URL",
]


def validate_environment() -> list[str]:
    """Check that all required environment variables are set.
    Returns a list of missing variables (empty if all present).
    """
    missing = []
    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            missing.append(var)
    # Check Gemini key
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        missing.append("GOOGLE_API_KEY (or GEMINI_API_KEY)")
    return missing


def check_environment():
    """Validate environment at startup."""
    missing = validate_environment()
    if missing:
        msg = f"CRITICAL: Missing required environment variables: {', '.join(missing)}"
        logger.error(msg)
        print("=" * 60)
        print(msg)
        print("Set these in your .env file or server environment.")
        print("=" * 60)


# Run environment check
check_environment()

RESUME_EVENTS = {}

# Global graph - initialized asynchronously at startup
_compiled_graph = None
_graph_init_lock = asyncio.Lock()

# Concurrency guard — max 5 simultaneous research requests
_research_semaphore = asyncio.Semaphore(5)


async def get_graph():
    """Get or initialize the compiled graph (singleton pattern)."""
    global _compiled_graph
    if _compiled_graph is None:
        async with _graph_init_lock:
            if _compiled_graph is None:
                logger.info("Initializing LangGraph pipeline at startup...")
                _compiled_graph = await build_graph()
                logger.info("LangGraph pipeline initialized successfully")
    return _compiled_graph


def get_cache_key(query: str) -> str:
    """Generate cache key from query."""
    import re
    return re.sub(r'\s+', ' ', query.strip().lower())


app = FastAPI(
    title="Multi-Agent Research Assistant",
    version="2.0.0",
    description="6-agent LangGraph pipeline with real-time SSE streaming",
)

@app.on_event("startup")
async def _startup():
    """Pre-warm the SentenceTransformer model and LangGraph pipeline at startup.
    Eliminates the ~80s cold-load lag on the first user query.
    """
    import threading
    from agents.retriever import get_embeddings
    loop = asyncio.get_event_loop()

    def _warmup():
        try:
            logger.info("Pre-warming SentenceTransformer embedding model...")
            get_embeddings()
            logger.info("SentenceTransformer pre-warm complete.")
        except Exception as e:
            logger.warning(f"Embedding pre-warm failed (non-fatal): {e}")

    # Run in a thread so it doesn't block the startup coroutine
    threading.Thread(target=_warmup, daemon=True, name="st-prewarm").start()
    # Also init the graph in background
    asyncio.create_task(get_graph())

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Rate Limiting Middleware ──────────────────────────────────────────────────
_rate_limit_records = defaultdict(list)
_rate_limit_lock = threading.Lock()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce rate limits per client IP (sliding 60-second window)."""

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Exempt health checks and static files
        if path.startswith("/static") or path in ("/health", "/health/deep", "/health/metrics", "/"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = 60.0

        with _rate_limit_lock:
            timestamps = [t for t in _rate_limit_records[client_ip] if now - t < window]
            if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
                return StarletteJSONResponse(
                    {
                        "detail": f"Rate limit exceeded. Maximum {RATE_LIMIT_PER_MINUTE} requests per minute.",
                        "retry_after": 60,
                    },
                    status_code=429,
                )
            timestamps.append(now)
            _rate_limit_records[client_ip] = timestamps

        return await call_next(request)


app.add_middleware(RateLimitMiddleware)


# ── API-key authentication middleware ──────────────────────────────────────────────
class APIKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests missing a valid API key when APP_API_KEY is configured.
    Supports X-API-Key header or Authorization: Bearer <token>.
    """

    async def dispatch(self, request: Request, call_next):
        if not APP_API_KEY:
            return await call_next(request)

        path = request.url.path
        if path in ("/health", "/health/deep", "/health/metrics") and request.method == "GET":
            return await call_next(request)
        if path == "/" or path.startswith("/static"):
            return await call_next(request)

        provided_key = request.headers.get("X-API-Key", "")
        if not provided_key:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                provided_key = auth_header[7:].strip()

        if provided_key != APP_API_KEY:
            return StarletteJSONResponse(
                {"detail": "Invalid or missing API key. Provide X-API-Key header or Authorization: Bearer token."},
                status_code=401,
            )

        return await call_next(request)


app.add_middleware(APIKeyMiddleware)

# ── Mount static files (frontend) ─────────────────────────────────────────────
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))


# ── Request / Response models ─────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    user_id: str = "default_user"
    enable_hitl: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or purely whitespace.")
        if len(cleaned) > MAX_QUERY_LENGTH:
            raise ValueError(f"Query exceeds maximum allowed length of {MAX_QUERY_LENGTH} characters.")
        return cleaned


class DoubtRequest(BaseModel):
    report: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=1000)


class ResumeRequest(BaseModel):
    thread_id: str


# ── SSE event helper ──────────────────────────────────────────────────────────
def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def await_with_heartbeats(coro, step_name: str, interval: float = 15.0):
    """Awaits a coroutine while yielding periodic SSE heartbeat events every 15s."""
    task = asyncio.create_task(coro)
    while not task.done():
        done, _ = await asyncio.wait([task], timeout=interval)
        if not done:
            yield sse_event("ping", {
                "heartbeat": True,
                "step": step_name,
                "timestamp": time.time(),
                "message": f"Processing {step_name}..."
            })
    yield await task


# ── Health check routes ───────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Lightweight health check for load balancers."""
    result = health_checker.check_shallow()
    return {
        "status": result["status"],
        "version": "2.0.0",
        "timestamp": result["timestamp"],
        "config": result["config"],
    }


@app.get("/health/deep")
async def health_deep():
    """Full deep health check — tests Groq, Gemini, Qdrant, Tavily.
    Returns HTTP 200 if all healthy, or HTTP 503 if any service is down.
    """
    services = await health_checker.check_all_deep()
    status_code = 200 if services["status"] == "healthy" else 503
    return StarletteJSONResponse(
        content={
            "status": services["status"],
            "version": "2.0.0",
            "timestamp": time.time(),
            "services": services["services"],
        },
        status_code=status_code,
    )


@app.get("/health/metrics")
async def metrics_endpoint():
    """Get current metrics summary."""
    return metrics.get_summary()


# ── Core pipeline streaming ───────────────────────────────────────────────────
async def stream_pipeline(query: str, user_id: str = "default_user", enable_hitl: bool = False) -> AsyncGenerator[str, None]:
    """Run the multi-agent pipeline with SSE streaming and 15s heartbeats."""
    cleaned_query = query.strip() if query else ""
    if not cleaned_query:
        yield sse_event("error", {"message": "Query cannot be empty.", "code": "EMPTY_QUERY", "timestamp": time.time()})
        return

    if len(cleaned_query) > MAX_QUERY_LENGTH:
        yield sse_event("error", {"message": f"Query exceeds maximum allowed length of {MAX_QUERY_LENGTH} characters.", "code": "QUERY_TOO_LONG", "timestamp": time.time()})
        return

    cache_key = get_cache_key(cleaned_query)
    thread_id = str(uuid.uuid4())
    RESUME_EVENTS[thread_id] = asyncio.Event()

    yield sse_event("start", {"query": cleaned_query, "timestamp": time.time()})
    await asyncio.sleep(0.05)

    # Check cache
    cached_result = await cache_manager.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for query: {cleaned_query}")
        agents = [
            ("router", "🔀", "Router", "Cache hit: Bypassing execution"),
            ("planner", "📋", "Planner", "Cache hit: Bypassing execution"),
            ("researcher", "⚡", "Researcher", "Cache hit: Bypassing execution"),
            ("retriever", "🧠", "Retriever", "Cache hit: Bypassing execution"),
            ("critic", "🧐", "Critic", "Cache hit: Bypassing execution"),
            ("reporter", "📝", "Reporter", "Cache hit: Bypassing execution"),
        ]
        for agent_id, icon, label, msg in agents:
            yield sse_event("agent_start", {"agent": agent_id, "icon": icon, "label": label, "message": msg, "timestamp": time.time()})
            await asyncio.sleep(0.05)
            yield sse_event("agent_done", {"agent": agent_id, "label": label, "result": {"cached": True}, "timestamp": time.time()})
        yield sse_event("complete", cached_result)
        return

    # Check concurrency semaphore
    if not _research_semaphore._value:
        yield sse_event("error", {
            "message": "Server is busy. Too many concurrent research requests. Please try again shortly.",
            "code": "CONCURRENCY_LIMIT",
            "timestamp": time.time(),
        })
        return

    async with _research_semaphore:
        try:
            metrics.start_request(cleaned_query)

            from core.state import ResearchState
            from agents.router import router_node
            from agents.planner import planner_node
            from agents.researcher import researcher_node
            from agents.retriever import retriever_node
            from agents.critic import critic_node
            from agents.reporter import reporter_node_streaming

            initial_state: ResearchState = {
                "query": cleaned_query,
                "user_id": user_id,
                "plan": [],
                "research_results": [],
                "retrieved_docs": [],
                "critique": "",
                "confidence_score": 0.0,
                "iteration_count": 0,
                "final_report": "",
                "sources": [],
            }

            state = initial_state.copy()

            # ── Agent 1: Router ──
            yield sse_event("agent_start", {
                "agent": "router", "label": "Router", "icon": "🔀",
                "message": "Analyzing query intent & detecting tone...", "timestamp": time.time(),
            })
            metrics.start_agent("router")
            try:
                async for item in await_with_heartbeats(
                    asyncio.wait_for(router_node(state), timeout=AGENT_TIMEOUT),
                    step_name="Router"
                ):
                    if isinstance(item, str) and item.startswith("event: ping"):
                        yield item
                    else:
                        state = item

                yield sse_event("agent_done", {
                    "agent": "router", "label": "Router",
                    "result": {"is_casual": state.get("is_casual", False), "tone": state.get("tone", "professional")},
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.warning(f"Router error/timeout: {e}")
                state["is_casual"] = False
                yield sse_event("agent_done", {"agent": "router", "label": "Router", "error": str(e)})

            if state.get("is_casual"):
                complete_payload = {
                    "report": state.get("final_report", ""),
                    "confidence": 1.0,
                    "iterations": 0,
                    "tone": state.get("tone", "casual"),
                    "is_casual": True,
                    "timestamp": time.time(),
                }
                await cache_manager.set(cache_key, complete_payload)
                metrics.end_request()
                yield sse_event("complete", complete_payload)
                return

            await asyncio.sleep(0.1)

            # ── Agent 2: Planner ──
            yield sse_event("agent_start", {
                "agent": "planner", "label": "Planner", "icon": "📋",
                "message": "Breaking query into research sub-tasks...", "timestamp": time.time(),
            })
            metrics.start_agent("planner")
            try:
                async for item in await_with_heartbeats(
                    asyncio.wait_for(planner_node(state), timeout=AGENT_TIMEOUT),
                    step_name="Planner"
                ):
                    if isinstance(item, str) and item.startswith("event: ping"):
                        yield item
                    else:
                        state = item

                yield sse_event("agent_done", {
                    "agent": "planner", "label": "Planner",
                    "result": {"plan": state.get("plan", [])}, "timestamp": time.time(),
                })
            except Exception as e:
                logger.warning(f"Planner error/timeout: {e}")
                yield sse_event("agent_done", {"agent": "planner", "label": "Planner", "error": str(e)})

            await asyncio.sleep(0.1)

            # ── Iterative Research Loop ──
            iteration = 0
            score = 0.0
            while iteration < MAX_ITERATIONS:
                iteration += 1

                # Agent 3: Researcher
                yield sse_event("agent_start", {
                    "agent": "researcher", "label": "Researcher", "icon": "⚡",
                    "message": f"Parallel web search — pass {iteration}...",
                    "iteration": iteration, "timestamp": time.time(),
                })
                metrics.start_agent("researcher")
                metrics.record_tavily_call(num_calls=len(state.get("plan", [])))
                try:
                    async for item in await_with_heartbeats(
                        asyncio.wait_for(researcher_node(state), timeout=AGENT_TIMEOUT * 2),
                        step_name=f"Researcher pass {iteration}"
                    ):
                        if isinstance(item, str) and item.startswith("event: ping"):
                            yield item
                        else:
                            state = item

                    yield sse_event("agent_done", {
                        "agent": "researcher", "label": "Researcher",
                        "result": {
                            "sources_found": len(state.get("research_results", [])),
                            "iteration": iteration,
                            "sources": state.get("sources", [])
                        },
                        "timestamp": time.time(),
                    })
                except Exception as e:
                    logger.warning(f"Researcher error/timeout on iteration {iteration}: {e}")
                    yield sse_event("agent_done", {"agent": "researcher", "label": "Researcher", "error": str(e)})

                await asyncio.sleep(0.1)

                # Agent 4: Retriever
                yield sse_event("agent_start", {
                    "agent": "retriever", "label": "Retriever", "icon": "🧠",
                    "message": "Embedding & semantic retrieval from Qdrant...",
                    "iteration": iteration, "timestamp": time.time(),
                })
                metrics.start_agent("retriever")
                metrics.record_embedding_call(num_texts=len(state.get("research_results", [])))
                try:
                    async for item in await_with_heartbeats(
                        asyncio.wait_for(retriever_node(state), timeout=AGENT_TIMEOUT),
                        step_name=f"Retriever pass {iteration}"
                    ):
                        if isinstance(item, str) and item.startswith("event: ping"):
                            yield item
                        else:
                            state = item

                    yield sse_event("agent_done", {
                        "agent": "retriever", "label": "Retriever",
                        "result": {"docs_retrieved": len(state.get("retrieved_docs", [])), "iteration": iteration},
                        "timestamp": time.time(),
                    })
                except Exception as e:
                    logger.warning(f"Retriever error/timeout on iteration {iteration}: {e}")
                    yield sse_event("agent_done", {"agent": "retriever", "label": "Retriever", "error": str(e)})

                await asyncio.sleep(0.1)

                # Agent 5: Critic
                yield sse_event("agent_start", {
                    "agent": "critic", "label": "Critic", "icon": "🧐",
                    "message": "Evaluating research quality with hybrid scorer...",
                    "iteration": iteration, "timestamp": time.time(),
                })
                metrics.start_agent("critic")
                try:
                    async for item in await_with_heartbeats(
                        asyncio.wait_for(critic_node(state), timeout=AGENT_TIMEOUT),
                        step_name=f"Critic pass {iteration}"
                    ):
                        if isinstance(item, str) and item.startswith("event: ping"):
                            yield item
                        else:
                            state = item

                    score = state.get("confidence_score", 0.0)
                    yield sse_event("agent_done", {
                        "agent": "critic", "label": "Critic",
                        "result": {
                            "confidence": round(score, 3),
                            "critique": state.get("critique", ""),
                            "score_breakdown": state.get("score_breakdown", {}),
                            "passed": score >= CONFIDENCE_THRESHOLD,
                            "iteration": iteration,
                        },
                        "timestamp": time.time(),
                    })
                except Exception as e:
                    logger.warning(f"Critic error/timeout on iteration {iteration}: {e}")
                    yield sse_event("agent_done", {"agent": "critic", "label": "Critic", "error": str(e)})

                await asyncio.sleep(0.1)

                if score >= CONFIDENCE_THRESHOLD:
                    break

            # ── Human-in-the-Loop (HITL) Check (only if explicitly enabled) ──
            if enable_hitl:
                yield sse_event("hitl_pause", {
                    "message": "Research pass complete. Waiting for approval.",
                    "plan": state.get("plan", []),
                    "timestamp": time.time(),
                    "thread_id": thread_id,
                })
                try:
                    await asyncio.wait_for(RESUME_EVENTS[thread_id].wait(), timeout=30)
                except asyncio.TimeoutError:
                    logger.info(f"Auto-resuming HITL for thread {thread_id}")
                finally:
                    RESUME_EVENTS.pop(thread_id, None)

            # ── Agent 6: Reporter ──
            yield sse_event("agent_start", {
                "agent": "reporter", "label": "Reporter", "icon": "📝",
                "message": "Writing tone-aware research report...", "timestamp": time.time(),
            })
            metrics.start_agent("reporter")
            try:
                _chunk_queue: asyncio.Queue = asyncio.Queue()

                async def _enqueue_chunk(chunk_text: str):
                    await _chunk_queue.put(chunk_text)

                reporter_task = asyncio.create_task(
                    reporter_node_streaming(state, _enqueue_chunk)
                )

                while not reporter_task.done() or not _chunk_queue.empty():
                    try:
                        chunk_text = _chunk_queue.get_nowait()
                        yield sse_event("report_chunk", {"chunk": chunk_text, "timestamp": time.time()})
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.02)

                state = await reporter_task

                yield sse_event("agent_done", {
                    "agent": "reporter", "label": "Reporter",
                    "result": {"report_length": len(state.get("final_report", "")), "timestamp": time.time()},
                })
            except Exception as e:
                logger.warning(f"Reporter error/timeout: {e}")
                yield sse_event("agent_done", {"agent": "reporter", "label": "Reporter", "error": str(e)})

            complete_payload = {
                "report": state.get("final_report", ""),
                "confidence": round(state.get("confidence_score", 0.0), 3),
                "iterations": state.get("iteration_count", 0),
                "tone": state.get("tone", "professional"),
                "is_casual": False,
                "score_breakdown": state.get("score_breakdown", {}),
                "timestamp": time.time(),
            }
            await cache_manager.set(cache_key, complete_payload)

            save_report(
                query=cleaned_query,
                report=state.get("final_report", ""),
                confidence=round(state.get("confidence_score", 0.0), 3),
                iterations=state.get("iteration_count", 0),
                tone=state.get("tone", "professional"),
                user_id=user_id,
            )
            metrics.end_request()
            yield sse_event("complete", complete_payload)

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[API ERROR] {e}\n{tb}")
            metrics.end_request()
            yield sse_event("error", {
                "message": str(e),
                "code": type(e).__name__,
                "detail": str(e),
                "timestamp": time.time(),
                "recoverable": False,
            })


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/research/stream")
async def research_stream(payload: ResearchRequest, request: Request):
    """SSE endpoint — streams agent events and keepalive pings."""
    cleaned = payload.query.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(cleaned) > MAX_QUERY_LENGTH:
        raise HTTPException(status_code=400, detail=f"Query exceeds maximum allowed length of {MAX_QUERY_LENGTH} characters.")

    return StreamingResponse(
        stream_pipeline(payload.query, payload.user_id, payload.enable_hitl),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


@app.post("/research")
async def research_sync(payload: ResearchRequest, request: Request):
    """Synchronous fallback route."""
    cleaned = payload.query.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(cleaned) > MAX_QUERY_LENGTH:
        raise HTTPException(status_code=400, detail=f"Query exceeds maximum allowed length of {MAX_QUERY_LENGTH} characters.")

    cache_key = get_cache_key(cleaned)
    cached = await cache_manager.get(cache_key)
    if cached:
        return {
            "report": cached.get("report", ""),
            "confidence": cached.get("confidence", 0.0),
            "iterations": cached.get("iterations", 0),
            "tone": cached.get("tone", "professional"),
        }

    try:
        from core.state import ResearchState
        thread_id = str(uuid.uuid4())
        initial_state: ResearchState = {
            "query": payload.query,
            "user_id": payload.user_id,
            "plan": [],
            "research_results": [],
            "retrieved_docs": [],
            "critique": "",
            "confidence_score": 0.0,
            "iteration_count": 0,
            "final_report": "",
            "sources": [],
        }
        config = {"configurable": {"thread_id": thread_id}}
        compiled_graph = await get_graph()
        await compiled_graph.ainvoke(initial_state, config=config)
        final_state = await compiled_graph.ainvoke(None, config=config)

        complete_payload = {
            "report": final_state.get("final_report", ""),
            "confidence": final_state.get("confidence_score", 0.0),
            "iterations": final_state.get("iteration_count", 0),
            "tone": final_state.get("tone", "professional"),
        }
        if cache_manager.enabled:
            await cache_manager.set(payload.query, complete_payload)
        return complete_payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")


@app.post("/doubt")
async def doubt_sync(request: DoubtRequest):
    """Answer questions strictly based on the generated report."""
    try:
        from agents.doubt import answer_doubt
        answer = await answer_doubt(request.report, request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process doubt: {e}")


@app.post("/research/resume")
async def resume_stream(request: ResumeRequest):
    """Resume the SSE stream for a given thread_id."""
    if request.thread_id in RESUME_EVENTS:
        RESUME_EVENTS[request.thread_id].set()
        return {"status": "resumed"}
    return {"error": "thread_id not found or expired"}


@app.get("/reports")
async def get_reports(user_id: str = None, limit: int = 20):
    """List recent research reports (newest first)."""
    return {"reports": list_reports(user_id=user_id, limit=min(limit, 100))}


@app.get("/reports/{report_id}")
async def get_report_by_id(report_id: int):
    """Fetch a single report by its ID."""
    report = get_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return report


@app.delete("/reports/{report_id}")
async def delete_report_by_id(report_id: int):
    """Delete a report by ID."""
    success = delete_report(report_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return {"status": "deleted", "id": report_id}


# Mount static files at root / so style.css and app.js resolve correctly
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="web_root")
