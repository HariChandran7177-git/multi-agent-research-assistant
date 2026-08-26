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

# Force correct MIME types for Windows registry issues
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Environment Validation ────────────────────────────────────────────────────
REQUIRED_ENV_VARS = [
    "GROQ_API_KEY",
    "TAVILY_API_KEY",
    "GOOGLE_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
]


def validate_environment() -> list[str]:
    """Check that all required environment variables are set.

    Returns a list of missing variables (empty if all present).
    """
    missing = []
    for var in REQUIRED_ENV_VARS:
        if not os.environ.get(var):
            missing.append(var)
    return missing


def check_environment():
    """Validate environment at startup. Exit if any vars are missing."""
    missing = validate_environment()
    if missing:
        print("=" * 60)
        print("CRITICAL: Missing required environment variables")
        print("=" * 60)
        print("The following environment variables are not set:")
        for var in missing:
            print(f"  - {var}")
        print()
        print("Please set them and restart the server.")
        print("=" * 60)
        sys.exit(1)


# ── Rate Limiting Setup (slowapi) ─────────────────────────────────────────────
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

limiter = Limiter(key_func=get_remote_address, default_limits=["5 per minute"])

# ── Add root to path so we can import core / agents ──────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from core.logger import get_logger
from core.graph import build_graph
from core.cache import cache as cache_manager
from core.metrics import metrics
from core.health import health_checker

logger = get_logger(__name__)
RESUME_EVENTS = {}

# Report history
from core.report_history import save_report, list_reports, get_report, delete_report


# Global graph - initialized asynchronously at startup
_compiled_graph = None
_graph_init_lock = asyncio.Lock()

# Concurrency guard — max 5 simultaneous research requests
# Prevents SQLite checkpoint conflicts and resource exhaustion
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
    return re.sub(r'\s+', ' ', query.strip().lower())


app = FastAPI(
    title="Multi-Agent Research Assistant",
    version="2.0.0",
    description="6-agent LangGraph pipeline with real-time SSE streaming",
)

# Validate environment at startup
check_environment()

# Register rate limiting exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Mount static files (frontend) ─────────────────────────────────────────────
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))


# ── Request / Response models ─────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    query: str
    user_id: str = "default_user"


class DoubtRequest(BaseModel):
    report: str
    question: str


class ResumeRequest(BaseModel):
    thread_id: str


# ── SSE event helper ──────────────────────────────────────────────────────────
def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Health check route ────────────────────────────────────────────────────���───
@app.get("/health")
async def health():
    """Basic health check with all service dependencies."""
    services = await health_checker.check_all()
    overall = "healthy" if services["status"] == "healthy" else "degraded"
    return {
        "status": overall,
        "version": "2.0.0",
        "timestamp": time.time(),
        "services": services["services"],
    }


@app.get("/health/metrics")
async def metrics_endpoint():
    """Get current metrics summary."""
    return metrics.get_summary()


# ── Core pipeline streaming ───────────────────────────────────────────────────
async def stream_pipeline(query: str, user_id: str = "default_user") -> AsyncGenerator[str, None]:
    """
    Run the real LangGraph pipeline, emitting SSE events at each stage.
    Uses async agent nodes for better performance.
    """
    cache_key = get_cache_key(query)
    thread_id = str(uuid.uuid4())
    RESUME_EVENTS[thread_id] = asyncio.Event()

    async def send(event: str, **kwargs):
        yield sse_event(event, {"timestamp": time.time(), **kwargs})

    # ── Stage 0: start ────────────────────────────────────────────────────────
    yield sse_event("start", {"query": query, "timestamp": time.time()})
    await asyncio.sleep(0.1)

    # Check Redis cache
    cached_result = cache_manager.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for query: {query}")

        # Fast-forward simulation
        agents = [
            ("router", "🔀", "Router", "Cache hit: Bypassing execution"),
            ("planner", "📋", "Planner", "Cache hit: Bypassing execution"),
            ("researcher", "⚡", "Researcher", "Cache hit: Bypassing execution"),
            ("retriever", "🧠", "Retriever", "Cache hit: Bypassing execution"),
            ("critic", "🧐", "Critic", "Cache hit: Bypassing execution"),
            ("reporter", "📝", "Reporter", "Cache hit: Bypassing execution"),
        ]

        for agent_id, icon, label, msg in agents:
            yield sse_event("agent_start", {
                "agent": agent_id, "icon": icon, "label": label, "message": msg, "timestamp": time.time()
            })
            await asyncio.sleep(0.05)
            yield sse_event("agent_done", {
                "agent": agent_id, "label": label, "result": {"cached": True}, "timestamp": time.time()
            })

        yield sse_event("complete", cached_result)
        return

    # Acquire semaphore — rejects if already 5 active requests
    if not _research_semaphore._value:  # check without blocking
        yield sse_event("error", {
            "message": "Server is busy. Too many concurrent research requests. Please try again in a moment.",
            "code": "CONCURRENCY_LIMIT",
            "timestamp": time.time(),
        })
        return

    async with _research_semaphore:
        try:
            # Start per-request cost tracking
            metrics.start_request(query)

            # Import pipeline components
            from core.state import ResearchState
            from agents.router import router_node
            from agents.planner import planner_node
            from agents.researcher import researcher_node
            from agents.retriever import retriever_node
            from agents.critic import critic_node
            from agents.reporter import reporter_node
            from core.config import CONFIDENCE_THRESHOLD, MAX_ITERATIONS, AGENT_TIMEOUT

            initial_state: ResearchState = {
                "query": query,
                "user_id": user_id,
                "plan": [],
                "research_results": [],
                "retrieved_docs": [],
                "critique": "",
                "confidence_score": 0.0,
                "iteration_count": 0,
                "final_report": "",
            }

            # Get the compiled graph
            compiled_graph = await get_graph()

            # Start Plain LLM task concurrently
            from langchain_groq import ChatGroq
            from core.config import GROQ_REPORTER_MODEL, GROQ_API_KEY
            plain_llm = ChatGroq(model=GROQ_REPORTER_MODEL, api_key=GROQ_API_KEY, temperature=0.3)
            plain_llm_task = asyncio.create_task(
                plain_llm.ainvoke(f"Please answer this query directly without using any external tools or web search. Give a concise but complete answer:\n\n{query}")
            )
            plain_llm_sent = False

            state = initial_state.copy()

            # ── Agent 1: Router ────────────────────────────────────────────────────
            yield sse_event("agent_start", {
                "agent": "router", "label": "Router", "icon": "🔀",
                "message": "Analyzing query intent & detecting tone...", "timestamp": time.time(),
            })
            metrics.start_agent("router")
            try:
                state = await asyncio.wait_for(
                    router_node(state),
                    timeout=AGENT_TIMEOUT
                )
                yield sse_event("agent_done", {
                    "agent": "router", "label": "Router",
                    "result": {"is_casual": state.get("is_casual", False), "tone": state.get("tone", "professional")},
                    "timestamp": time.time(),
                })
            except asyncio.TimeoutError:
                logger.warning("Router timed out")
                state["is_casual"] = False
                yield sse_event("agent_done", {"agent": "router", "label": "Router", "error": "timeout"})

            if state.get("is_casual"):
                if not plain_llm_sent:
                    try:
                        plain_res = await plain_llm_task
                        yield sse_event("plain_llm_done", {"response": plain_res.content, "timestamp": time.time()})
                    except Exception as e:
                        logger.error(f"Plain LLM failed: {e}")
                        yield sse_event("plain_llm_done", {"response": "Failed to generate plain LLM response.", "timestamp": time.time()})

                complete_payload = {
                    "report": state.get("final_report", ""),
                    "confidence": 1.0, "iterations": 0,
                    "tone": state.get("tone", "casual"),
                    "is_casual": True, "timestamp": time.time(),
                }
                cache_manager.set(cache_key, complete_payload)
                logger.info(f"Cache miss for query: {query}. Stored in cache.")
                metrics.end_request()
                yield sse_event("complete", complete_payload)
                return

            await asyncio.sleep(0.2)

            # ── Agent 2: Planner ───────────────────────────────────────────────────
            yield sse_event("agent_start", {
                "agent": "planner", "label": "Planner", "icon": "📋",
                "message": "Breaking query into research sub-tasks...", "timestamp": time.time(),
            })
            metrics.start_agent("planner")
            try:
                state = await asyncio.wait_for(
                    planner_node(state),
                    timeout=AGENT_TIMEOUT
                )
                yield sse_event("agent_done", {
                    "agent": "planner", "label": "Planner",
                    "result": {"plan": state.get("plan", [])}, "timestamp": time.time(),
                })
            except asyncio.TimeoutError:
                logger.warning("Planner timed out")
                yield sse_event("agent_done", {"agent": "planner", "label": "Planner", "error": "timeout"})

            await asyncio.sleep(0.2)

            # ── Research loop ──────────────────────────────────────────────────────
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
                    state = await asyncio.wait_for(
                        researcher_node(state),
                        timeout=AGENT_TIMEOUT * 2  # Web search takes longer
                    )
                    yield sse_event("agent_done", {
                        "agent": "researcher", "label": "Researcher",
                        "result": {"sources_found": len(state.get("research_results", [])), "iteration": iteration},
                        "timestamp": time.time(),
                    })
                except asyncio.TimeoutError:
                    logger.warning(f"Researcher timed out on iteration {iteration}")
                    yield sse_event("agent_done", {"agent": "researcher", "label": "Researcher", "error": "timeout"})

                await asyncio.sleep(0.2)

                # Agent 4: Retriever
                yield sse_event("agent_start", {
                    "agent": "retriever", "label": "Retriever", "icon": "🧠",
                    "message": "Embedding & semantic retrieval from Qdrant...",
                    "iteration": iteration, "timestamp": time.time(),
                })
                metrics.start_agent("retriever")
                metrics.record_embedding_call(num_texts=len(state.get("research_results", [])))
                try:
                    state = await asyncio.wait_for(
                        retriever_node(state),
                        timeout=AGENT_TIMEOUT
                    )
                    yield sse_event("agent_done", {
                        "agent": "retriever", "label": "Retriever",
                        "result": {"docs_retrieved": len(state.get("retrieved_docs", [])), "iteration": iteration},
                        "timestamp": time.time(),
                    })
                except asyncio.TimeoutError:
                    logger.warning(f"Retriever timed out on iteration {iteration}")
                    yield sse_event("agent_done", {"agent": "retriever", "label": "Retriever", "error": "timeout"})

                await asyncio.sleep(0.2)

                # Agent 5: Critic
                yield sse_event("agent_start", {
                    "agent": "critic", "label": "Critic", "icon": "🧐",
                    "message": "Evaluating research quality with hybrid scorer...",
                    "iteration": iteration, "timestamp": time.time(),
                })
                metrics.start_agent("critic")
                try:
                    state = await asyncio.wait_for(
                        critic_node(state),
                        timeout=AGENT_TIMEOUT
                    )
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
                except asyncio.TimeoutError:
                    logger.warning(f"Critic timed out on iteration {iteration}")
                    yield sse_event("agent_done", {"agent": "critic", "label": "Critic", "error": "timeout"})

                await asyncio.sleep(0.2)

                if score >= CONFIDENCE_THRESHOLD:
                    break

            # Human in the Loop (HITL) check
            yield sse_event("hitl_pause", {
                "message": "Human review required. Do you want to generate the final report?",
                "timestamp": time.time(), "thread_id": thread_id,
            })

            # Wait for user to confirm via /research/resume endpoint
            try:
                await asyncio.wait_for(
                    RESUME_EVENTS[thread_id].wait(),
                    timeout=300  # 5 minute wait
                )
            except asyncio.TimeoutError:
                logger.warning(f"HITL timeout for thread {thread_id}")
                del RESUME_EVENTS[thread_id]
                yield sse_event("error", {"message": "Human review timeout", "timestamp": time.time()})
                return

            del RESUME_EVENTS[thread_id]

            # Agent 6: Reporter
            yield sse_event("agent_start", {
                "agent": "reporter", "label": "Reporter", "icon": "📝",
                "message": "Writing tone-aware research report...", "timestamp": time.time(),
            })
            metrics.start_agent("reporter")
            try:
                state = await asyncio.wait_for(
                    reporter_node(state),
                    timeout=AGENT_TIMEOUT * 2  # Report generation takes longer
                )
                yield sse_event("agent_done", {
                    "agent": "reporter", "label": "Reporter",
                    "result": {"report_length": len(state.get("final_report", "")), "timestamp": time.time()},
                })
            except asyncio.TimeoutError:
                logger.warning("Reporter timed out")
                yield sse_event("agent_done", {"agent": "reporter", "label": "Reporter", "error": "timeout"})

            # Ensure plain LLM response is sent
            if not plain_llm_sent:
                try:
                    plain_res = await plain_llm_task
                    yield sse_event("plain_llm_done", {"response": plain_res.content, "timestamp": time.time()})
                except Exception as e:
                    logger.error(f"Plain LLM failed: {e}")

            complete_payload = {
                "report": state.get("final_report", ""),
                "confidence": round(state.get("confidence_score", 0.0), 3),
                "iterations": state.get("iteration_count", 0),
                "tone": state.get("tone", "professional"),
                "is_casual": False,
                "score_breakdown": state.get("score_breakdown", {}),
                "timestamp": time.time(),
            }
            cache_manager.set(cache_key, complete_payload)
            logger.info(f"Cache miss for query: {query}. Stored in cache.")
            # Persist to report history
            save_report(
                query=query,
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
                "detail": tb[:500],
                "timestamp": time.time(),
                "recoverable": isinstance(e, (asyncio.TimeoutError, ConnectionError)),
            })


# ── Routes ────────────────────────────────────────────────────────────────────


@app.post("/research/stream")
@limiter.limit("5 per minute")
async def research_stream(request: ResearchRequest, req: Request):
    """SSE endpoint — streams agent events as they happen."""
    return StreamingResponse(
        stream_pipeline(request.query, request.user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/research")
@limiter.limit("5 per minute")
async def research_sync(request: ResearchRequest, req: Request):
    """Synchronous fallback — waits for full pipeline then returns."""
    cache_key = get_cache_key(request.query)
    cached = cache_manager.get(cache_key)
    if cached:
        logger.info(f"Cache hit for query: {request.query}")
        return {
            "report": cached.get("report", ""),
            "confidence": cached.get("confidence", 0.0),
            "iterations": cached.get("iterations", 0),
            "tone": cached.get("tone", "professional"),
        }

    try:
        from core.state import ResearchState
        import uuid

        thread_id = str(uuid.uuid4())
        initial_state: ResearchState = {
            "query": request.query,
            "user_id": request.user_id,
            "plan": [],
            "research_results": [],
            "retrieved_docs": [],
            "critique": "",
            "confidence_score": 0.0,
            "iteration_count": 0,
            "final_report": "",
        }
        config = {"configurable": {"thread_id": thread_id}}

        # Get compiled graph and use async invoke
        compiled_graph = await get_graph()

        # Run until interrupt (HITL)
        await compiled_graph.ainvoke(initial_state, config=config)
        # Resume to complete
        final_state = await compiled_graph.ainvoke(None, config=config)

        complete_payload = {
            "report": final_state.get("final_report", ""),
            "confidence": final_state.get("confidence_score", 0.0),
            "iterations": final_state.get("iteration_count", 0),
            "tone": final_state.get("tone", "professional"),
        }
        if cache.enabled:
            await cache.set(request.query, complete_payload)
        return complete_payload
    except Exception as e:
        return {"error": str(e), "report": "Pipeline failed. Check your API keys."}


@app.post("/doubt")
async def doubt_sync(request: DoubtRequest):
    """Answer questions strictly based on the generated report."""
    try:
        from agents.doubt import answer_doubt
        answer = await answer_doubt(request.report, request.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e), "answer": "Failed to process doubt."}


@app.post("/research/resume")
async def resume_stream(request: ResumeRequest):
    """Resume the SSE stream for a given thread_id."""
    if request.thread_id in RESUME_EVENTS:
        RESUME_EVENTS[request.thread_id].set()
        return {"status": "resumed"}
    return {"error": "thread_id not found or expired"}


# ── Report History Routes ───────────────────────────────────────────────────────
@app.get("/reports")
async def get_reports(user_id: str = None, limit: int = 20):
    """List recent research reports (newest first)."""
    return {"reports": list_reports(user_id=user_id, limit=min(limit, 100))}


@app.get("/reports/{report_id}")
async def get_report_by_id(report_id: int):
    """Fetch a single report by its ID."""
    report = get_report(report_id)
    if not report:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return report


@app.delete("/reports/{report_id}")
async def delete_report_by_id(report_id: int):
    """Delete a report by ID."""
    success = delete_report(report_id)
    if not success:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")
    return {"status": "deleted", "id": report_id}


# ── PDF Export (Group 5.3) ─────────────────────────────────────────────────────
@app.get("/reports/{report_id}/export.pdf")
async def export_report_pdf(report_id: int):
    """Export a report as a styled PDF (HTML-rendered via markdown2)."""
    from fastapi import HTTPException
    from fastapi.responses import Response
    import markdown2

    record = get_report(report_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    md_text = record.get("report", "")
    html_body = markdown2.markdown(md_text, extras=["fenced-code-blocks", "tables", "header-ids"])

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Research Report #{report_id}</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 860px; margin: 40px auto; padding: 0 24px;
         color: #1a1a2e; line-height: 1.7; font-size: 15px; }}
  h1 {{ font-size: 2em; color: #16213e; border-bottom: 3px solid #0f3460; padding-bottom: 8px; }}
  h2 {{ font-size: 1.4em; color: #0f3460; margin-top: 2em; }}
  h3 {{ color: #533483; }}
  code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }}
  pre  {{ background: #1a1a2e; color: #eee; padding: 16px; border-radius: 8px; overflow-x: auto; }}
  blockquote {{ border-left: 4px solid #0f3460; margin-left: 0; padding-left: 16px; color: #555; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th {{ background: #0f3460; color: white; }}
  .meta {{ font-size: 0.85em; color: #666; margin-bottom: 2em; }}
</style>
</head><body>
<div class="meta">
  <strong>Query:</strong> {record.get('query', '')}<br/>
  <strong>Confidence:</strong> {record.get('confidence', 0):.0%} &nbsp;|&nbsp;
  <strong>Tone:</strong> {record.get('tone', 'professional').title()} &nbsp;|&nbsp;
  <strong>Iterations:</strong> {record.get('iterations', 0)} &nbsp;|&nbsp;
  <strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(record.get('created_at', 0)))}
</div>
{html_body}
</body></html>"""

    # Try weasyprint (richer PDF); fall back to plain HTML download
    try:
        import weasyprint
        pdf_bytes = weasyprint.HTML(string=html).write_pdf()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=report_{report_id}.pdf"},
        )
    except ImportError:
        # Serve styled HTML as fallback — browser can print-to-PDF
        return Response(
            content=html.encode(),
            media_type="text/html",
            headers={"Content-Disposition": f"inline; filename=report_{report_id}.html"},
        )


# ── Report Versioning (Group 5.2) ──────────────────────────────────────────────
@app.get("/reports/versions")
async def list_report_versions(query: str):
    """Return all report versions for the same query string."""
    import sqlite3
    from core.report_history import _get_conn
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT id, confidence, iterations, tone, created_at FROM reports WHERE query=? ORDER BY created_at DESC",
            (query,)
        ).fetchall()
        return {"query": query, "versions": [dict(r) for r in rows], "count": len(rows)}
    finally:
        conn.close()


# ── Multi-Language Research (Group 5.4) ────────────────────────────────────────
class MultiLangRequest(BaseModel):
    query: str
    user_id: str = "default_user"
    language: str = "auto"   # "auto" = detect from query, or explicit e.g. "es", "fr", "de"


@app.post("/research/multilang/stream")
@limiter.limit("5 per minute")
async def research_multilang_stream(request: MultiLangRequest, req: Request):
    """
    Multilingual research: detects query language (or uses 'language' param),
    runs the full pipeline, and instructs the reporter to respond in that language.
    """
    lang = request.language
    query = request.query

    # Build language-prefixed query so the reporter knows the target language
    if lang == "auto":
        # Simple heuristic: use the query as-is; reporter prompt already adapts tone
        prefixed_query = query
        lang_note = "Respond in the same language as the query above."
    else:
        lang_map = {
            "es": "Spanish", "fr": "French", "de": "German", "hi": "Hindi",
            "zh": "Chinese", "ar": "Arabic", "pt": "Portuguese", "ja": "Japanese",
            "ko": "Korean", "it": "Italian", "ru": "Russian",
        }
        lang_name = lang_map.get(lang.lower(), lang)
        prefixed_query = f"[Respond entirely in {lang_name}] {query}"
        lang_note = f"Respond entirely in {lang_name}."

    # Inject language note into user_id so reporter can see it via state (simple hack)
    augmented_user_id = f"{request.user_id}|lang:{lang}"

    return StreamingResponse(
        stream_pipeline(prefixed_query, augmented_user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Research-Language": lang,
        },
    )


# Mount static files at root / so style.css and app.js resolve correctly
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="web_root")

