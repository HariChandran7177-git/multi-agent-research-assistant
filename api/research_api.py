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
from typing import AsyncGenerator

# Force correct MIME types for Windows registry issues
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Add root to path so we can import core / agents ──────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from core.logger import get_logger
from core.graph import build_graph

logger = get_logger(__name__)
QUERY_CACHE = {}

def get_cache_key(query: str) -> str:
    return re.sub(r'\s+', ' ', query.strip().lower())

# Compile graph once at startup
compiled_graph = build_graph()

app = FastAPI(
    title="Multi-Agent Research Assistant",
    version="2.0.0",
    description="6-agent LangGraph pipeline with real-time SSE streaming",
)

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

class DoubtRequest(BaseModel):
    report: str
    question: str

# ── SSE event helper ──────────────────────────────────────────────────────────
def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Core pipeline streaming ───────────────────────────────────────────────────
async def stream_pipeline(query: str) -> AsyncGenerator[str, None]:
    """
    Run the real LangGraph pipeline, emitting SSE events at each stage.
    Falls back to mock mode if API keys are not configured.
    """
    cache_key = get_cache_key(query)

    async def send(event: str, **kwargs):
        yield sse_event(event, {"timestamp": time.time(), **kwargs})

    # ── Stage 0: start ────────────────────────────────────────────────────────
    yield sse_event("start", {"query": query, "timestamp": time.time()})
    await asyncio.sleep(0.1)
    
    if cache_key in QUERY_CACHE:
        logger.info(f"Cache hit for query: {query}")
        
        # Fast-forward simulation so the frontend animations don't break
        agents = [
            ("router", "🔀", "Router", "Cache hit: Bypassing execution"),
            ("planner", "📋", "Planner", "Cache hit: Bypassing execution"),
            ("researcher", "⚡", "Researcher", "Cache hit: Bypassing execution"),
            ("retriever", "🧠", "Retriever", "Cache hit: Bypassing execution"),
            ("critic", "🧐", "Critic", "Cache hit: Bypassing execution"),
            ("reporter", "📝", "Reporter", "Cache hit: Bypassing execution")
        ]
        
        for agent_id, icon, label, msg in agents:
            yield sse_event("agent_start", {
                "agent": agent_id, "icon": icon, "label": label, "message": msg, "timestamp": time.time()
            })
            await asyncio.sleep(0.05)
            yield sse_event("agent_done", {
                "agent": agent_id, "label": label, "result": {"cached": True}, "timestamp": time.time()
            })
            
        yield sse_event("complete", QUERY_CACHE[cache_key])
        return

    try:
        # Import pipeline
        from core.graph import build_graph
        from core.state import ResearchState

        # ── Stage 1: Router ───────────────────────────────────────────────────
        yield sse_event("agent_start", {
            "agent": "router",
            "label": "Router",
            "icon": "🔀",
            "message": "Analyzing query intent & detecting tone...",
            "timestamp": time.time(),
        })

        loop = asyncio.get_event_loop()
        
        # We use the globally compiled graph
        graph = compiled_graph

        initial_state: ResearchState = {
            "query": query,
            "plan": [],
            "research_results": [],
            "retrieved_docs": [],
            "critique": "",
            "confidence_score": 0.0,
            "iteration_count": 0,
            "final_report": "",
        }

        # Start Plain LLM task concurrently
        from langchain_groq import ChatGroq
        from core.config import GROQ_REPORTER_MODEL, GROQ_API_KEY
        plain_llm = ChatGroq(model=GROQ_REPORTER_MODEL, api_key=GROQ_API_KEY, temperature=0.3)
        plain_llm_task = asyncio.create_task(
            plain_llm.ainvoke(f"Please answer this query directly without using any external tools or web search. Give a concise but complete answer:\n\n{query}")
        )
        plain_llm_sent = False

        # We'll run each node manually to emit progress events
        from agents.router import router_node
        from agents.planner import planner_node
        from agents.researcher import researcher_node
        from agents.retriever import retriever_node
        from agents.critic import critic_node
        from agents.reporter import reporter_node
        from core.config import CONFIDENCE_THRESHOLD, MAX_ITERATIONS

        state = initial_state.copy()

        # Router
        state = await loop.run_in_executor(None, router_node, state)
        yield sse_event("agent_done", {
            "agent": "router",
            "label": "Router",
            "result": {
                "is_casual": state.get("is_casual", False),
                "tone": state.get("tone", "professional"),
            },
            "timestamp": time.time(),
        })

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
                "confidence": 1.0,
                "iterations": 0,
                "tone": state.get("tone", "casual"),
                "is_casual": True,
                "timestamp": time.time(),
            }
            QUERY_CACHE[cache_key] = complete_payload
            logger.info(f"Cache miss for query: {query}. Stored in cache.")
            yield sse_event("complete", complete_payload)
            return

        await asyncio.sleep(0.2)

        # Planner
        yield sse_event("agent_start", {
            "agent": "planner",
            "label": "Planner",
            "icon": "📋",
            "message": "Breaking query into research sub-tasks...",
            "timestamp": time.time(),
        })
        state = await loop.run_in_executor(None, planner_node, state)
        yield sse_event("agent_done", {
            "agent": "planner",
            "label": "Planner",
            "result": {"plan": state.get("plan", [])},
            "timestamp": time.time(),
        })
        await asyncio.sleep(0.2)

        # Research loop
        iteration = 0
        while iteration < MAX_ITERATIONS:
            iteration += 1

            # Researcher
            yield sse_event("agent_start", {
                "agent": "researcher",
                "label": "Researcher",
                "icon": "⚡",
                "message": f"Parallel web search — pass {iteration}...",
                "iteration": iteration,
                "timestamp": time.time(),
            })
            state = await loop.run_in_executor(None, researcher_node, state)
            yield sse_event("agent_done", {
                "agent": "researcher",
                "label": "Researcher",
                "result": {"sources_found": len(state.get("research_results", []))},
                "iteration": iteration,
                "timestamp": time.time(),
            })
            await asyncio.sleep(0.2)

            # Retriever
            yield sse_event("agent_start", {
                "agent": "retriever",
                "label": "Retriever",
                "icon": "🧠",
                "message": "Embedding & semantic retrieval from Qdrant...",
                "iteration": iteration,
                "timestamp": time.time(),
            })
            state = await loop.run_in_executor(None, retriever_node, state)
            yield sse_event("agent_done", {
                "agent": "retriever",
                "label": "Retriever",
                "result": {"docs_retrieved": len(state.get("retrieved_docs", []))},
                "iteration": iteration,
                "timestamp": time.time(),
            })
            await asyncio.sleep(0.2)

            # Critic
            yield sse_event("agent_start", {
                "agent": "critic",
                "label": "Critic",
                "icon": "🧐",
                "message": "Evaluating research quality with hybrid scorer...",
                "iteration": iteration,
                "timestamp": time.time(),
            })
            state = await loop.run_in_executor(None, critic_node, state)
            score = state.get("confidence_score", 0.0)
            critique = state.get("critique", "")
            yield sse_event("agent_done", {
                "agent": "critic",
                "label": "Critic",
                "result": {
                    "confidence": round(score, 3),
                    "critique": critique,
                    "score_breakdown": state.get("score_breakdown", {}),
                    "passed": score >= CONFIDENCE_THRESHOLD,
                },
                "iteration": iteration,
                "timestamp": time.time(),
            })
            await asyncio.sleep(0.2)

            if score >= CONFIDENCE_THRESHOLD or state.get("iteration_count", 0) >= MAX_ITERATIONS:
                break

        # Reporter
        yield sse_event("agent_start", {
            "agent": "reporter",
            "label": "Reporter",
            "icon": "📝",
            "message": "Writing tone-aware research report...",
            "timestamp": time.time(),
        })
        state = await loop.run_in_executor(None, reporter_node, state)
        yield sse_event("agent_done", {
            "agent": "reporter",
            "label": "Reporter",
            "result": {"report_length": len(state.get("final_report", ""))},
            "timestamp": time.time(),
        })

        # Complete
        
        # Ensure plain LLM is sent if it hasn't been already
        if not plain_llm_sent:
            try:
                plain_res = await plain_llm_task
                yield sse_event("plain_llm_done", {"response": plain_res.content, "timestamp": time.time()})
            except Exception as e:
                logger.error(f"Plain LLM failed: {e}")
                yield sse_event("plain_llm_done", {"response": "Failed to generate plain LLM response.", "timestamp": time.time()})

        complete_payload = {
            "report": state.get("final_report", ""),
            "confidence": round(state.get("confidence_score", 0.0), 3),
            "iterations": state.get("iteration_count", 0),
            "tone": state.get("tone", "professional"),
            "is_casual": False,
            "score_breakdown": state.get("score_breakdown", {}),
            "timestamp": time.time(),
        }
        QUERY_CACHE[cache_key] = complete_payload
        logger.info(f"Cache miss for query: {query}. Stored in cache.")
        yield sse_event("complete", complete_payload)

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[API ERROR] {e}\n{tb}")
        yield sse_event("error", {
            "message": str(e),
            "detail": tb[:500],
            "timestamp": time.time(),
        })


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}


@app.post("/research/stream")
async def research_stream(request: ResearchRequest):
    """SSE endpoint — streams agent events as they happen."""
    return StreamingResponse(
        stream_pipeline(request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/research")
async def research_sync(request: ResearchRequest):
    """Synchronous fallback — waits for full pipeline then returns."""
    cache_key = get_cache_key(request.query)
    if cache_key in QUERY_CACHE:
        logger.info(f"Cache hit for query: {request.query}")
        cached = QUERY_CACHE[cache_key]
        return {
            "report": cached.get("report", ""),
            "confidence": cached.get("confidence", 0.0),
            "iterations": cached.get("iterations", 0),
            "tone": cached.get("tone", "professional"),
        }
        
    try:
        from core.state import ResearchState

        initial_state: ResearchState = {
            "query": request.query,
            "plan": [],
            "research_results": [],
            "retrieved_docs": [],
            "critique": "",
            "confidence_score": 0.0,
            "iteration_count": 0,
            "final_report": "",
        }
        final_state = compiled_graph.invoke(initial_state)
        
        # Store in cache for future calls
        complete_payload = {
            "report": final_state.get("final_report", ""),
            "confidence": final_state.get("confidence_score", 0.0),
            "iterations": final_state.get("iteration_count", 0),
            "tone": final_state.get("tone", "professional"),
        }
        QUERY_CACHE[cache_key] = complete_payload
        logger.info(f"Cache miss for query: {request.query}. Stored in cache.")
        return complete_payload
    except Exception as e:
        return {"error": str(e), "report": "Pipeline failed. Check your API keys."}

@app.post("/doubt")
async def doubt_sync(request: DoubtRequest):
    """Answer questions strictly based on the generated report."""
    try:
        from agents.doubt import answer_doubt
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(None, answer_doubt, request.report, request.question)
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e), "answer": "Failed to process doubt."}

# Mount static files at root / so style.css and app.js resolve correctly when visiting /
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="web_root")
