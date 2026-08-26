from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from core.state import ResearchState
from core.config import CONFIDENCE_THRESHOLD, MAX_ITERATIONS
from core.logger import get_logger
from agents.router import router_node
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.retriever import retriever_node
from agents.critic import critic_node
from agents.reporter import reporter_node
import os
import aiosqlite

def should_loop(state: ResearchState) -> str:
    """Decides what happens after the Critic runs."""
    score = state.get("confidence_score", 0.0)
    iterations = state.get("iteration_count", 0)

    if score >= CONFIDENCE_THRESHOLD:
        return "reporter"
    if iterations >= MAX_ITERATIONS:
        return "reporter"  # force stop even if confidence is low
    return "researcher"  # loop back for more research


def route_after_router(state: ResearchState) -> str:
    """Decides if we should bypass the research pipeline."""
    if state.get("is_casual"):
        return END
    return "planner"


# Checkpoint database path
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints.sqlite"))
CONN_STR = f"sqlite:///{DB_PATH}"


# Global connection - kept open for the checkpointer
_db_conn = None


async def _get_db_connection():
    """Get or create the aiosqlite connection."""
    global _db_conn
    if _db_conn is None:
        _db_conn = await aiosqlite.connect(DB_PATH)
        # Configure for better performance
        await _db_conn.execute("PRAGMA journal_mode=WAL")
        await _db_conn.execute("PRAGMA synchronous=NORMAL")
    return _db_conn


async def _create_checkpointer():
    """Create and return a new AsyncSqliteSaver instance."""
    conn = await _get_db_connection()
    return AsyncSqliteSaver(conn)


async def build_graph():
    """Build the LangGraph pipeline with async checkpointer."""
    workflow = StateGraph(ResearchState)

    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("reporter", reporter_node)

    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", route_after_router, {END: END, "planner": "planner"})
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "retriever")
    workflow.add_edge("retriever", "critic")
    workflow.add_conditional_edges("critic", should_loop, {"researcher": "researcher", "reporter": "reporter"})
    workflow.add_edge("reporter", END)

    # Create checkpointer with connection
    checkpointer = await _create_checkpointer()

    return workflow.compile(checkpointer=checkpointer, interrupt_before=["reporter"])


# Global compiled graph - initialized once at startup
_compiled_graph = None
_graph_init_lock = None


async def get_compiled_graph():
    """Get or create the singleton compiled graph."""
    global _compiled_graph, _graph_init_lock
    if _graph_init_lock is None:
        import asyncio
        _graph_init_lock = asyncio.Lock()

    if _compiled_graph is None:
        async with _graph_init_lock:
            if _compiled_graph is None:
                logger = get_logger(__name__)
                logger.info("Building LangGraph pipeline at startup...")
                _compiled_graph = await build_graph()
                logger.info("LangGraph pipeline built successfully")
    return _compiled_graph


async def close_db_connection():
    """Close the database connection (for cleanup)."""
    global _db_conn
    if _db_conn:
        await _db_conn.close()
        _db_conn = None


if __name__ == "__main__":
    async def main():
        graph = await build_graph()

        initial_state: ResearchState = {
            "query": "What are the latest advancements in quantum computing?",
            "plan": [],
            "research_results": [],
            "retrieved_docs": [],
            "critique": "",
            "confidence_score": 0.0,
            "iteration_count": 0,
            "final_report": "",
        }

        import uuid
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        final_state = await graph.ainvoke(initial_state, config=config)
        print("\n=== FINAL REPORT ===\n")
        print(final_state.get("final_report", "No report generated."))

        # Cleanup
        await close_db_connection()

    asyncio.run(main())
