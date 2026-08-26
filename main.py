
import asyncio
import sys
from core.graph import build_graph
from core.state import ResearchState


# Global graph instance
_graph = None


async def get_graph():
    """Get or initialize the singleton graph."""
    global _graph
    if _graph is None:
        _graph = await build_graph()
    return _graph


async def run_pipeline(query: str) -> str:
    graph = await get_graph()

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

    final_state = await graph.ainvoke(initial_state)
    return final_state["final_report"]


async def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"Please type the question you want to research here !\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\nResearching: {query}\n")
    print("Running pipeline... (this may take 30-60 seconds)\n")

    report = await run_pipeline(query)

    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50 + "\n")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
