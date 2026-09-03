
import asyncio
import sys
from core.graph import build_graph
from core.state import ResearchState
from agents.doubt import answer_doubt


# Global graph instance
_graph = None


async def get_graph():
    """Get or initialize the singleton graph."""
    global _graph
    if _graph is None:
        _graph = await build_graph()
    return _graph


from core.logger import current_query

async def run_pipeline(query: str) -> str:
    current_query.set(query)
    graph = await get_graph()
    import uuid

    initial_state: ResearchState = {
        "query": query,
        "user_id": "cli_user",
        "plan": [],
        "research_results": [],
        "retrieved_docs": [],
        "critique": "",
        "confidence_score": 0.0,
        "iteration_count": 0,
        "final_report": "",
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # The graph has interrupt_before=["reporter"], so the first ainvoke pauses
    await graph.ainvoke(initial_state, config=config)
    
    # Resume the graph to actually run the reporter
    final_state = await graph.ainvoke(None, config=config)
    
    return final_state.get("final_report", "No report generated.")


async def main():
    if len(sys.argv) < 2:
        query = input("Please type the question you want to research here: ")
        if not query.strip():
            print("No question provided. Exiting.")
            sys.exit(1)
    else:
        query = " ".join(sys.argv[1:])
    print(f"\nResearching: {query}\n")
    print("Running pipeline... (this may take 30-60 seconds)\n")

    report = await run_pipeline(query)

    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50 + "\n")
    print(report)

    print("\n" + "=" * 50)
    print("FOLLOW-UP QUESTIONS")
    print("=" * 50)
    while True:
        try:
            follow_up = input("\nDo you have any follow-up questions about this report? (type 'exit' to quit): ")
            if follow_up.lower().strip() in ['exit', 'quit', 'q', 'no']:
                print("Exiting. Have a great day!")
                break
            if not follow_up.strip():
                continue
            
            print("\nThinking...")
            answer = await answer_doubt(report, follow_up)
            print(f"\nAnswer: {answer}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            import os
            os._exit(0)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.exceptions.CancelledError):
        import os
        os._exit(0)
