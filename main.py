# ruff: noqa: E402

import asyncio
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore", message=".*Direct use of automatic function calling.*")

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
    state_after_research = await graph.ainvoke(initial_state, config=config)
    
    # Check if the graph paused at reporter. If not, it means the router bypassed research.
    state_snapshot = await graph.aget_state(config)
    next_nodes = state_snapshot.next
    
    if "reporter" not in next_nodes:
        # Bypassed research (casual query)
        print("\n" + "=" * 50)
        print("FINAL REPORT")
        print("=" * 50 + "\n")
        report = state_after_research.get("final_report", "No report generated.")
        print(report)
        return report
    
    # Resume the graph to actually run the reporter and stream tokens
    final_report = ""
    header_printed = False
    
    async for event in graph.astream_events(None, config=config, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and isinstance(chunk.content, str):
                if not header_printed:
                    print("\n" + "=" * 50)
                    print("FINAL REPORT")
                    print("=" * 50 + "\n")
                    header_printed = True
                print(chunk.content, end="", flush=True)
                final_report += chunk.content
    
    if not final_report:
        # If streaming didn't catch anything, fetch and print from final state
        final_state = await graph.aget_state(config)
        final_report = final_state.values.get("final_report", "No report generated.")
        
        if not header_printed:
            print("\n" + "=" * 50)
            print("FINAL REPORT")
            print("=" * 50 + "\n")
            
        print(final_report)
    else:
        print()
    
    return final_report


async def main():
    from core.report_history import save_report

    if len(sys.argv) >= 2:
        query = " ".join(sys.argv[1:])
    else:
        query = None

    while True:
        if not query:
            query = input("Please type the question you want to research here (type 'exit' to quit): ")
            if query.lower().strip() in ['exit', 'quit']:
                print("Exiting. Have a great day!")
                break
            if not query.strip():
                print("No question provided. Exiting.")
                sys.exit(1)

        print(f"\nResearching: {query}\n")
        print("Running pipeline... (this may take 30-60 seconds)\n")

        report = await run_pipeline(query)

        print("\n" + "=" * 50)
        print("FOLLOW-UP QUESTIONS")
        print("=" * 50)
        while True:
            try:
                follow_up = input("\nDo you have any follow-up questions about this report? (type 'exit' or 'no' to finish): ")
                if follow_up.lower().strip() in ['exit', 'quit', 'q', 'no']:
                    break
                if not follow_up.strip():
                    continue
                
                print("\nThinking...")
                answer = await answer_doubt(report, follow_up)
                print(f"\nAnswer: {answer}")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting follow-up...")
                break

        print("\n" + "=" * 50)
        try:
            another = input("Do you want to generate another report? (yes/no): ")
        except (KeyboardInterrupt, EOFError):
            another = "no"

        print("Archiving the current report...")
        save_report(query=query, report=report)
        print("Report archived successfully!")

        if another.lower().strip() in ['yes', 'y']:
            query = None  # Reset query to prompt again
        else:
            print("Exiting. Have a great day!")
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.exceptions.CancelledError):
        import os
        os._exit(0)
