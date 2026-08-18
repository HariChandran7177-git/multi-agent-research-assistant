
import sys
from core.graph import build_graph
from core.state import ResearchState

# Build the graph once at startup — reused across all queries
_graph = build_graph()


def run_pipeline(query: str) -> str:
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

    final_state = _graph.invoke(initial_state)
    return final_state["final_report"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"Please type the question you want to research here !\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\nResearching: {query}\n")
    print("Running pipeline... (this may take 30-60 seconds)\n")

    report = run_pipeline(query)

    print("\n" + "=" * 50)
    print("FINAL REPORT")
    print("=" * 50 + "\n")
    print(report)


if __name__ == "__main__":
    main()