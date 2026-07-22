
import sys
from core.graph import build_graph
from core.state import ResearchState


def run_pipeline(query: str) -> str:
    graph = build_graph()

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

    final_state = graph.invoke(initial_state)
    return final_state["final_report"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"your research question here\"")
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