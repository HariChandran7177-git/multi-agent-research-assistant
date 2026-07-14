
from langgraph.graph import StateGraph, END
from core.state import ResearchState
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.retriever import retriever_node
from agents.critic import critic_node
from agents.reporter import reporter_node

CONFIDENCE_THRESHOLD = 0.7
MAX_ITERATIONS = 3


def should_loop(state: ResearchState) -> str:
    """Decides what happens after the Critic runs."""
    score = state.get("confidence_score", 0.0)
    iterations = state.get("iteration_count", 0)

    if score >= CONFIDENCE_THRESHOLD:
        return "reporter"
    if iterations >= MAX_ITERATIONS:
        return "reporter"  # force stop even if confidence is low
    return "researcher"  # loop back for more research


def build_graph():
    workflow = StateGraph(ResearchState)

    # Register each agent as a node
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("reporter", reporter_node)

    # Set the starting point
    workflow.set_entry_point("planner")

    # Straight-line connections
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "retriever")
    workflow.add_edge("retriever", "critic")

    # Conditional branch after critic
    workflow.add_conditional_edges(
        "critic",
        should_loop,
        {
            "researcher": "researcher",
            "reporter": "reporter",
        },
    )

    # Reporter is the final step
    workflow.add_edge("reporter", END)

    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()

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

    final_state = graph.invoke(initial_state)
    print("\n=== FINAL REPORT ===\n")
    print(final_state["final_report"])