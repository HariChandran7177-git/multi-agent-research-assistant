
from typing import TypedDict, List


class ResearchState(TypedDict):
    query: str                  # the original user question
    plan: List[str]             # sub-tasks the Planner breaks the query into
    research_results: List[str] # raw findings from the Researcher agent
    retrieved_docs: List[str]   # relevant chunks pulled from Qdrant
    critique: str                # Critic's feedback on the current draft
    confidence_score: float     # Critic's score, 0-1, decides loop vs proceed
    iteration_count: int        # safety cap to stop infinite loops
    final_report: str           # Reporter's output