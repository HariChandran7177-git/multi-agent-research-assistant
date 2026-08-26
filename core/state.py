
from typing import TypedDict, List, Dict, Any


class ResearchState(TypedDict, total=False):
    query: str                      # the original user question
    user_id: str                    # user ID for multi-tenant isolation
    is_casual: bool                 # True if it's a simple chat query
    tone: str                       # Detected tone for the reporter to use
    plan: List[str]                 # sub-tasks the Planner breaks the query into
    research_results: List[str]     # raw findings from the Researcher agent
    retrieved_docs: List[str]       # relevant chunks pulled from Qdrant
    qdrant_scores: List[float]      # Qdrant cosine similarity scores per retrieved doc
    retrieval_available: bool       # True if Qdrant retrieval worked, False if fallback used
    critique: str                   # Critic's feedback on the current draft
    confidence_score: float         # Hybrid score (LLM + objective), 0-1, decides loop vs proceed
    score_breakdown: Dict[str, Any] # Per-signal debug breakdown from scorer
    iteration_count: int            # safety cap to stop infinite loops
    final_report: str               # Reporter's output