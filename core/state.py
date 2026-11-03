from typing import TypedDict, List, Dict, Any


class ResearchState(TypedDict, total=False):
    query: str                      # the original user question
    user_id: str                    # user ID for multi-tenant isolation
    is_casual: bool                 # True if it's a simple chat query
    tone: str                       # Detected tone for the reporter to use
    plan: List[str]                 # sub-tasks the Planner breaks the query into
    research_results: List[str]     # raw findings from the Researcher agent
    sources: List[Dict[str, str]]   # title, url, domain, date for frontend display
    retrieved_docs: List[str]       # relevant numbered chunks pulled from Qdrant
    retrieved_chunks: List[Dict[str, Any]] # Structured chunk objects (id, text, url, title, domain, date, subtask)
    qdrant_scores: List[float]      # Qdrant cosine similarity scores per retrieved doc
    retrieval_available: bool       # True if Qdrant retrieval worked, False if fallback used
    subtask_coverage: Dict[str, bool] # Per-subtask coverage status
    critique: str                   # Critic's feedback on the current draft
    confidence_score: float         # Hybrid score (LLM + objective), 0-1, decides loop vs proceed
    score_breakdown: Dict[str, Any] # Per-signal debug breakdown from scorer
    iteration_count: int            # safety cap to stop infinite loops
    final_report: str               # Reporter's output
