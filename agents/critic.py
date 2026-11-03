import json
import asyncio
import re
from langchain_groq import ChatGroq
from core.state import ResearchState
from dotenv import load_dotenv
from core.metrics import metrics
from tenacity import retry, stop_after_attempt, wait_exponential
from core.logger import get_logger
from core.scorer import calculate_objective_score, calculate_hybrid_score
from core.config import GROQ_CRITIC_MODEL, GROQ_API_KEY, RETRY_ATTEMPTS, RETRY_MULTIPLIER, RETRY_WAIT_MIN, RETRY_WAIT_MAX, AGENT_TIMEOUT

load_dotenv()
logger = get_logger(__name__)

llm = ChatGroq(
    model=GROQ_CRITIC_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.1,
)

CRITIC_PROMPT = """You are a rigorous research quality evaluator.
Evaluate whether the retrieved evidence provides solid, specific coverage for EVERY planned sub-task, with verified source citations.

Original Query: {query}

Planned Sub-tasks:
{plan}

Numbered Retrieved Chunks:
{retrieved_docs}

EVALUATION CRITERIA:
1. Sub-task Coverage: Does EVERY planned sub-task have at least one relevant, cited evidence chunk?
2. Factual Specificity: Are there hard numbers, dates, version numbers, or experimental/financial data rather than vague generalizations?
3. Source Diversity: Are citations from credible, diverse domains (not all from one site)?

Assign a confidence score:
- 0.85 to 1.0: Full coverage of all sub-tasks with hard numbers, verified citations, and high depth.
- 0.65 to 0.84: Good coverage, but one sub-task is relatively thin.
- Below 0.65: Any sub-task is completely missing evidence, or content is purely generic with no numbers.

Respond ONLY with valid JSON in this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "missing_subtasks": [<list of sub-task names lacking evidence, or empty>],
  "critique": "<1-2 sentence explanation of coverage and data quality>"
}}
"""


@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=RETRY_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX))
def invoke_with_retry(llm, prompt):
    return llm.invoke(prompt)


async def critic_node(state: ResearchState) -> ResearchState:
    """Async critic node with per-subtask coverage scoring and domain diversity enforcement."""
    loop = asyncio.get_running_loop()
    logger.info("Evaluating research quality and sub-task coverage")

    plan = state.get("plan", [])
    chunks = state.get("retrieved_chunks", [])
    subtask_coverage = state.get("subtask_coverage", {})
    docs_text = "\n".join(state.get("retrieved_docs", []))[:4000]

    # Rule-based hard checks:
    distinct_domains = set(ch.get("domain") for ch in chunks if ch.get("domain"))
    missing_subtasks = [task for task in plan if not subtask_coverage.get(task, False)]

    prompt = CRITIC_PROMPT.format(
        query=state["query"],
        plan="\n".join(f"- {p}" for p in plan),
        retrieved_docs=docs_text,
    )

    print(f"[DEBUG CRITIC] Invoking Critic LLM ({GROQ_CRITIC_MODEL})...", flush=True)
    try:
        response = await asyncio.wait_for(
            loop.run_in_executor(None, invoke_with_retry, llm, prompt),
            timeout=AGENT_TIMEOUT
        )
        content = response.content.strip()
        print("[DEBUG CRITIC] Critic LLM response received.", flush=True)
    except Exception as e:
        logger.warning(f"Critic LLM evaluation encountered error: {e}. Using deterministic scoring.")
        content = '{"score": 0.70, "critique": "Fallback to objective rules", "missing_subtasks": []}'
        print(f"[DEBUG CRITIC] Critic LLM failed ({e}), using fallback.", flush=True)

    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        content = match.group(0)

    try:
        parsed = json.loads(content)
        llm_score = float(parsed.get("score", 0.75))
        critique = parsed.get("critique", "Coverage evaluated.")
        # Hard check: only count truly missing sub-tasks based on verified retrieved chunks
        missing_subtasks = [task for task in plan if not subtask_coverage.get(task, False)]
    except Exception:
        llm_score = 0.75
        critique = "Objective coverage check completed."
        missing_subtasks = [task for task in plan if not subtask_coverage.get(task, False)]

    # --- HARD PENALTIES (User Item 6) ---
    # Hard penalty 1: If any sub-task has zero retrieved chunks, cap score
    if missing_subtasks and len(plan) > 1:
        coverage_ratio = max(0.0, (len(plan) - len(missing_subtasks)) / len(plan))
        llm_score = min(llm_score, round(coverage_ratio * 0.85, 2))
        critique += f" Note: Sub-task(s) lacking evidence: {', '.join(missing_subtasks[:2])}."

    # Hard penalty 2: Minimum distinct domains (at least 2 distinct domains required)
    if len(distinct_domains) < 2 and len(chunks) > 2:
        logger.warning(f"Low domain diversity ({len(distinct_domains)} domains). Applying diversity penalty.")
        llm_score = min(llm_score, 0.60)
        critique += f" Low domain diversity (only {len(distinct_domains)} source domain)."

    # --- Objective scoring ---
    qdrant_scores = state.get("qdrant_scores", [])
    objective_breakdown = calculate_objective_score(state, qdrant_scores=qdrant_scores)

    # --- Hybrid: blend LLM judgment with objective signals ---
    hybrid_score = calculate_hybrid_score(llm_score, objective_breakdown)

    logger.info(
        f"Critic -> LLM: {llm_score:.3f} | Objective: {objective_breakdown['objective_score']:.3f} | "
        f"Hybrid: {hybrid_score:.3f} | Distinct Domains: {len(distinct_domains)}"
    )

    state["prev_confidence_score"] = state.get("confidence_score", -1.0)
    state["confidence_score"] = hybrid_score
    state["score_breakdown"] = {
        "llm_score": round(llm_score, 3),
        "distinct_domains": len(distinct_domains),
        "missing_subtasks": missing_subtasks,
        **objective_breakdown
    }
    state["critique"] = critique
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    metrics.end_agent("critic", input_tokens=len(docs_text), output_tokens=len(content))
    return state
