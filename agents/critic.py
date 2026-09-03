import json
import asyncio
import re
from langchain_groq import ChatGroq
from core.state import ResearchState
import os
from dotenv import load_dotenv
from core.metrics import metrics
from tenacity import retry, stop_after_attempt, wait_exponential
from core.logger import get_logger
from core.scorer import calculate_objective_score, calculate_hybrid_score
from core.config import GROQ_MODEL, GROQ_API_KEY, RETRY_ATTEMPTS, RETRY_MULTIPLIER, RETRY_WAIT_MIN, RETRY_WAIT_MAX, AGENT_TIMEOUT, MAX_ITERATIONS

load_dotenv()

logger = get_logger(__name__)

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.2,
)

CRITIC_PROMPT = """You are a rigorous research quality evaluator.

Original Query: {query}

Research Findings:
{research_results}

Retrieved Documents:
{retrieved_docs}

Evaluate whether this research provides deep, actionable, and specific strategic insight to answer the query, including verified source URLs.
Assign a confidence score:
- 0.85 to 1.0: Comprehensive data, multi-angle business analysis, clear strategic details, and source citations present.
- 0.60 to 0.84: Good coverage but missing tactical depth, financial metrics, or specific source references.
- Below 0.60: Shallow, generic, or incomplete findings requiring additional research pass.

Respond ONLY with valid JSON in this exact format, with no markdown fences or extra text:
{{
  "score": <float between 0.0 and 1.0>,
  "critique": "<1-2 sentence explanation of gaps or confirmation of quality>"
}}
"""


@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=RETRY_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX))
def invoke_with_retry(llm, prompt):
    return llm.invoke(prompt)


async def critic_node(state: ResearchState) -> ResearchState:
    """Async critic node with timeout and metrics."""
    loop = asyncio.get_event_loop()
    logger.info("Evaluating research quality")

    research_text = "\n".join(state.get("research_results", []))[:8000]
    docs_text = "\n".join(state.get("retrieved_docs", []))[:6000]

    prompt = CRITIC_PROMPT.format(
        query=state["query"],
        research_results=research_text,
        retrieved_docs=docs_text,
    )

    try:
        # Timeout protection
        response = await asyncio.wait_for(
            loop.run_in_executor(None, invoke_with_retry, llm, prompt),
            timeout=AGENT_TIMEOUT
        )
        content = response.content.strip()
    except asyncio.TimeoutError:
        logger.error(f"Critic timeout after {AGENT_TIMEOUT}s")
        logger.warning("Critic failed — confidence forced to 0 to trigger downstream safety checks")
        state["confidence_score"] = 0.0
        state["critique"] = "Critic evaluation failed due to a timeout — confidence forced to 0 to trigger downstream safety checks."
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        metrics.end_agent("critic", error="timeout")
        return state
    except Exception as e:
        logger.error(f"LLM call failed after retries: {e}")
        logger.warning("Critic failed — confidence forced to 0 to trigger downstream safety checks")
        state["confidence_score"] = 0.0
        state["critique"] = "Critic evaluation failed due to an API error — confidence forced to 0 to trigger downstream safety checks."
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        metrics.end_agent("critic", error=str(e))
        return state

    # Robustly extract JSON object in case of extra text or markdown
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        content = match.group(0)

    try:
        parsed = json.loads(content)
        llm_score = float(parsed.get("score", 0.5))
        critique = parsed.get("critique", "No critique provided.")
    except (json.JSONDecodeError, ValueError):
        # Fallback if the LLM doesn't return clean JSON
        llm_score = 0.5
        critique = f"Could not parse critic output: {content[:200]}"
        logger.warning("Failed to parse LLM JSON output, using fallback score")

    # --- Objective scoring ---
    qdrant_scores = state.get("qdrant_scores", [])
    objective_breakdown = calculate_objective_score(state, qdrant_scores=qdrant_scores)

    # --- Hybrid: blend LLM judgment with objective signals ---
    hybrid_score = calculate_hybrid_score(llm_score, objective_breakdown)

    logger.info(
        f"Critic scores -> LLM: {llm_score:.3f} | "
        f"Objective: {objective_breakdown['objective_score']:.3f} | "
        f"Hybrid: {hybrid_score:.3f}"
    )

    state["confidence_score"] = hybrid_score
    state["score_breakdown"]  = {"llm_score": round(llm_score, 3), **objective_breakdown}
    state["critique"]         = critique
    state["iteration_count"]  = state.get("iteration_count", 0) + 1

    # Record metrics
    metrics.end_agent("critic", input_tokens=len(research_text) + len(docs_text), output_tokens=len(content))

    return state


if __name__ == "__main__":
    test_state: ResearchState = {
        "query": "What are the latest advancements in quantum computing?",
        "plan": ["Search recent breakthroughs", "Find key companies", "Identify challenges"],
        "research_results": ["IBM announced a 1000-qubit chip in 2023.", "Quantum error correction remains a major hurdle."],
        "retrieved_docs": ["IBM's Condor processor has 1,121 qubits."],
        "critique": "",
        "confidence_score": 0.0,
        "iteration_count": 0,
        "final_report": "",
    }
    result = asyncio.run(critic_node(test_state))
    print("Score:", result["confidence_score"])
    print("Critique:", result["critique"])
    print("Iteration:", result["iteration_count"])
