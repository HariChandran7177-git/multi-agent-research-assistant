import json
from langchain_groq import ChatGroq
from core.state import ResearchState
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from core.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def invoke_with_retry(llm, prompt):
    return llm.invoke(prompt)


def critic_node(state: ResearchState) -> ResearchState:
    logger.info("Evaluating research quality")

    research_text = "\n".join(state.get("research_results", []))[:8000]
    docs_text = "\n".join(state.get("retrieved_docs", []))[:6000]

    prompt = CRITIC_PROMPT.format(
        query=state["query"],
        research_results=research_text,
        retrieved_docs=docs_text,
    )

    try:
        response = invoke_with_retry(llm, prompt)
        content = response.content.strip()
    except Exception as e:
        logger.error(f"LLM call failed after retries: {e}")
        state["confidence_score"] = 0.0
        state["critique"] = f"Critic failed to run: {e}"
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        return state

    # Strip accidental markdown fences if the model adds them
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    try:
        parsed = json.loads(content)
        score = float(parsed.get("score", 0.5))
        critique = parsed.get("critique", "No critique provided.")
    except (json.JSONDecodeError, ValueError):
        # Fallback if the LLM doesn't return clean JSON
        score = 0.5
        critique = f"Could not parse critic output: {content[:200]}"
        logger.warning(
            f"Failed to parse LLM JSON output, using fallback score")

    logger.info(f"Confidence score: {score}")
    state["confidence_score"] = score
    state["critique"] = critique
    state["iteration_count"] = state.get("iteration_count", 0) + 1

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
    result = critic_node(test_state)
    print("Score:", result["confidence_score"])
    print("Critique:", result["critique"])
    print("Iteration:", result["iteration_count"])
