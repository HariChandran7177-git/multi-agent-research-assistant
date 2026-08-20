import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from core.logger import get_logger

# Load .env from the project root (one level up from agents/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from core.config import GROQ_MODEL, GROQ_API_KEY, RETRY_ATTEMPTS, RETRY_MULTIPLIER, RETRY_WAIT_MIN, RETRY_WAIT_MAX

logger = get_logger(__name__)

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)


@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=RETRY_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX))
def invoke_with_retry(llm, prompt):
    return llm.invoke(prompt)


def planner_node(state):
    query = state["query"]
    logger.info(f"Planning sub-tasks for query: {query}")

    prompt = f"""Classify the query intent into one of the following 10 task types and break it down into 4-5 focused, tactical research sub-tasks:

1. Business strategy ("Should we expand into X?"): Focus on Situation/Context, Strategic Options, Risks & Tradeoffs, Recommendations, and Implementation Steps.
2. Market/competitive research ("Research EV market"): Focus on Market Overview, Key Players, Industry Trends, Gaps & Opportunities, and Market Forecast.
3. Technical explainer ("Explain transformers"): Focus on Concept Overview, Core Technical Mechanics, Code/Architecture Examples, and Practical Significance.
4. Comparison/decision ("AWS vs GCP"): Focus on Comparison Criteria, Option A Strengths/Weaknesses, Option B Strengths/Weaknesses, Tradeoffs, and Final Recommendation.
5. Trend/current-state ("Latest AI regulations"): Focus on Timeline of Developments, Industry/Policy Implications, Key Indicators, and What to Watch.
6. How-to/process ("How RAG retrieval works"): Focus on Prerequisites, Sequential Step-by-Step Execution, Optimization Best Practices, and Key Summary.
7. Pros/cons/evaluation ("Is Qdrant good for prod?"): Focus on Core Advantages (Pros), Limitations/Drawbacks (Cons), Production Verdict, and Caveats.
8. Problem-diagnosis ("Why churn rate increasing?"): Focus on Likely Root Causes (Ranked), Empirical Evidence & Metrics, Investigation Steps, and Remediation.
9. Broad/open research ("Research quantum computing"): Focus on Background Context, Key Findings by Sub-Topic, Industry Impact, and Synthesis.
10. Risk assessment ("Risks of microservices"): Focus on Risk Categories, Severity & Likelihood, Mitigation Strategies, and Actionable Safeguards.

Return ONLY a plain list of clear research topics, one per line. Do NOT include numbers, bullet points, or prefixes.

Query: {query}"""

    try:
        response = invoke_with_retry(llm, prompt)
    except Exception as e:
        logger.error(f"LLM call failed after retries: {e}")
        state["plan"] = [query]
        state["iteration_count"] = 0
        return state

    raw_lines = response.content.strip().split("\n")
    cleaned_tasks = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading numbers (e.g. "1. ", "1)") or bullets ("- ", "* ")
        while line and (line[0].isdigit() or line[0] in ".-*#\t "):
            line = line.lstrip("0123456789.-*#\t ")
        if line:
            cleaned_tasks.append(line)

    logger.info(f"Generated {len(cleaned_tasks)} cleaned sub-tasks")
    state["plan"] = cleaned_tasks
    state["iteration_count"] = 0
    return state


if __name__ == "__main__":
    sample_state = {"query": "AI agent architectures and multi-agent coordination"}
    print("\n--- Testing Planner Agent ---")
    print(f"Query: {sample_state['query']}\n")
    result = planner_node(sample_state)
    print("Generated Plan:")
    for task in result["plan"]:
        print(f" - {task}")

