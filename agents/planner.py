# ruff: noqa: E402
from core.config import GEMINI_PLANNER_MODEL, GEMINI_API_KEY, RETRY_ATTEMPTS, RETRY_MULTIPLIER, RETRY_WAIT_MIN, RETRY_WAIT_MAX, AGENT_TIMEOUT
import os
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from core.state import ResearchState
from core.logger import get_logger
from core.metrics import metrics
from tenacity import retry, stop_after_attempt, wait_exponential

# Load .env from the project root (one level up from agents/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


logger = get_logger(__name__)

llm = ChatGoogleGenerativeAI(
    model=GEMINI_PLANNER_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
)

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=RETRY_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX))
def invoke_with_retry(llm, prompt):
    return llm.invoke(prompt)


async def planner_node(state: ResearchState) -> ResearchState:
    """Async planner node with timeout and metrics."""
    loop = asyncio.get_running_loop()
    query = state["query"]
    logger.info(f"Planning sub-tasks for query: {query}")

    prompt = f"""Classify the query intent into one of the following 10 task types and break it down into 4-5 focused, tactical research sub-tasks.

[... your existing 10 categories stay exactly the same ...]

CRITICAL: Every single sub-task you output MUST explicitly include the query's subject matter — never output a bare category label on its own.
For example, if the query is "explain about the ai voice agents" and the task type is Technical explainer, output:
Concept Overview of AI Voice Agents
Core Technical Mechanics of AI Voice Agents
Code/Architecture Examples of AI Voice Agents
Practical Significance of AI Voice Agents

Do NOT output just "Concept Overview" or "Core Technical Mechanics" — these are category labels, not final sub-tasks. Always attach the topic.

Return ONLY a plain list of clear research topics, one per line. Do NOT include numbers, bullet points, or prefixes.

Query: {query}"""

    try:
        # Timeout protection
        response = await asyncio.wait_for(
            loop.run_in_executor(None, invoke_with_retry, llm, prompt),
            timeout=AGENT_TIMEOUT
        )
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

        CATEGORY_LABELS = {
            "business strategy", "market/competitive research", "technical explainer",
            "comparison/decision", "trend/current-state", "how-to/process",
            "pros/cons/evaluation", "problem-diagnosis", "broad/open research",
            "risk assessment"
        }
        cleaned_tasks = [
            t for t in cleaned_tasks
            if t.strip().lower() not in CATEGORY_LABELS
        ]

        logger.info(f"Generated {len(cleaned_tasks)} cleaned sub-tasks")
        state["plan"] = cleaned_tasks
        state["iteration_count"] = 0

        # Record metrics
        metrics.end_agent("planner", input_tokens=len(query),
                          output_tokens=len(response.content))

    except asyncio.TimeoutError:
        logger.warning(f"Planner timeout after {AGENT_TIMEOUT}s")
        state["plan"] = [query]
        state["iteration_count"] = 0
        metrics.end_agent("planner", error="timeout")
    except Exception as e:
        logger.error(f"LLM call failed after retries: {e}")
        state["plan"] = [query]
        state["iteration_count"] = 0
        metrics.end_agent("planner", error=str(e))

    return state


if __name__ == "__main__":
    sample_state = {
        "query": "AI agent architectures and multi-agent coordination"}
    print("\n--- Testing Planner Agent ---")
    print(f"Query: {sample_state['query']}\n")
    result = asyncio.run(planner_node(sample_state))
    print("Generated Plan:")
    for task in result["plan"]:
        print(f" - {task}")
