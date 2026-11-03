# ruff: noqa: E402
import os
import asyncio
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from core.state import ResearchState
from core.logger import get_logger
from core.metrics import metrics
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import (
    GEMINI_PLANNER_MODEL, GEMINI_API_KEY,
    GROQ_RESEARCHER_MODEL, GROQ_API_KEY,
    RETRY_ATTEMPTS, RETRY_MULTIPLIER, RETRY_WAIT_MIN, RETRY_WAIT_MAX, AGENT_TIMEOUT
)

# Suppress Google SDK Automatic Function Calling (AFC) deprecation/recommendation warnings
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", message=".*generate_content.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.*")

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
logger = get_logger(__name__)

CURRENT_YEAR = 2026

# Primary model: Groq gpt-oss-20b (~0.4s, high reliability, no quota error); fallback: Gemini
if GROQ_API_KEY:
    planner_llm = ChatGroq(
        model=GROQ_RESEARCHER_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
    )
elif GEMINI_API_KEY:
    planner_llm = ChatGoogleGenerativeAI(
        model=GEMINI_PLANNER_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )
else:
    planner_llm = None


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
def invoke_with_retry(llm_instance, prompt):
    return llm_instance.invoke(prompt)


async def planner_node(state: ResearchState) -> ResearchState:
    """Async planner node with freshness awareness (2026) and tactical query planning."""
    loop = asyncio.get_running_loop()
    query = state["query"]
    logger.info(f"Planning sub-tasks for query: {query}")

    prompt = f"""You are an elite research planner.
The current year is {CURRENT_YEAR}.

Break down the user's research query into 4 distinct, concrete, and highly targeted research sub-tasks that directly address every aspect of the question:
1. Context & Architecture/Mechanics: Foundational background, core mechanisms, or historical baseline.
2. Direct Technical Specifications & Growth Drivers: Key features, operational details, benchmarks, or financial metrics.
3. Current Numbers & Hard Data ({CURRENT_YEAR-1}-{CURRENT_YEAR}): Latest delivery numbers, revenue, benchmarks, clinical trial results, or adoption statistics.
4. Future Trends, Strategic Challenges & Outlook ({CURRENT_YEAR}+): Market shifts, open problems, regulatory landscape, or next-phase developments.

CRITICAL RULES:
- Explicitly inject '{CURRENT_YEAR-1}-{CURRENT_YEAR}' into data, metrics, trends, and market sub-tasks so searches return recent facts.
- Every sub-task MUST be a concrete search query that includes the subject entity name (e.g., 'Tesla current deliveries revenue and market share {CURRENT_YEAR-1}-{CURRENT_YEAR}').
- Return ONLY 4 clean lines, one per sub-task. No bullets, no numbers, no markdown fences.

Query: {query}"""

    try:
        response = await asyncio.wait_for(
            loop.run_in_executor(None, invoke_with_retry, planner_llm, prompt),
            timeout=15
        )
        raw_lines = response.content.strip().split("\n")
        cleaned_tasks = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
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

        if len(cleaned_tasks) > 4:
            cleaned_tasks = cleaned_tasks[:4]

        logger.info(f"Generated {len(cleaned_tasks)} freshness-aware sub-tasks")
        state["plan"] = cleaned_tasks
        state["iteration_count"] = 0

        metrics.end_agent("planner", input_tokens=len(query), output_tokens=len(response.content))
    except Exception as e:
        logger.warning(f"Planner LLM failed ({e}), using default targeted breakdown")
        state["plan"] = [
            f"{query} overview architecture and mechanics",
            f"{query} key drivers and specifications",
            f"{query} latest data and numbers {CURRENT_YEAR-1}-{CURRENT_YEAR}",
            f"{query} future trends and outlook {CURRENT_YEAR}"
        ]
        state["iteration_count"] = 0

    return state
