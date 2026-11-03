import json
import asyncio
import warnings
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from core.state import ResearchState
from core.logger import get_logger
from core.metrics import metrics
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import (
    GEMINI_ROUTER_MODEL, GEMINI_API_KEY,
    GROQ_RESEARCHER_MODEL, GROQ_API_KEY,
    RETRY_ATTEMPTS, RETRY_MULTIPLIER, RETRY_WAIT_MIN, RETRY_WAIT_MAX, AGENT_TIMEOUT
)

# Suppress Google SDK Automatic Function Calling (AFC) deprecation/recommendation warnings
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", message=".*generate_content.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.*")

load_dotenv()
logger = get_logger(__name__)

# Primary model: Groq gpt-oss-20b (fast ~0.3s, no 429 quota limits); fallback: Gemini
if GROQ_API_KEY:
    primary_llm = ChatGroq(
        model=GROQ_RESEARCHER_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.1,
    )
elif GEMINI_API_KEY:
    primary_llm = ChatGoogleGenerativeAI(
        model=GEMINI_ROUTER_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )
else:
    primary_llm = None

ROUTER_PROMPT = """You are an intelligent query router and tone detector.

TASK 1 — Is this a research query or a casual chat?
- CASUAL: greetings, jokes, "how are you", "what's 2+2", simple pleasantries.
- RESEARCH: anything requiring web search, current events, comparisons, analysis, "how to", technical deep-dives, or future predictions.

CRITICAL RULE: Default to RESEARCH (`is_casual: false`) for almost all queries. ONLY use CASUAL for simple greetings like "hello" or "how are you". If the user is asking a question, asking for advice, or asking for an explanation, it MUST be RESEARCH. Do not try to answer it yourself!

TASK 2 — Detect the exact writing tone from the query's wording:
- If the query says "explain like I'm 5" or "ELI5" → tone = "ELI5 — explain like I am 5 years old, use simple analogies"
- If it says "like a senior engineer" or "technical" → tone = "senior software engineer — precise, technical, no hand-holding"
- If it says "academic" or "research paper" → tone = "academic — formal, citations, structured"
- If it says "funny", "comedian", "make it fun" → tone = "funny — use humour and wit, be entertaining"
- If it says "simple", "beginner", "basic" → tone = "beginner-friendly — clear, no jargon"
- If it says "professional", "formal", "business" → tone = "professional and formal"
- If the query is phrased casually (slang, emoji) → tone = "casual and friendly"
- If the query is phrased neutrally with no tone cues → tone = "professional and informative"

Original query: "{query}"

Return ONLY a valid JSON object, no markdown code blocks, no conversational text:
{{"is_casual": <true|false>, "response": "<if casual: short friendly answer, else empty string>", "tone": "<detected tone string>"}}
"""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
def invoke_with_retry(llm_instance, prompt):
    return llm_instance.invoke(prompt)


def _detect_tone_heuristic(query: str) -> str:
    """Fast keyword-based fallback tone detector."""
    q = query.lower()
    if any(x in q for q in ["eli5", "explain like", "like i'm 5", "like i am 5", "simple", "beginner"]):
        return "ELI5 — explain like I am 5 years old, use simple analogies"
    if any(x in q for q in ["senior engineer", "technical", "in depth", "deep dive", "architecture"]):
        return "senior software engineer — precise, technical, no hand-holding"
    if any(x in q for q in ["academic", "research paper", "literature", "scholarly"]):
        return "academic — formal, citations, structured"
    if any(x in q for q in ["funny", "joke", "humour", "humor", "comedian", "fun"]):
        return "funny — use humour and wit, be entertaining"
    if any(x in q for q in ["professional", "formal", "business", "executive"]):
        return "professional and formal"
    return "professional and informative"


async def router_node(state: ResearchState) -> ResearchState:
    """Async router node with high-speed classification and graceful fallbacks."""
    loop = asyncio.get_running_loop()
    logger.info("Routing query to determine path and tone")
    query = state["query"]

    # Short-circuit obvious casual queries
    clean_q = query.strip().lower()
    if clean_q in ["hi", "hello", "hey", "hola", "sup", "how are you", "who are you"]:
        logger.info("Direct rule matched casual greeting.")
        state["is_casual"] = True
        state["tone"] = "casual and friendly"
        state["final_report"] = "Hello! I am your AI Multi-Agent Research Assistant. How can I help you research today?"
        return state

    try:
        response = await asyncio.wait_for(
            loop.run_in_executor(None, invoke_with_retry, primary_llm, ROUTER_PROMPT.format(query=query)),
            timeout=10
        )
        content = response.content
        if isinstance(content, list):
            content = " ".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in content])
        content = str(content).strip()

        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            is_casual = parsed.get("is_casual", False)
            tone = parsed.get("tone") or _detect_tone_heuristic(query)
            state["is_casual"] = is_casual
            state["tone"] = tone
            if is_casual:
                state["final_report"] = parsed.get("response", "Hello! How can I assist your research today?")
            logger.info(f"Detected tone: {tone} | Casual: {is_casual}")
            return state
    except Exception as e:
        logger.warning(f"Router LLM fallback to heuristic: {e}")

    # Fallback to research mode with heuristic tone
    state["is_casual"] = False
    state["tone"] = _detect_tone_heuristic(query)
    logger.info(f"Fallback routing: research=True, tone='{state['tone']}'")
    return state
