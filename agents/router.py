import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from core.state import ResearchState
from core.logger import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import GROQ_MODEL, GROQ_API_KEY, RETRY_ATTEMPTS, RETRY_MULTIPLIER, RETRY_WAIT_MIN, RETRY_WAIT_MAX

load_dotenv()
logger = get_logger(__name__)

# Use groq/compound-mini for fast routing decisions
llm = ChatGroq(
    model="groq/compound-mini",
    api_key=GROQ_API_KEY,
    temperature=0.1,
)

ROUTER_PROMPT = """You are an intelligent query router and tone detector.

TASK 1 — Is this a research query or a casual chat?
- CASUAL: greetings, jokes, "how are you", "what's 2+2", simple facts you know instantly
- RESEARCH: anything requiring web search, current events, comparisons, analysis, "how to", technical deep-dives

TASK 2 — Detect the exact writing tone from the query's wording:
- If the query says "explain like I'm 5" or "ELI5" → tone = "ELI5 — explain like I am 5 years old, use simple analogies"
- If it says "like a senior engineer" or "technical" → tone = "senior software engineer — precise, technical, no hand-holding"
- If it says "academic" or "research paper" → tone = "academic — formal, citations, structured"
- If it says "funny", "comedian", "make it fun" → tone = "funny — use humour and wit, be entertaining"
- If it says "simple", "beginner", "basic" → tone = "beginner-friendly — clear, no jargon"
- If it says "professional", "formal", "business" → tone = "professional and formal"
- If the query is phrased casually (lots of slang, emoji, exclamation) → tone = "casual and friendly"
- If the query is phrased neutrally with no tone cues → tone = "professional and informative"

Original query: "{query}"

Return ONLY a valid JSON object, no markdown:
{{"is_casual": <true|false>, "response": "<if casual: your answer, else empty string>", "tone": "<detected tone string>"}}
"""

@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=RETRY_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX))
def invoke_with_retry(llm, prompt):
    return llm.invoke(prompt)

def _detect_tone_heuristic(query: str) -> str:
    """Fast keyword-based fallback tone detector if LLM fails."""
    q = query.lower()
    if any(x in q for x in ["eli5", "explain like", "like i'm 5", "like i am 5", "simple", "beginner"]):
        return "ELI5 — explain like I am 5 years old, use simple analogies"
    if any(x in q for x in ["senior engineer", "technical", "in depth", "deep dive", "architecture"]):
        return "senior software engineer — precise, technical, no hand-holding"
    if any(x in q for x in ["academic", "research paper", "literature", "scholarly"]):
        return "academic — formal, citations, structured"
    if any(x in q for x in ["funny", "joke", "humour", "humor", "comedian", "fun"]):
        return "funny — use humour and wit, be entertaining"
    if any(x in q for x in ["professional", "formal", "business", "executive"]):
        return "professional and formal"
    if any(x in q for x in ["?", "how", "what", "why", "when", "which", "who"]):
        return "professional and informative"
    return "professional and informative"

def router_node(state: ResearchState) -> ResearchState:
    logger.info("Routing query to determine path and tone")
    query = state["query"]
    prompt = ROUTER_PROMPT.format(query=query)

    try:
        response = invoke_with_retry(llm, prompt)
        content = response.content.strip()
        # Strip markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()
        elif content.startswith("```"):
            content = content[3:].strip()
            if content.endswith("```"):
                content = content[:-3].strip()

        parsed = json.loads(content)
        state["is_casual"] = parsed.get("is_casual", False)
        state["tone"] = parsed.get("tone") or _detect_tone_heuristic(query)
        logger.info(f"Detected tone: {state['tone']}")

        if state["is_casual"]:
            logger.info("Query classified as casual. Bypassing research.")
            state["final_report"] = parsed.get("response", "I'm here to help! What's on your mind?")
        else:
            logger.info("Query requires research. Routing to Planner.")

    except Exception as e:
        logger.warning(f"Router LLM failed ({e}), using heuristic tone detection.")
        state["is_casual"] = False
        state["tone"] = _detect_tone_heuristic(query)
        logger.info(f"Heuristic tone: {state['tone']}")

    return state

if __name__ == "__main__":
    tests = [
        "Hello there! How are you?",
        "Explain microservices vs monolith like a senior engineer",
        "Explain quantum computing like I'm 5",
        "What is the state of AI regulation in 2025?",
        "Best practices for production RAG systems — make it funny",
        "AWS vs GCP for startups",
    ]
    for q in tests:
        s: ResearchState = {"query": q, "is_casual": False, "plan": [], "research_results": [],
                            "retrieved_docs": [], "critique": "", "confidence_score": 0.0,
                            "iteration_count": 0, "final_report": ""}
        r = router_node(s)
        print(f"Q: {q!r}\n  → casual={r['is_casual']}, tone={r['tone']!r}\n")
