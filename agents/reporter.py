from langchain_groq import ChatGroq
from core.state import ResearchState
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from core.logger import get_logger

load_dotenv()

from core.config import GROQ_MODEL, GROQ_REPORTER_MODEL, GROQ_API_KEY, RETRY_ATTEMPTS, RETRY_MULTIPLIER, RETRY_WAIT_MIN, RETRY_WAIT_MAX

logger = get_logger(__name__)

# Use a separate, lighter model for report writing to avoid rate limits on groq/compound
llm = ChatGroq(
    model=GROQ_REPORTER_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.4,
)

REPORTER_PROMPT = """You are writing a research report. Follow these instructions in order.

## TONE (follow this first, before anything else)
Write in this exact style: **{tone}**
Examples of what this means in practice:
- "ELI5": use analogies like "imagine a pizza delivery...", zero jargon
- "senior software engineer": be precise, skip basics, use technical terms freely
- "academic": cite sources inline, use formal language, structured sections
- "funny": use wit and humour while still being accurate
- "professional and informative": clear, factual, no filler phrases

## YOUR QUERY
{query}

## RESEARCH PLAN (sub-tasks that were investigated)
{plan}

## RESEARCH FINDINGS (use specific facts, names, numbers from here)
{research_results}

## TOP RETRIEVED DOCUMENTS
{retrieved_docs}

## FORMAT RULES
Choose format based on what the content needs — not habit:
- Use a **comparison table** if comparing 2+ options
- Use **numbered steps** if explaining a process
- Use **bullet points** for scannable key facts
- Use **paragraphs** for explanations and analysis
- Do NOT use generic headers like "Introduction" or "Conclusion" — use content-specific headers

## CONTENT RULES
- Open by directly addressing the query — no preamble
- Bold specific names, numbers, and key facts: **AWS holds 31% market share**
- Use real data from the research above — no invented examples
- End with a "## Bottom Line" section: one sharp paragraph wrapping up the key takeaway
- Final section "## Sources": bullet list of markdown links from URLs in the research data

Write the full report now in the tone: **{tone}**
"""


from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def _is_rate_limit(exc):
    return "rate" in str(exc).lower() or "429" in str(exc)

@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def invoke_with_retry(llm, prompt):
    return llm.invoke(prompt)


def reporter_node(state: ResearchState) -> ResearchState:
    logger.info("Writing final report")

    plan_text = "\n".join(state.get("plan", []))
    # Truncate aggressively to stay under Groq token limits
    research_text = "\n".join(state.get("research_results", []))[:4000]
    docs_text = "\n".join(state.get("retrieved_docs", []))[:3000]
    tone = state.get("tone", "super friendly and conversational")

    prompt = REPORTER_PROMPT.format(
        tone=tone,
        query=state["query"],
        plan=plan_text,
        research_results=research_text,
        retrieved_docs=docs_text,
    )

    try:
        response = invoke_with_retry(llm, prompt)
        state["final_report"] = response.content.strip()
        logger.info("Report generated successfully")
    except Exception as e:
        logger.error(f"LLM call failed after retries: {e}")
        # Return a meaningful markdown fallback so the frontend still renders
        state["final_report"] = f"""# Research Results for: {state['query']}

> ⚠️ The report writer hit a rate limit. Here are the raw research findings:

## Research Plan
{plan_text}

## Key Findings
{research_text[:2000]}

---
*Report generation failed due to API rate limits. Try again in a minute.*
"""

    return state



if __name__ == "__main__":
    test_state: ResearchState = {
        "query": "What are the latest advancements in quantum computing?",
        "plan": ["Search recent breakthroughs", "Find key companies", "Identify challenges"],
        "research_results": ["IBM announced a 1000-qubit chip in 2023.", "Quantum error correction remains a major hurdle."],
        "retrieved_docs": ["IBM's Condor processor has 1,121 qubits."],
        "critique": "Good coverage of major players.",
        "confidence_score": 0.85,
        "iteration_count": 1,
        "final_report": "",
    }
    result = reporter_node(test_state)
    print(result["final_report"])
