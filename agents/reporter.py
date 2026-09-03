import asyncio
from langchain_groq import ChatGroq
from core.state import ResearchState
import os
from dotenv import load_dotenv
from core.metrics import metrics
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.logger import get_logger
from core.config import GROQ_MODEL, GROQ_REPORTER_MODEL, GROQ_API_KEY, RETRY_ATTEMPTS, RETRY_MULTIPLIER, RETRY_WAIT_MIN, RETRY_WAIT_MAX, AGENT_TIMEOUT

load_dotenv()

logger = get_logger(__name__)

# Use a separate, lighter model for report writing to avoid rate limits on groq/compound
llm = ChatGroq(
    model=GROQ_REPORTER_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.4,
)

# Minimum confidence below which we still write a report, but force strict grounding
LOW_CONFIDENCE_THRESHOLD = 0.5

REPORTER_PROMPT = """You are writing a research report. Follow these instructions in order.

## TONE (follow this first, before anything else)
Write in this exact style: **{tone}**
Examples of what this means in practice:
- "ELI5": use analogies like "imagine a pizza delivery...", zero jargon
- "senior software engineer": be precise, skip basics, use technical terms freely
- "academic": cite sources inline, use formal language, structured sections
- "funny": use wit and humour while still being accurate
- "professional and informative": clear, factual, no filler phrases
- "storyteller": frame the explanation as a narrative with a beginning, middle, and end
- "skeptical analyst": question claims, note where evidence is thin, avoid overstating certainty
- "casual friend": relaxed, first-person asides, contractions, no corporate tone
- "executive briefing": lead with the bottom line, keep it tight, use short punchy bullets

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

LOW_CONFIDENCE_APPENDIX = """

IMPORTANT: The research for this query was limited or incomplete (confidence score: {confidence:.2f}). Only state facts explicitly present in the research findings and retrieved documents above. Do not add outside knowledge, invented statistics, dates, prices, or assumed details. If information is missing or unclear, say so directly in the report instead of filling the gap. Add a short note near the top flagging that research coverage was limited.
"""


def _is_rate_limit(exc):
    return "rate" in str(exc).lower() or "429" in str(exc)


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_MULTIPLIER,
                          min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def invoke_with_retry(llm, prompt):
    return llm.invoke(prompt)


def _build_prompt(tone, query, plan_text, research_text, docs_text, confidence=None):
    """Build the reporter prompt, optionally appending a strict-grounding warning for low-confidence research."""
    prompt = REPORTER_PROMPT.format(
        tone=tone,
        query=query,
        plan=plan_text,
        research_results=research_text,
        retrieved_docs=docs_text,
    )
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        prompt += LOW_CONFIDENCE_APPENDIX.format(confidence=confidence)
    return prompt


def _no_research_fallback(query, plan_text, confidence):
    """Honest fallback report when no research data was gathered at all."""
    return f"""# {query}

> ⚠️ **Limited research available** — search and retrieval did not return verified sources for this query (possible cause: API rate limits or credit exhaustion). Confidence score: {confidence:.2f}

## What I can tell you
I wasn't able to gather real-time, sourced information on this topic right now. Rather than guess, here's what I'd recommend:
- Try the query again in a few minutes (search API limits often reset quickly)
- Rephrase the query to be more specific
- If this persists, check that your search API key/credits are active

## Research Plan (what was attempted)
{plan_text[:1000] if plan_text else "No plan was generated."}

---
*No report was generated from unverified sources, to avoid presenting fabricated information as fact.*
"""


def _timeout_fallback(query, plan_text, research_text):
    return f"""# Research Results for: {query}

> ⚠️ The report writer timed out after {AGENT_TIMEOUT} seconds.

## Research Plan
{plan_text[:1000]}

## Key Findings
{research_text[:1500]}

---
*Report generation timed out. Please try again.*
"""


def _error_fallback(query, plan_text, research_text, error):
    return f"""# Research Results for: {query}

> ⚠️ The report writer encountered an error: {str(error)[:100]}

## Research Plan
{plan_text[:1000]}

## Key Findings
{research_text[:1500]}

---
*Report generation failed. Please check your API keys and try again.*
"""


async def reporter_node(state: ResearchState) -> ResearchState:
    """Async reporter node with timeout, metrics, and grounding safeguards.

    Behavior:
    - No research data at all -> honest fallback, no LLM call (saves credits, avoids hallucination)
    - Some research but confidence below LOW_CONFIDENCE_THRESHOLD -> write report, but force strict
      grounding to retrieved data only, and flag the limitation in the output
    - Confidence at/above threshold -> normal report generation
    """
    loop = asyncio.get_event_loop()
    logger.info("Writing final report")

    plan_text = "\n".join(state.get("plan", []))
    research_list = state.get("research_results", [])
    docs_list = state.get("retrieved_docs", [])
    # truncate aggressively to stay under token limits
    research_text = "\n".join(research_list)[:4000]
    docs_text = "\n".join(docs_list)[:3000]
    tone = state.get("tone", "super friendly and conversational")
    confidence = state.get("confidence_score", 0)

    has_research = bool(research_list) or bool(docs_list)

    if not has_research:
        logger.warning(
            f"No research data at all (confidence={confidence}) — returning honest limited response")
        state["final_report"] = _no_research_fallback(
            state["query"], plan_text, confidence)
        metrics.end_agent("reporter", error="no_research_data")
        return state

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        logger.warning(
            f"Low confidence research (confidence={confidence}) — writing report with strict grounding")

    prompt = _build_prompt(
        tone, state["query"], plan_text, research_text, docs_text, confidence)

    try:
        response = await asyncio.wait_for(
            loop.run_in_executor(None, invoke_with_retry, llm, prompt),
            timeout=AGENT_TIMEOUT
        )
        state["final_report"] = response.content.strip()
        logger.info("Report generated successfully")

        input_tokens = len(prompt)
        output_tokens = len(response.content)
        metrics.end_agent("reporter", input_tokens=input_tokens,
                          output_tokens=output_tokens)

    except asyncio.TimeoutError:
        logger.error(f"Reporter timeout after {AGENT_TIMEOUT}s")
        state["final_report"] = _timeout_fallback(
            state["query"], plan_text, research_text)
        metrics.end_agent("reporter", error="timeout")

    except Exception as e:
        logger.error(f"LLM call failed after retries: {e}")
        state["final_report"] = _error_fallback(
            state["query"], plan_text, research_text, e)
        metrics.end_agent("reporter", error=str(e))

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
    result = asyncio.run(reporter_node(test_state))
    print(result["final_report"])
