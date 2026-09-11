import asyncio
from typing import Callable
from langchain_groq import ChatGroq
from core.state import ResearchState
from dotenv import load_dotenv
from core.metrics import metrics
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.logger import get_logger
from core.config import GROQ_REPORTER_MODEL, GROQ_API_KEY, RETRY_ATTEMPTS, RETRY_MULTIPLIER, RETRY_WAIT_MIN, RETRY_WAIT_MAX, AGENT_TIMEOUT

load_dotenv()

logger = get_logger(__name__)

# Use a separate, lighter model for report writing to avoid rate limits on groq/compound
llm = ChatGroq(
    model=GROQ_REPORTER_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.3,
)

# Minimum confidence below which we still write a report, but force strict grounding
LOW_CONFIDENCE_THRESHOLD = 0.5

# ── Token caps (conservative to avoid context overflow and token limit errors) ──
# Each char ≈ 0.25 tokens for English text. These caps keep total prompt well under 8k tokens.
RESEARCH_CHAR_CAP = 3000   # ~750 tokens of research findings
DOCS_CHAR_CAP = 2000       # ~500 tokens of retrieved docs
PLAN_CHAR_CAP = 800        # ~200 tokens of plan

REPORTER_PROMPT = """You are writing a research report. Follow these instructions strictly.

## CRITICAL: NO HALLUCINATION RULE (apply this before everything else)
You MUST ONLY use facts, numbers, dates, names, and URLs that are EXPLICITLY present in the
RESEARCH FINDINGS and TOP RETRIEVED DOCUMENTS sections below.
- DO NOT invent statistics, dates, prices, version numbers, or product names.
- DO NOT add outside knowledge that is not in the research data below.
- If a fact is not found in the research data, write "information not available" instead of guessing.
- The "## Sources" section MUST only contain URLs that literally appear in the research data below.
  Do not fabricate URLs or domain names.
- A shorter accurate report is ALWAYS better than a longer hallucinated one.

## TONE (apply this after the no-hallucination rule)
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

## RESEARCH FINDINGS (use ONLY these facts — do NOT add outside knowledge)
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
- Use ONLY real data from the research above — no invented examples
- End with a "## Bottom Line" section: one sharp paragraph wrapping up the key takeaway
- Final section "## Sources": bullet list of ONLY the markdown links whose URLs appear in the research data above

Write the full report now in the tone: **{tone}**
"""

LOW_CONFIDENCE_APPENDIX = """

IMPORTANT: The research for this query was limited or incomplete (confidence score: {confidence:.2f}).
Only state facts explicitly present in the research findings and retrieved documents above.
Do NOT add outside knowledge, invented statistics, dates, prices, or assumed details.
If information is missing or unclear, write "information not available" in the report instead of filling the gap.
Add a short ⚠️ note near the top flagging that research coverage was limited.
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
    """Build the reporter prompt, applying token caps and optionally appending strict-grounding warning."""
    prompt = REPORTER_PROMPT.format(
        tone=tone,
        query=query,
        plan=plan_text[:PLAN_CHAR_CAP],
        research_results=research_text[:RESEARCH_CHAR_CAP],
        retrieved_docs=docs_text[:DOCS_CHAR_CAP],
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
    loop = asyncio.get_running_loop()
    logger.info("Writing final report")

    plan_text = "\n".join(state.get("plan", []))
    research_list = state.get("research_results", [])
    docs_list = state.get("retrieved_docs", [])
    # Aggressive truncation to stay under token limits and prevent hallucination from context overflow
    research_text = "\n".join(research_list)[:RESEARCH_CHAR_CAP]
    docs_text = "\n".join(docs_list)[:DOCS_CHAR_CAP]
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


async def reporter_node_streaming(
    state: ResearchState,
    chunk_callback: Callable
) -> ResearchState:
    """Streaming variant of reporter_node.

    Streams LLM tokens via chunk_callback as they are generated, then
    stores the completed report in state["final_report"].

    Args:
        state: The current ResearchState.
        chunk_callback: An async callable that receives each text chunk string.
    """
    logger.info("Writing final report (streaming mode)")

    plan_text = "\n".join(state.get("plan", []))
    research_list = state.get("research_results", [])
    docs_list = state.get("retrieved_docs", [])
    research_text = "\n".join(research_list)[:RESEARCH_CHAR_CAP]
    docs_text = "\n".join(docs_list)[:DOCS_CHAR_CAP]
    tone = state.get("tone", "super friendly and conversational")
    confidence = state.get("confidence_score", 0)
    has_research = bool(research_list) or bool(docs_list)

    if not has_research:
        logger.warning(
            f"No research data at all (confidence={confidence}) — using honest fallback")
        fallback = _no_research_fallback(state["query"], plan_text, confidence)
        state["final_report"] = fallback
        # Stream fallback as a single chunk so UI updates
        await chunk_callback(fallback)
        metrics.end_agent("reporter", error="no_research_data")
        return state

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        logger.warning(
            f"Low confidence research (confidence={confidence}) — strict grounding mode")

    prompt = _build_prompt(
        tone, state["query"], plan_text, research_text, docs_text, confidence)

    full_report = ""

    try:
        async for chunk in llm.astream(prompt):
            text = chunk.content
            if text:
                full_report += text
                await chunk_callback(text)

        state["final_report"] = full_report.strip()
        logger.info(f"Streaming report complete — {len(full_report)} chars")
        metrics.end_agent("reporter", input_tokens=len(prompt), output_tokens=len(full_report))

    except Exception as e:
        err_type = type(e).__name__
        logger.error(f"Streaming LLM failed ({err_type}): {e}")
        # If we got partial output, use it; otherwise serve error fallback
        if full_report.strip():
            logger.warning("Using partial streaming output as report")
            state["final_report"] = full_report.strip()
        else:
            fallback = _error_fallback(state["query"], plan_text, research_text, e)
            state["final_report"] = fallback
            await chunk_callback(fallback)
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
