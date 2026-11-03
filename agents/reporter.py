import asyncio
import re
from datetime import datetime
from urllib.parse import urlparse
from typing import Callable, Dict, Any, List
from langchain_groq import ChatGroq
from core.state import ResearchState
from dotenv import load_dotenv
from core.metrics import metrics
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.logger import get_logger
from core.config import GROQ_REPORTER_MODEL, GROQ_API_KEY, RETRY_ATTEMPTS, RETRY_MULTIPLIER, RETRY_WAIT_MIN, RETRY_WAIT_MAX, AGENT_TIMEOUT

load_dotenv()
logger = get_logger(__name__)

llm = ChatGroq(
    model=GROQ_REPORTER_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.15,
)

LOW_CONFIDENCE_THRESHOLD = 0.50

REPORTER_STRICT_PROMPT = """You are an elite, objective research analyst.
Your job is to synthesize a factual, impeccably grounded research report using ONLY the numbered evidence chunks provided below.

============================================================
CRITICAL ACCURACY & GROUNDING RULES (VIOLATION IS A FAILURE)
============================================================
1. ONLY USE NUMBERED SOURCES: Rely exclusively on the numbered chunks [1], [2], [3]... below. NEVER use outside knowledge.
2. CITATION FORMAT: Always write citations as [1], [2], [3] etc. NEVER use 【1†...】, (1), or any other format.
3. INLINE NUMBERED CITATIONS: Every factual sentence MUST end with a bracketed citation pointing to the exact supporting chunk, e.g. "Model Y was the top selling vehicle globally in 2023 [1]."
4. HARD NUMBERS & DATES: State the exact date or timeframe next to every statistic, revenue figure, or metric. If a source is older than 12 months, explicitly state the year.
5. ZERO INVENTION: If the sources do not cover an aspect of the query, write "Not found in the verified sources" instead of guessing or filling gaps.
6. CONFLICTING EVIDENCE: If two sources disagree, show both perspectives and cite each: "Source [1] reported X, while Source [3] reported Y."
7. NO SPECULATION: Do not make unverified predictions or call a product "dominant", "leading", or "revolutionary" unless an explicit source states it.
8. NO REPETITION: Mention each fact ONCE. Do NOT repeat the same facts across summary, details, and tables.
9. FORMATTING HYGIENE: Avoid over-bolding. Do not bold every second word — only bold section titles, key entity names, and primary metrics.

============================================================
MANDATORY FIXED REPORT TEMPLATE (FOLLOW THIS EXACT STRUCTURE)
============================================================

## Summary
(3-4 lines: concise direct answer to the user's question, cited with [n])

## Key findings
(4-6 scannable bullet points covering the main insights, each ending with a citation like [n])

## Details
(One subsection per sub-task analyzing the core mechanisms, technology, or business data, cited thoroughly with [n])

## Data / timeline
(Include a clean Markdown table ONLY if you have 3 or more rows of real sourced numbers or dates. If fewer than 3 rows exist, omit the table and write a single factual sentence)

## Gaps and caveats
(Honest statement of what the sources did not cover, unresolved debates, or outdated data)

## Sources
(Numbered list — this will be replaced by the system with verified source entries. Write a placeholder line per citation you used, e.g. "[1] Source title — domain")

Tone to maintain: {tone}

USER QUERY:
{query}

PLANNED SUB-TASKS:
{plan}

NUMBERED RETRIEVED EVIDENCE CHUNKS:
{retrieved_docs}

Write the report now adhering strictly to the template and citation rules above:"""


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=RETRY_MULTIPLIER,
                          min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def invoke_with_retry(llm_instance, prompt):
    return llm_instance.invoke(prompt)


def _build_deterministic_sources_section(chunks: List[Dict[str, Any]]) -> str:
    """Build a deterministic, un-hallucinated ## Sources list mapped by [n]."""
    if not chunks:
        return "## Sources\n*No external sources verified.*"

    sources_lines = ["## Sources"]
    seen_ids = set()
    for ch in chunks:
        cid = ch.get("id")
        if not cid or cid in seen_ids:
            continue
        seen_ids.add(cid)
        title = ch.get("title", "Source").replace("[", "(").replace("]", ")")
        url = ch.get("url", "#")
        domain = ch.get("domain", "") or urlparse(
            url).netloc.lower().lstrip("www.")
        date = ch.get("date", "Recent")
        sources_lines.append(f"[{cid}] [{title}]({url}) — *{domain}* ({date})")

    return "\n".join(sources_lines)


# Domains considered highly credible — score 1.0
_TRUSTED_DOMAINS = {
    ".edu", ".gov", ".ac.uk", ".ac.in",
    "nature.com", "science.org", "arxiv.org", "pubmed.ncbi", "pmc.ncbi",
    "reuters.com", "bbc.com", "bbc.co.uk", "nytimes.com", "wsj.com",
    "wired.com", "techcrunch.com", "forbes.com", "bloomberg.com",
    "who.int", "cdc.gov", "nih.gov", "europa.eu",
    "ieee.org", "acm.org", "springer.com", "sciencedirect.com",
    "clevelandclinic.org", "mayoclinic.org",
}

# Domains to deprioritise (sales / market-research paywalls)
_LOW_QUALITY_DOMAINS = {
    "idtechex.com", "researchandmarkets.com", "transparencymarketresearch.com",
    "marketresearchfuture.com", "grandviewresearch.com",
    "fortunebusinessinsights.com", "mordorintelligence.com",
    "expertmarketresearch.com", "verifiedmarketresearch.com",
}

MAX_SOURCES = 9   # hard cap shown to LLM and in Sources section


def _domain_score(url: str) -> float:
    """Return a credibility score [0.0–1.0] for a URL's domain."""
    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return 0.4
    if any(domain.endswith(t) or t in domain for t in _TRUSTED_DOMAINS):
        return 1.0
    if any(bad in domain for bad in _LOW_QUALITY_DOMAINS):
        return 0.3
    return 0.6


def _parse_date_score(date_str: str) -> float:
    """Convert a date string to a freshness score (0–1).  Newer = higher."""
    if not date_str or date_str.lower() == "recent":
        return 0.7  # unknown date is neutral-positive
    # RFC-2822 or ISO-8601 patterns
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(date_str[:len(fmt)+4].strip(), fmt)
            age_days = (datetime(2026, 11, 1) - dt).days
            # 0 days old → 1.0;  730+ days (2 years) → 0.0
            return max(0.0, 1.0 - age_days / 730)
        except ValueError:
            pass
    return 0.5


def _select_top_sources(chunks: List[Dict[str, Any]],
                        max_n: int = MAX_SOURCES) -> List[Dict[str, Any]]:
    """
    Deduplicate chunks by URL, rank by (domain credibility + freshness),
    keep the top max_n unique sources, and renumber them 1..N so the LLM
    and the Sources section always show exactly max_n clean entries.

    Returns a NEW list of chunk dicts with consecutive integer 'id' values.
    """
    if not chunks:
        return []

    # --- 1. Group all chunks by canonical URL ---
    url_to_best: Dict[str, Dict[str, Any]] = {}
    for ch in chunks:
        url = ch.get("url", "") or ""
        # Normalise: strip trailing slash and fragment
        canon_url = url.split("#")[0].rstrip("/")
        if not canon_url:
            canon_url = ch.get("title", "") or str(ch.get("id", ""))

        existing = url_to_best.get(canon_url)
        if existing is None:
            url_to_best[canon_url] = ch
        else:
            # Keep the chunk with more text (richer content)
            if len(ch.get("text", "")) > len(existing.get("text", "")):
                url_to_best[canon_url] = ch

    # --- 2. Score each unique source ---
    scored: List[tuple] = []
    for canon_url, ch in url_to_best.items():
        d_score = _domain_score(canon_url)
        f_score = _parse_date_score(ch.get("date", ""))
        # Weighted: credibility matters more than freshness
        total = round(0.65 * d_score + 0.35 * f_score, 4)
        scored.append((total, ch))

    # --- 3. Sort descending, keep top max_n ---
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [ch for _, ch in scored[:max_n]]

    # --- 4. Renumber sequentially 1..N ---
    renumbered = []
    for new_id, ch in enumerate(top_chunks, start=1):
        new_ch = dict(ch)   # shallow copy — don't mutate original
        new_ch["id"] = new_id
        renumbered.append(new_ch)

    logger.info(
        f"Source deduplication: {len(chunks)} chunks → "
        f"{len(url_to_best)} unique URLs → {len(renumbered)} selected"
    )
    return renumbered


def _validate_citations_and_numbers(report_text: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Deterministic code check:
    1. Normalises OpenAI-style 【n†...】 citations → [n] format.
    2. Ensures all cited [n] exist in the retrieved set; removes hallucinated IDs.
    3. Checks numbers in the report against retrieved chunks to detect hallucinations.
    4. Replaces or appends the deterministic ## Sources section.
    """
    valid_ids = {str(ch.get("id")) for ch in chunks if ch.get("id")}
    combined_chunk_text = " ".join(ch.get("text", "") for ch in chunks).lower()

    # Strip existing ## Sources section generated by LLM to replace with deterministic list
    report_body = re.split(r'\n##\s*Sources', report_text,
                           flags=re.IGNORECASE)[0].strip()

    # 0. Normalise 【n†L1-L9】 or 【n†source】 → [n]  (OpenAI-style citations)
    report_body = re.sub(r'【(\d+)[†\u2020][^】]*】', r'[\1]', report_body)
    # Also handle 【n】 bare form
    report_body = re.sub(r'【(\d+)】', r'[\1]', report_body)

    # 1. Clean invalid [n] citations (e.g. LLM invented [99])
    def replace_citation(match):
        cid = match.group(1)
        if cid in valid_ids:
            return f"[{cid}]"
        return ""  # remove hallucinated citation ID

    report_body = re.sub(r'\[(\d+)\]', replace_citation, report_body)

    # 2. Audit numbers (e.g. $54.9M, 400 Wh/kg, 2026, 1.81 million)
    unverified_numbers = []
    number_claims = re.findall(
        r'(\b\d+(?:\.\d+)?%?\b(?:\s+[a-zA-Z]+){1,3})', report_body)
    for claim in number_claims[:12]:
        num_match = re.search(r'\b\d+(?:\.\d+)?', claim)
        if num_match and num_match.group(0) not in combined_chunk_text:
            unverified_numbers.append(claim.strip())

    if unverified_numbers:
        logger.warning(
            f"Unverified numbers found in report: {unverified_numbers[:3]}")

    # 3. Append deterministic Sources section
    sources_section = _build_deterministic_sources_section(chunks)
    final_output = f"{report_body}\n\n{sources_section}"
    return final_output


async def reporter_node(state: ResearchState) -> ResearchState:
    """Async reporter node with strict template, fact verification, and deterministic citation mapping."""
    loop = asyncio.get_running_loop()
    logger.info("Writing final research report with strict template")

    plan_text = "\n".join(f"- {p}" for p in state.get("plan", []))
    raw_chunks = state.get("retrieved_chunks", [])
    docs_list = state.get("retrieved_docs", [])
    tone = state.get("tone", "professional and informative")

    if not docs_list and not raw_chunks:
        logger.warning("No research documents available for report writing")
        state["final_report"] = f"# Research Report: {state['query']}\n\n> ⚠️ **No verified research findings available** for this query.\n"
        metrics.end_agent("reporter", error="no_research_data")
        return state

    # ── Deduplicate & rank: collapse same-URL chunks, keep top 7-9 sources ──
    chunks = _select_top_sources(raw_chunks, max_n=MAX_SOURCES)
    if not chunks:
        # Fallback if chunk metadata is incomplete — use raw list, capped
        chunks = raw_chunks[:MAX_SOURCES]
        for i, ch in enumerate(chunks, start=1):
            ch = dict(ch)
            ch["id"] = i

    # Build numbered evidence text for the LLM from deduplicated sources only
    def _fmt_chunk(ch: Dict[str, Any]) -> str:
        return (
            f"[{ch['id']}] Title: {ch.get('title','')}\n"
            f"URL: {ch.get('url','')}\n"
            f"Date: {ch.get('date','Recent')}\n"
            f"Text: {ch.get('text','')[:800]}"
        )

    docs_text = "\n\n".join(_fmt_chunk(ch) for ch in chunks)

    prompt = REPORTER_STRICT_PROMPT.format(
        tone=tone,
        query=state["query"],
        plan=plan_text,
        retrieved_docs=docs_text
    )

    print(
        f"[DEBUG REPORTER] Invoking Reporter LLM ({GROQ_REPORTER_MODEL})...", flush=True)
    try:
        response = await asyncio.wait_for(
            loop.run_in_executor(None, invoke_with_retry, llm, prompt),
            timeout=AGENT_TIMEOUT
        )
        raw_report = response.content.strip()
        print(
            f"[DEBUG REPORTER] Reporter LLM generated {len(raw_report.split())} words.", flush=True)

        # Deterministic citation and number validation pass
        validated_report = _validate_citations_and_numbers(raw_report, chunks)

        state["final_report"] = validated_report
        logger.info(
            f"Report generated and validated successfully ({len(validated_report.split())} words)")

        metrics.end_agent(
            "reporter",
            input_tokens=len(prompt),
            output_tokens=len(validated_report)
        )
    except asyncio.TimeoutError:
        logger.error(f"Reporter timed out after {AGENT_TIMEOUT}s")
        sources_sec = _build_deterministic_sources_section(chunks)
        state["final_report"] = f"# Research Report: {state['query']}\n\n> ⚠️ Report generation timed out.\n\n{sources_sec}"
        metrics.end_agent("reporter", error="timeout")
    except Exception as e:
        logger.error(f"Reporter failed: {e}")
        sources_sec = _build_deterministic_sources_section(chunks)
        state["final_report"] = f"# Research Report: {state['query']}\n\n> ⚠️ Report writer encountered an error: {e}\n\n{sources_sec}"
        metrics.end_agent("reporter", error=str(e))

    return state


async def reporter_node_streaming(state: ResearchState, chunk_callback: Callable) -> ResearchState:
    """Streaming variant that streams generated tokens and validates final output."""
    loop = asyncio.get_running_loop()
    logger.info("Streaming final report with strict template")

    plan_text = "\n".join(f"- {p}" for p in state.get("plan", []))
    raw_chunks = state.get("retrieved_chunks", [])
    tone = state.get("tone", "professional and informative")

    # Deduplicate & rank sources — same logic as reporter_node
    chunks = _select_top_sources(raw_chunks, max_n=MAX_SOURCES)
    if not chunks:
        chunks = raw_chunks[:MAX_SOURCES]

    def _fmt_chunk(ch: Dict[str, Any]) -> str:
        return (
            f"[{ch['id']}] Title: {ch.get('title','')}\n"
            f"URL: {ch.get('url','')}\n"
            f"Date: {ch.get('date','Recent')}\n"
            f"Text: {ch.get('text','')[:800]}"
        )

    docs_text = "\n\n".join(_fmt_chunk(ch) for ch in chunks)

    prompt = REPORTER_STRICT_PROMPT.format(
        tone=tone,
        query=state["query"],
        plan=plan_text,
        retrieved_docs=docs_text
    )

    full_response = []
    try:
        async for chunk in llm.astream(prompt):
            content = chunk.content
            if content:
                full_response.append(content)
                if asyncio.iscoroutinefunction(chunk_callback):
                    await chunk_callback(content)
                else:
                    chunk_callback(content)

        raw_report = "".join(full_response).strip()
        state["final_report"] = _validate_citations_and_numbers(
            raw_report, chunks)
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        state = await reporter_node(state)

    return state
