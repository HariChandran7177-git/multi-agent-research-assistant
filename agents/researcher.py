# ruff: noqa: E402
import os
import asyncio
import concurrent.futures
import re
from urllib.parse import urlparse
from typing import Dict, Any, List

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import (
    GROQ_RESEARCHER_MODEL,
    GROQ_API_KEY,
    RETRY_ATTEMPTS,
    RETRY_MULTIPLIER,
    RETRY_WAIT_MIN,
    RETRY_WAIT_MAX,
    TAVILY_MAX_RESULTS,
    AGENT_TIMEOUT,
)
from core.state import ResearchState
from core.logger import get_logger
from core.metrics import metrics

load_dotenv()

logger = get_logger(__name__)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

llm = ChatGroq(
    model=GROQ_RESEARCHER_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.1
)

# Known low-quality / paywalled sales / ad-heavy domains to deprioritize
BLOCKED_OR_PENALIZED_DOMAINS = {
    "idtechex.com", "researchandmarkets.com", "transparencymarketresearch.com",
    "marketresearchfuture.com", "verifiedmarketresearch.com", "grandviewresearch.com",
    "expertmarketresearch.com", "fortunebusinessinsights.com"
}

_TAVILY_CACHE = {}


@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=RETRY_MULTIPLIER,
        min=RETRY_WAIT_MIN,
        max=RETRY_WAIT_MAX
    )
)
def search_with_retry(client: TavilyClient, task: str) -> Dict[str, Any]:
    """
    Perform a Tavily search with automatic retries and caching.
    Uses search_depth='advanced' and news topic detection for fresh current events.
    """
    clean_task = task.strip().lstrip("0123456789.-*#\t ")
    if clean_task in _TAVILY_CACHE:
        return _TAVILY_CACHE[clean_task]

    # Detect if freshness / news filter is needed (e.g. 2025, 2026, latest, trends, delivery)
    is_news_or_recent = any(kw in clean_task.lower() for kw in ["2025", "2026", "latest", "recent", "trends", "deliveries", "quarter", "q1", "q2", "q3", "q4"])
    
    search_kwargs = {
        "query": clean_task,
        "max_results": TAVILY_MAX_RESULTS,
        "search_depth": "advanced",
        "include_raw_content": False,
    }
    
    if is_news_or_recent:
        search_kwargs["topic"] = "news"
        search_kwargs["days"] = 365

    try:
        result = client.search(**search_kwargs)
    except Exception as e:
        # Fallback to general search if news filter raises error
        if is_news_or_recent:
            search_kwargs.pop("topic", None)
            search_kwargs.pop("days", None)
            result = client.search(**search_kwargs)
        else:
            raise e

    _TAVILY_CACHE[clean_task] = result
    return result


def _normalize_fingerprint(text: str) -> str:
    """Create normalized 80-char fingerprint to filter duplicate snippets."""
    return re.sub(r"\s+", " ", text.lower().strip())[:80]


async def _process_task(task: str, executor: concurrent.futures.Executor) -> Dict[str, Any]:
    """Execute Tavily search for one research sub-task with domain capping and duplicate filtering."""
    loop = asyncio.get_event_loop()

    def _do_search():
        try:
            search_response = search_with_retry(tavily, task)
            raw_results = search_response.get("results", [])

            processed_items = []
            task_sources = []
            domain_counts = {}
            seen_fingerprints = set()

            for res in raw_results:
                title = res.get("title", "Untitled Source").strip()
                url = res.get("url", "").strip()
                content = res.get("content", "").strip()
                published_date = res.get("published_date") or res.get("publishedDate") or "Recent"

                if not url or not content or len(content) < 50:
                    continue

                domain = urlparse(url).netloc.lower().lstrip("www.")

                # Filter paywalled sales / low-value market pitch aggregators
                if domain in BLOCKED_OR_PENALIZED_DOMAINS:
                    logger.debug(f"Skipping penalized sales domain: {domain}")
                    continue

                # Domain capping: max 2 passages per domain per subtask
                if domain_counts.get(domain, 0) >= 2:
                    logger.debug(f"Domain cap reached for: {domain}")
                    continue

                # Near-duplicate passage filtering
                fingerprint = _normalize_fingerprint(content)
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)

                domain_counts[domain] = domain_counts.get(domain, 0) + 1

                formatted_item = (
                    f"Subtask: {task}\n"
                    f"Title: {title}\n"
                    f"Domain: {domain}\n"
                    f"Date: {published_date}\n"
                    f"URL: {url}\n"
                    f"Content: {content}"
                )
                processed_items.append(formatted_item)
                task_sources.append({
                    "title": title,
                    "url": url,
                    "domain": domain,
                    "date": str(published_date)
                })

            if not processed_items:
                logger.info(f"No valid results found for: {task}")
                return {"findings": [], "sources": []}

            logger.info(f"Completed search for: {task} ({len(processed_items)} quality snippets)")
            return {
                "findings": processed_items,
                "sources": task_sources
            }
        except Exception as e:
            logger.warning(f"Failed on task '{task}' after retries: {e}")
            return {"findings": [], "sources": []}

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, _do_search),
            timeout=AGENT_TIMEOUT
        )
        return result or {"findings": [], "sources": []}
    except asyncio.TimeoutError:
        logger.error(f"Task '{task}' timed out after {AGENT_TIMEOUT}s — skipping")
        return {"findings": [], "sources": []}


async def researcher_node(state: ResearchState) -> ResearchState:
    """
    Researcher node without lossy truncation.
    Gathers structured factual evidence tagged by sub-task and passes it directly to Qdrant/Retriever.
    """
    plan = state.get("plan", [])
    if not plan:
        plan = [state.get("query", "")]

    all_findings = []
    all_sources = []

    logger.info(f"Starting parallel research for {len(plan)} sub-tasks")

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(plan) * 2, 10)) as executor:
        async_tasks = [_process_task(task, executor) for task in plan]
        results = await asyncio.gather(*async_tasks, return_exceptions=True)

        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Task failed with exception: {res}")
            elif res and res.get("findings"):
                all_findings.extend(res["findings"])
                all_sources.extend(res.get("sources", []))

    # Deduplicate sources by URL
    unique_sources_map = {}
    for s in all_sources:
        u = s.get("url")
        if u and u not in unique_sources_map:
            unique_sources_map[u] = s

    state["sources"] = list(unique_sources_map.values())
    state["research_results"] = all_findings

    logger.info(f"Research gathering complete: {len(all_findings)} structured passages preserved (zero truncation).")

    metrics.end_agent(
        "researcher",
        input_tokens=sum(len(p) for p in plan),
        output_tokens=sum(len(f) for f in all_findings)
    )

    return state
