# ruff: noqa: E402

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
import os
import asyncio
import concurrent.futures

from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from core.state import ResearchState
from core.logger import get_logger
from core.metrics import metrics

load_dotenv()


logger = get_logger(__name__)

tavily = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

llm = ChatGroq(
    model=GROQ_RESEARCHER_MODEL,
    api_key=GROQ_API_KEY
)


# ============================================================
# TAVILY RETRY & CACHE
# ============================================================

_TAVILY_CACHE = {}

@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=RETRY_MULTIPLIER,
        min=RETRY_WAIT_MIN,
        max=RETRY_WAIT_MAX
    )
)
def search_with_retry(client, task):
    """
    Perform a Tavily search with automatic retries
    if the request temporarily fails.
    Caches results to avoid redundant web searches on loops.
    """

    # Remove numbering/bullets from Planner's task
    clean_task = task.strip().lstrip(
        "0123456789.-*#\t "
    )
    
    if clean_task in _TAVILY_CACHE:
        return _TAVILY_CACHE[clean_task]

    result = client.search(
        query=clean_task,
        max_results=TAVILY_MAX_RESULTS,
        search_depth="advanced",
        include_raw_content=False
    )
    
    _TAVILY_CACHE[clean_task] = result
    return result


# ============================================================
# LLM RETRY
# ============================================================

@retry(
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=RETRY_MULTIPLIER,
        min=RETRY_WAIT_MIN,
        max=RETRY_WAIT_MAX
    )
)
def invoke_llm_with_retry(prompt):
    """
    Invoke the LLM with automatic retries.
    """

    return llm.invoke(prompt)


# ============================================================
# SEARCH ONE TASK
# ============================================================

async def _process_task(
    task: str,
    executor: concurrent.futures.Executor
) -> dict:
    """
    Execute Tavily search for one research task.

    Important:
    This function ONLY performs retrieval.

    It does NOT polish or verify the results.

    Polishing and verification happen after ALL
    research tasks have completed.
    """

    loop = asyncio.get_event_loop()

    def _do_search():
        try:

            # ------------------------------------------------
            # 1. Search Tavily
            # ------------------------------------------------

            search_response = search_with_retry(
                tavily,
                task
            )

            raw_items = []
            task_sources = []

            # ------------------------------------------------
            # 2. Collect raw results
            # ------------------------------------------------

            for result in search_response.get(
                "results",
                []
            ):

                title = result.get(
                    "title",
                    "Untitled Source"
                )

                url = result.get(
                    "url",
                    ""
                )

                content = result.get(
                    "content",
                    ""
                )

                raw_items.append(
                    f"Source: [{title}]({url})\n"
                    f"URL: {url}\n"
                    f"Content: {content}"
                )
                if url:
                    task_sources.append({"title": title, "url": url})

            # ------------------------------------------------
            # 3. Handle no results
            # ------------------------------------------------

            if not raw_items:

                logger.info(
                    f"No results found for: {task}"
                )

                return {"findings": "", "sources": []}

            logger.info(
                f"Completed search for: {task}"
            )

            # ------------------------------------------------
            # 4. Return raw research
            # ------------------------------------------------

            return {
                "findings": f"Research Findings for '{task}':\n" + "\n\n".join(raw_items),
                "sources": task_sources
            }

        except Exception as e:

            logger.warning(
                f"Failed on task '{task}' "
                f"after retries: {e}"
            )

            return {"findings": "", "sources": []}

    try:

        # ----------------------------------------------------
        # Run blocking Tavily call inside executor
        # ----------------------------------------------------

        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                _do_search
            ),
            timeout=AGENT_TIMEOUT
        )

        return result or {"findings": "", "sources": []}

    except asyncio.TimeoutError:

        logger.error(
            f"Task '{task}' timed out "
            f"after {AGENT_TIMEOUT}s — skipping"
        )

        return {"findings": "", "sources": []}


# ============================================================
# RESEARCHER NODE
# ============================================================

async def researcher_node(
    state: ResearchState
) -> ResearchState:
    """
    Main Researcher node.

    Flow:

        Planner
           ↓
        Multiple tasks
           ↓
        Parallel Tavily searches
           ↓
        Gather ALL raw research
           ↓
        LLM Pass 1: Polish
           ↓
        LLM Pass 2: Verify
           ↓
        Verified research
    """

    plan = state["plan"]

    all_results = []
    all_sources = []

    logger.info(
        f"Starting parallel research "
        f"for {len(plan)} sub-tasks"
    )

    # ========================================================
    # STEP 1: PARALLEL RESEARCH
    # ========================================================

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=10
    ) as executor:

        async_tasks = [
            _process_task(
                task,
                executor
            )
            for task in plan
        ]

        results = await asyncio.gather(
            *async_tasks,
            return_exceptions=True
        )

        # ----------------------------------------------------
        # Collect results from every task
        # ----------------------------------------------------

        for result in results:

            if isinstance(result, Exception):

                logger.error(
                    f"Task failed with exception: "
                    f"{result}"
                )

            elif result and result.get("findings"):
                all_results.append(result["findings"])
                all_sources.extend(result.get("sources", []))

    state["sources"] = all_sources

    logger.info(
        f"Raw research complete — "
        f"{len(all_results)} task results gathered"
    )

    # ========================================================
    # STEP 2: COMBINE ALL RESEARCH
    # ========================================================

    raw_research = "\n\n".join(
        all_results
    )
    
    # Truncate to prevent APIStatusError (context length exceeded) on smaller models
    max_chars = 20000
    if len(raw_research) > max_chars:
        logger.warning(f"Truncating raw research from {len(raw_research)} to {max_chars} characters to fit context window.")
        raw_research = raw_research[:max_chars] + "\n...[TRUNCATED]"

    # If every Tavily search failed
    if not raw_research:

        logger.warning(
            "No research results available "
            "for polishing."
        )

        state["research_results"] = []

        return state

    # ========================================================
    # STEP 3: PASS ONE - POLISH
    # ========================================================

    polish_prompt = f"""
You are a research synthesis assistant.

The following are raw research findings collected
from multiple research tasks.

RAW RESEARCH
============

{raw_research}

TASK
====

Create ONE polished and cohesive research summary
that combines the relevant findings from all research
tasks.

REQUIREMENTS
============

1. Preserve important factual information.
2. Do not invent information.
3. Do not use your own outside knowledge.
4. Remove obvious duplication.
5. Organize related findings together.
6. Keep the research relevant to the original tasks.
7. Preserve source URLs for claims whenever available.
8. Do not remove important source information.
9. Clearly distinguish different findings when sources
   disagree.
10. The output must remain grounded in the RAW RESEARCH.

Return ONLY the polished research.
"""

    try:

        logger.info(
            "Starting research polishing..."
        )

        polished_result = invoke_llm_with_retry(
            polish_prompt
        ).content

        logger.info(
            f"Research polishing completed. Length: {len(polished_result)}"
        )

    except Exception as e:

        logger.error(
            f"Research polishing failed: {e}"
        )

        # If polishing fails, keep the raw research
        polished_result = raw_research

    # ========================================================
    # STEP 4: PASS TWO - VERIFICATION
    # ========================================================

    verify_prompt = f"""
You are a research verification assistant.

Your job is to verify a polished research summary
against the original raw research.

RAW RESEARCH
============

{raw_research}

POLISHED RESEARCH
=================

{polished_result}

VERIFICATION RULES
==================

1. Treat RAW RESEARCH as the only source of truth.
2. Remove claims that are not supported by RAW RESEARCH.
3. IMPORTANT: Be lenient, not strict. If a claim is reasonably 
   supported by the RAW RESEARCH — even if worded differently, 
   summarized, or combined from multiple sentences — KEEP it. 
   Only remove claims that are clearly invented, contradicted by 
   the RAW RESEARCH, or completely absent from it. Do not delete 
   large sections just because the exact wording doesn't match. 
   Your default should be to preserve content, not delete it.
4. Do not introduce new information.
5. Do not use your own knowledge.
6. Preserve claims that are directly supported.
7. Preserve source URLs where available.
8. Remove unsupported numbers, dates, statistics,
   names, or factual claims.
9. If the polished research combines multiple sources,
   make sure the combination is actually supported.
10. Do not strengthen claims beyond what the sources say.
11. Your output should normally be close in length to the polished 
    research. A large reduction in length is a sign you are being 
    too strict — reconsider before removing large sections.
12. Output ONLY the final verified research.

FINAL OUTPUT
============

Return the verified research only.
"""

    try:

        logger.info(
            "Starting research verification..."
        )

        verified_result = invoke_llm_with_retry(
            verify_prompt
        ).content
        
        if not verified_result.strip():
            logger.warning("Verification returned an empty result. Falling back to the polished result.")
            verified_result = polished_result

        logger.info(
            f"Research verification completed. Length: {len(verified_result)}"
        )

    except Exception as e:

        logger.error(
            f"Research verification failed: {e}"
        )

        # If verification fails, keep polished result
        verified_result = polished_result

    # ========================================================
    # STEP 5: SAVE FINAL RESEARCH
    # ========================================================

    state["research_results"] = [
        verified_result
    ]

    # ========================================================
    # METRICS
    # ========================================================

    input_tokens = sum(
        len(p)
        for p in plan
    )

    output_tokens = sum(
        len(r)
        for r in state["research_results"]
    )

    metrics.end_agent(
        "researcher",
        input_tokens=input_tokens,
        output_tokens=output_tokens
    )

    logger.info(
        "Researcher node completed successfully."
    )

    return state


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_state = {

        "plan": [

            "Identify the primary causes of climate "
            "change affecting coral reefs",

            "Research coral bleaching effects"

        ]

    }

    result = asyncio.run(
        researcher_node(test_state)
    )

    print("\n")
    print("=" * 70)
    print("FINAL VERIFIED RESEARCH")
    print("=" * 70)
    print("\n")

    for r in result["research_results"]:

        try:
            print(r)
        except UnicodeEncodeError:
            print(r.encode("utf-8", "replace").decode("utf-8"))

        print("\n")
        print("-" * 70)
