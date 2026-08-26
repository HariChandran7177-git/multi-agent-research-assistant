import os
import asyncio
import concurrent.futures
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from core.state import ResearchState
from core.logger import get_logger
from core.metrics import metrics
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

from core.config import GROQ_MODEL, GROQ_API_KEY, RETRY_ATTEMPTS, RETRY_MULTIPLIER, RETRY_WAIT_MIN, RETRY_WAIT_MAX, TAVILY_MAX_RESULTS, AGENT_TIMEOUT

logger = get_logger(__name__)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)


@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=RETRY_MULTIPLIER, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX))
def search_with_retry(client, task):
    # Ensure clean query without prefixes
    clean_task = task.strip().lstrip("0123456789.-*#\t ")
    return client.search(query=clean_task, max_results=TAVILY_MAX_RESULTS, search_depth="advanced", include_raw_content=False)


async def _process_task(task: str, executor: concurrent.futures.Executor) -> list:
    """Async wrapper for _process_task to enable timeout protection."""
    loop = asyncio.get_event_loop()
    local_results = []

    def _do_search():
        try:
            search_response = search_with_retry(tavily, task)
            for result in search_response.get("results", []):
                title = result.get("title", "Untitled Source")
                url = result.get("url", "")
                content = result.get("content", "")
                formatted_item = f"Source: [{title}]({url})\nURL: {url}\nContent: {content}"
                local_results.append(formatted_item)
            logger.info(f"Completed search for: {task}")
            return local_results
        except Exception as e:
            logger.warning(f"Failed on task '{task}' after retries: {e}")
            return []

    try:
        # Run with timeout protection
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, _do_search),
            timeout=AGENT_TIMEOUT
        )
        return result or []
    except asyncio.TimeoutError:
        logger.error(f"Task '{task}' timed out after {AGENT_TIMEOUT}s — skipping")
        return []


async def researcher_node(state: ResearchState) -> ResearchState:
    """Async researcher node with timeout and metrics."""
    loop = asyncio.get_event_loop()
    plan = state["plan"]
    all_results = []
    start_time = metrics.current_metrics.get("researcher", None)

    logger.info(f"Starting parallel research for {len(plan)} sub-tasks")

    # Use as_completed so each thread has its own try/except
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Create async tasks with timeout protection
        async_tasks = [
            _process_task(task, executor) for task in plan
        ]
        # Run all tasks concurrently and gather results
        results = await asyncio.gather(*async_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
            elif result:
                all_results.extend(result)

    logger.info(f"Research complete — {len(all_results)} total results gathered")
    state["research_results"] = all_results

    # Record metrics
    input_tokens = sum(len(p) for p in plan)
    output_tokens = sum(len(r) for r in all_results)
    metrics.end_agent("researcher", input_tokens=input_tokens, output_tokens=output_tokens)

    return state


if __name__ == "__main__":
    test_state = {
        "plan": [
            "Identify the primary causes of climate change affecting coral reefs",
            "Research coral bleaching effects"
        ]
    }
    result = researcher_node(test_state)
    for r in result["research_results"]:
        print(r[:150])
        print("---")
