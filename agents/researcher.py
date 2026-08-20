import os
import concurrent.futures
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from core.logger import get_logger

load_dotenv()

from core.config import GROQ_MODEL, GROQ_API_KEY, RETRY_ATTEMPTS, RETRY_WAIT_MIN, RETRY_WAIT_MAX, TAVILY_MAX_RESULTS

logger = get_logger(__name__)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)


@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX))
def search_with_retry(client, task):
    # Ensure clean query without prefixes
    clean_task = task.strip().lstrip("0123456789.-*#\t ")
    return client.search(query=clean_task, max_results=TAVILY_MAX_RESULTS, search_depth="advanced", include_raw_content=False)


def _process_task(task):
    """Helper function to run a single task and return formatted results."""
    local_results = []
    try:
        search_response = search_with_retry(tavily, task)
        for result in search_response.get("results", []):
            title = result.get("title", "Untitled Source")
            url = result.get("url", "")
            content = result.get("content", "")
            formatted_item = f"Source: [{title}]({url})\nURL: {url}\nContent: {content}"
            local_results.append(formatted_item)
        logger.info(f"Completed search for: {task}")
    except Exception as e:
        logger.warning(f"Failed on task '{task}' after retries: {e}")
    return local_results


def researcher_node(state):
    plan = state["plan"]
    all_results = []

    logger.info(f"Starting parallel research for {len(plan)} sub-tasks")

    # Use as_completed so each thread has its own try/except — one failure won't
    # silently kill the batch or return None (executor.map() would do that)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_task = {executor.submit(_process_task, task): task for task in plan}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result(timeout=30)
                all_results.extend(result)
            except concurrent.futures.TimeoutError:
                logger.error(f"Task '{task}' timed out after 30s — skipping")
            except Exception as e:
                logger.error(f"Task '{task}' failed: {e} — skipping")

    logger.info(f"Research complete — {len(all_results)} total results gathered")
    state["research_results"] = all_results
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
