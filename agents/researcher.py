import os
import concurrent.futures
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential
from core.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(model="llama-3.3-70b-versatile",
               api_key=os.getenv("GROQ_API_KEY"))


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def search_with_retry(client, task):
    # Ensure clean query without prefixes
    clean_task = task.strip().lstrip("0123456789.-*#\t ")
    return client.search(query=clean_task, max_results=4, search_depth="advanced", include_raw_content=False)


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

    # Use ThreadPoolExecutor to run tasks concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Map tasks to executor
        results_nested = list(executor.map(_process_task, plan))
        
    # Flatten the results
    for res_list in results_nested:
        all_results.extend(res_list)

    logger.info(
        f"Research complete — {len(all_results)} total results gathered")
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
