import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_groq import ChatGroq

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))


def researcher_node(state):
    plan = state["plan"]
    all_results = []

    for task in plan:
        search_response = tavily.search(query=task, max_results=3)
        for result in search_response["results"]:
            all_results.append(f"{result['title']}: {result['content']}")

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
