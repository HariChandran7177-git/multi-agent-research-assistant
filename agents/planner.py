import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load .env from the project root (one level up from agents/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))


def planner_node(state):
    query = state["query"]
    prompt = f"""Break down this research query into 3-5 clear sub-tasks.
Return ONLY a numbered list, nothing else.

Query: {query}"""

    response = llm.invoke(prompt)
    lines = [line.strip() for line in response.content.split("\n") if line.strip()]

    state["plan"] = lines
    state["iteration_count"] = 0
    return state


