from langchain_groq import ChatGroq
from core.state import ResearchState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.4,
)

REPORTER_PROMPT = """You are a research report writer.

Original query: {query}

Research plan (sub-tasks investigated):
{plan}

Research findings:
{research_results}

Most relevant retrieved documents:
{retrieved_docs}

Write a clear, well-structured final report answering the original query.
Structure it as:
1. Introduction (1-2 sentences framing the question)
2. Key Findings (organized by sub-topic, using the research above)
3. Conclusion (1-2 sentence summary)

Write in plain, professional prose. Do not mention "sub-tasks" or reference
the internal process — just present the findings naturally.
"""


def reporter_node(state: ResearchState) -> ResearchState:
    # Truncate inputs to stay within Groq free-tier token limits
    plan_text = "\n".join(state.get("plan", []))
    research_text = "\n".join(state.get("research_results", []))[:4000]
    docs_text = "\n".join(state.get("retrieved_docs", []))[:3000]

    prompt = REPORTER_PROMPT.format(
        query=state["query"],
        plan=plan_text,
        research_results=research_text,
        retrieved_docs=docs_text,
    )

    response = llm.invoke(prompt)
    state["final_report"] = response.content.strip()

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
    result = reporter_node(test_state)
    print(result["final_report"])