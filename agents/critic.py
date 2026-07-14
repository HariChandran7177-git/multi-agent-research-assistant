import json
from langchain_groq import ChatGroq
from core.state import ResearchState
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
)

CRITIC_PROMPT = """You are a research quality critic.

Original query: {query}

Research findings:
{research_results}

Retrieved documents:
{retrieved_docs}

Evaluate whether this research sufficiently answers the original query.
Respond ONLY with valid JSON in this exact format, no other text:
{{
  "score": <float between 0 and 1>,
  "critique": "<1-2 sentence explanation of gaps or confirmation of quality>"
}}
"""


def critic_node(state: ResearchState) -> ResearchState:
    # Truncate inputs to stay within Groq free-tier token limits
    research_text = "\n".join(state.get("research_results", []))[:4000]
    docs_text = "\n".join(state.get("retrieved_docs", []))[:3000]

    prompt = CRITIC_PROMPT.format(
        query=state["query"],
        research_results=research_text,
        retrieved_docs=docs_text,
    )

    response = llm.invoke(prompt)
    content = response.content.strip()

    # Strip accidental markdown fences if the model adds them
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    try:
        parsed = json.loads(content)
        score = float(parsed.get("score", 0.5))
        critique = parsed.get("critique", "No critique provided.")
    except (json.JSONDecodeError, ValueError):
        # Fallback if the LLM doesn't return clean JSON
        score = 0.5
        critique = f"Could not parse critic output: {content[:200]}"

    state["confidence_score"] = score
    state["critique"] = critique
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    return state


if __name__ == "__main__":
    test_state: ResearchState = {
        "query": "What are the latest advancements in quantum computing?",
        "plan": ["Search recent breakthroughs", "Find key companies", "Identify challenges"],
        "research_results": ["IBM announced a 1000-qubit chip in 2023.", "Quantum error correction remains a major hurdle."],
        "retrieved_docs": ["IBM's Condor processor has 1,121 qubits."],
        "critique": "",
        "confidence_score": 0.0,
        "iteration_count": 0,
        "final_report": "",
    }
    result = critic_node(test_state)
    print("Score:", result["confidence_score"])
    print("Critique:", result["critique"])
    print("Iteration:", result["iteration_count"])
