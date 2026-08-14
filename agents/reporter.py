from langchain_groq import ChatGroq
from core.state import ResearchState
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from core.logger import get_logger

load_dotenv()

from core.config import GROQ_MODEL, GROQ_API_KEY, RETRY_ATTEMPTS, RETRY_WAIT_MIN, RETRY_WAIT_MAX

logger = get_logger(__name__)

llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.4,
)

REPORTER_PROMPT = """You are a highly knowledgeable research assistant. 
Your instructed tone for this report is: "{tone}". 
Ensure that you strictly follow this tone (e.g., if it says "super friendly", be very casual and use jokes. If it says "professional", be formal).
If you ever need to use a complex, technical, or "hard" word (jargon) that doesn't fit the tone, pause to explain it simply before moving on.

Original query: {query}

Research plan (sub-tasks investigated):
{plan}

Research findings:
{research_results}

Most relevant retrieved documents:
{retrieved_docs}

Write a report answering the query. The FORMAT must be chosen based purely on 
what the question actually needs — don't force a fixed template. Pick whichever 
of the following elements genuinely fit, and skip the ones that don't:

- **Comparison table** — use when comparing 2+ things
- **Hierarchy / nested bullets** — use when explaining categories
- **Flowchart** — use when explaining a process. Write it as a Mermaid diagram in a ```mermaid code block
- **Timeline** — use for anything involving dates or history
- **Bulleted key facts** — use for scannable lists of findings
- **Plain narrative paragraphs** — use for explanations

Additional rules:
- Do NOT use generic labels like "Introduction," "Key Findings," or "Conclusion." Use headers tied to the content.
- Open directly by addressing the user's question.
- Use bold for specific names, numbers, and key facts so they stand out.
- End with a short "Bottom line" that wraps up the report.
- Never mention "sub-tasks," "the plan," or any internal process — just share what you learned.
- Use specific facts, names, and figures from the research above.
- **Sources & References**: At the very end of your report, include a section titled "Sources & References". Extract the URLs from the research findings and present them as a bulleted list of clickable Markdown links: `- [Source Title](URL)`. Only use URLs actually present in the provided research data.

Write in the specified tone: {tone}
"""


@retry(stop=stop_after_attempt(RETRY_ATTEMPTS), wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX))
def invoke_with_retry(llm, prompt):
    return llm.invoke(prompt)


def reporter_node(state: ResearchState) -> ResearchState:
    logger.info("Writing final report")

    plan_text = "\n".join(state.get("plan", []))
    research_text = "\n".join(state.get("research_results", []))[:8000]
    docs_text = "\n".join(state.get("retrieved_docs", []))[:8000]
    tone = state.get("tone", "super friendly and conversational")

    prompt = REPORTER_PROMPT.format(
        tone=tone,
        query=state["query"],
        plan=plan_text,
        research_results=research_text,
        retrieved_docs=docs_text,
    )

    try:
        response = invoke_with_retry(llm, prompt)
        state["final_report"] = response.content.strip()
        logger.info("Report generated successfully")
    except Exception as e:
        logger.error(f"LLM call failed after retries: {e}")
        state["final_report"] = (
            "We were unable to generate a final report due to a technical issue "
            f"with the report-writing service. Error: {e}"
        )

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
