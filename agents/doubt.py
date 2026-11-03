import asyncio
from langchain_groq import ChatGroq
from core.config import GROQ_RESEARCHER_MODEL, GROQ_API_KEY, AGENT_TIMEOUT
from core.logger import get_logger
from core.metrics import metrics

logger = get_logger(__name__)

llm = ChatGroq(model=GROQ_RESEARCHER_MODEL, api_key=GROQ_API_KEY, temperature=0.1)


async def answer_doubt(report_text: str, question: str) -> str:
    """Async doubt answering that provides concise answers ('small matter') and prompts for next step."""
    loop = asyncio.get_running_loop()
    prompt = f"""You are a concise, sharp research assistant answering a user's question about a report.

CRITICAL INSTRUCTIONS:
1. KEEP IT SHORT ("SMALL MATTER"): Give a maximum of 2-3 concise bullet points or 2 short sentences. Do NOT write long paragraphs or essays, even if the question is complex.
2. Directly answer the question based on the report text. If the report doesn't cover it, state that in one sentence and give a brief fact.
3. ALWAYS end your response with a short question asking the user for their next step (e.g. "What would you like to explore next?", "Would you like me to dig deeper into [related topic] or export this summary?").

Report Context:
{report_text[:12000]}

Question: {question}

Answer (concise + ask for next step):"""
    try:
        # Timeout protection
        response = await asyncio.wait_for(
            loop.run_in_executor(None, llm.invoke, prompt),
            timeout=AGENT_TIMEOUT
        )
        answer = response.content.strip()
        metrics.end_agent("doubt", input_tokens=len(report_text) + len(question), output_tokens=len(answer))
        return answer
    except asyncio.TimeoutError:
        logger.error(f"Doubt answering timed out after {AGENT_TIMEOUT}s")
        metrics.end_agent("doubt", error="timeout")
        return "Sorry, I timed out while analyzing your question. Would you like me to try again or ask a different question?"
    except Exception as e:
        logger.error(f"Error in answer_doubt: {e}")
        metrics.end_agent("doubt", error=str(e))
        return f"Error analyzing doubt: {e}. Would you like to try asking in a different way?"
