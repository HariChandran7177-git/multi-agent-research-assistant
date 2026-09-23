import asyncio
from langchain_groq import ChatGroq
from core.config import GROQ_REPORTER_MODEL, GROQ_API_KEY, AGENT_TIMEOUT
from core.logger import get_logger
from core.metrics import metrics

logger = get_logger(__name__)

llm = ChatGroq(model=GROQ_REPORTER_MODEL, api_key=GROQ_API_KEY, temperature=0.1)


async def answer_doubt(report_text: str, question: str) -> str:
    """Async doubt answering with timeout and metrics."""
    loop = asyncio.get_event_loop()
    prompt = f"""You are a helpful assistant. You are given a research report and a question.
Your task is to answer the question.
First, check if the answer is present in the provided report text.
If the answer is in the report, start your response with "According to the report..." or a similar preface, and then provide the answer based on the report.
If the report does NOT cover the question, state that the report doesn't cover this, BUT then go ahead and answer the question using your general knowledge.

Report:
{report_text[:15000]}  # Truncate to stay under token limits

Question: {question}
"""
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
        return "Sorry, I timed out while analyzing your question. Please try again."
    except Exception as e:
        logger.error(f"Error in answer_doubt: {e}")
        metrics.end_agent("doubt", error=str(e))
        return f"Error analyzing doubt: {e}"
