from langchain_groq import ChatGroq
from core.config import GROQ_REPORTER_MODEL, GROQ_API_KEY
from core.logger import get_logger

logger = get_logger(__name__)

llm = ChatGroq(model=GROQ_REPORTER_MODEL, api_key=GROQ_API_KEY, temperature=0.1)

def answer_doubt(report_text: str, question: str) -> str:
    prompt = f"""You are a helpful assistant. You are given a research report and a question about it.
Your task is to answer the question ONLY based on the provided report text.
Keep your answer concise (2-3 sentences max).
If the report does not cover the question, state plainly: "The report does not cover this."

Report:
{report_text}

Question: {question}
"""
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error in answer_doubt: {e}")
        return f"Error analyzing doubt: {e}"
