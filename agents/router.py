import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from core.state import ResearchState
from core.logger import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import GROQ_MODEL, GROQ_API_KEY, RETRY_ATTEMPTS, RETRY_WAIT_MIN, RETRY_WAIT_MAX

load_dotenv()
logger = get_logger(__name__)

# Use a fast, small model for routing — no need for the heavy 70B model here
llm = ChatGroq(
    model="groq/compound-mini",
    api_key=GROQ_API_KEY,
    temperature=0.2,
)

ROUTER_PROMPT = """You are an intelligent query router. Your job is to determine if a user's query requires deep web research, or if it is a simple, casual conversational question (like "hello", "tell me a joke", "how are you"). You must also detect the requested tone of the response (e.g. "professional", "funny", "academic", "explain like I'm 5"). If no specific tone is implied, default to "super friendly and conversational".

Original query: "{query}"

If the query requires deep web research, return a JSON object with:
{{"is_casual": false, "response": "", "tone": "detected tone here"}}

If the query is a simple/casual question that you can answer immediately without searching the web, return a JSON object with a friendly, conversational response:
{{"is_casual": true, "response": "Your friendly answer here.", "tone": "detected tone here"}}

OUTPUT ONLY VALID JSON. Do not wrap it in markdown block.
"""

# Tighter retry settings for the router — fast fail, no long waits
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
def invoke_with_retry(llm, prompt):
    return llm.invoke(prompt)

def router_node(state: ResearchState) -> ResearchState:
    logger.info("Routing query to determine path and tone")
    prompt = ROUTER_PROMPT.format(query=state["query"])
    
    try:
        response = invoke_with_retry(llm, prompt)
        content = response.content.strip()
        # Clean up any potential markdown formatting from LLM
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        parsed = json.loads(content)
        state["is_casual"] = parsed.get("is_casual", False)
        state["tone"] = parsed.get("tone", "super friendly and conversational")
        logger.info(f"Detected tone: {state['tone']}")
        
        if state["is_casual"]:
            logger.info("Query classified as casual. Bypassing research.")
            state["final_report"] = parsed.get("response", "I'm here to chat! What's on your mind?")
        else:
            logger.info("Query requires research. Routing to Planner.")
            
    except Exception as e:
        logger.error(f"Router failed to parse LLM response: {e}. Defaulting to research path and friendly tone.")
        state["is_casual"] = False
        state["tone"] = "super friendly and conversational"
        
    return state

if __name__ == "__main__":
    # Test Casual
    state1: ResearchState = {"query": "Hello there! How are you?", "is_casual": False, "plan": [], "research_results": [], "retrieved_docs": [], "critique": "", "confidence_score": 0.0, "iteration_count": 0, "final_report": ""}
    res1 = router_node(state1)
    print(f"Casual Test: {res1.get('is_casual')} -> {res1.get('final_report')}")

    # Test Research
    state2: ResearchState = {"query": "What is the best AI agent for web scraping?", "is_casual": False, "plan": [], "research_results": [], "retrieved_docs": [], "critique": "", "confidence_score": 0.0, "iteration_count": 0, "final_report": ""}
    res2 = router_node(state2)
    print(f"Research Test: {res2.get('is_casual')} -> {res2.get('final_report')}")
