import os
from dotenv import load_dotenv

load_dotenv()

# LLM settings
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Graph / loop settings
CONFIDENCE_THRESHOLD = 0.8
MAX_ITERATIONS = 3

# Retry settings
RETRY_ATTEMPTS = 3
RETRY_WAIT_MIN = 2
RETRY_WAIT_MAX = 10

# Search settings
TAVILY_MAX_RESULTS = 5
