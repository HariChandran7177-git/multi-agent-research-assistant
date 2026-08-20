import os
from dotenv import load_dotenv

load_dotenv()

# LLM settings — groq/compound for orchestration, gpt-oss-20b for reporter (avoids compound rate limits)
GROQ_MODEL = "groq/compound"
GROQ_REPORTER_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Graph / loop settings
CONFIDENCE_THRESHOLD = 0.8
MAX_ITERATIONS = 3

# Retry settings — longer waits to handle rate limits
RETRY_ATTEMPTS = 3
RETRY_MULTIPLIER = 2
RETRY_WAIT_MIN = 4
RETRY_WAIT_MAX = 30

# Search settings
TAVILY_MAX_RESULTS = 5
