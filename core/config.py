import os
from dotenv import load_dotenv

load_dotenv()

# LLM settings
GEMINI_PLANNER_MODEL = "gemini-2.5-flash"
GEMINI_ROUTER_MODEL = "gemini-flash-lite-latest"
GROQ_RESEARCHER_MODEL = "openai/gpt-oss-20b"
GROQ_REPORTER_MODEL = "openai/gpt-oss-120b"
GROQ_CRITIC_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Graph / loop settings
CONFIDENCE_THRESHOLD = 0.7
MAX_ITERATIONS = 3

# Retry settings — longer waits to handle rate limits
RETRY_ATTEMPTS = 3
RETRY_MULTIPLIER = 2
RETRY_WAIT_MIN = 2
RETRY_WAIT_MAX = 10

# Search settings
TAVILY_MAX_RESULTS = 5

# Timeout settings (seconds)
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))

# Qdrant settings
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
