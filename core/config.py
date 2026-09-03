import os
from dotenv import load_dotenv

load_dotenv()

# LLM settings
GROQ_MODEL = "openai/gpt-oss-20b"
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

# Redis settings
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"

# Timeout settings (seconds)
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))

# Qdrant settings
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Google API settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
