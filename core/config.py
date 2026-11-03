import os
import warnings

# Stabilize OpenBLAS, MKL, and OpenMP thread pools on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Suppress Google automatic function calling deprecation warning
warnings.filterwarnings("ignore", message=".*automatic function calling.*")
warnings.filterwarnings("ignore", message=".*generate_content.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.*")

from dotenv import load_dotenv

load_dotenv()

# LLM settings
GEMINI_PLANNER_MODEL = "gemini-2.5-flash"
GEMINI_ROUTER_MODEL = "gemini-2.5-flash"
GROQ_RESEARCHER_MODEL = "openai/gpt-oss-20b"
GROQ_REPORTER_MODEL = "openai/gpt-oss-120b"
GROQ_CRITIC_MODEL = "openai/gpt-oss-20b"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1500"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))

# Graph / loop settings
# Threshold lowered from 0.70 → 0.65: objective score typically plateaus at 0.72+
# after iteration 1; extra loops waste 20s without improving quality.
CONFIDENCE_THRESHOLD = 0.65
MAX_ITERATIONS = 2

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

# ── Data directory (SQLite databases, logs) ────────────────────────────────────
DATA_DIR = os.path.abspath(os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")))
os.makedirs(DATA_DIR, exist_ok=True)

# ── Model cache directory (SentenceTransformer weights) ────────────────────────
MODEL_CACHE_DIR = os.path.abspath(os.getenv("MODEL_CACHE_DIR", os.path.join(os.path.dirname(__file__), "..", "model_cache")))
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# ── API-key authentication ─────────────────────────────────────────────────────
APP_API_KEY = os.getenv("APP_API_KEY")

# ── CORS origins ───────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# Logging level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
