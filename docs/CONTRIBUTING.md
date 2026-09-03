# Contributing

## Setup
1. Clone the repository and set up a Python 3.11+ environment.
2. Install dependencies: `pip install -r requirements.txt` (or use `pip install -e .[dev]` for dev tools).
3. Copy `.env.example` to `.env` and fill in the required keys:
   - `GROQ_API_KEY`: Groq for LLM routing and reasoning
   - `TAVILY_API_KEY`: Tavily for web search
   - `GOOGLE_API_KEY`: Gemini for vector embeddings
   - `QDRANT_URL` / `QDRANT_API_KEY`: Qdrant Cloud for semantic retrieval

## Running Tests
We use `pytest` for testing. All external APIs are mocked, so no live keys are needed to pass the suite.
```bash
pytest tests/ -v
```

## Code Style & Linting
This project enforces code formatting via `ruff` and `black`. Configuration is located in `pyproject.toml`.

- **Line Length:** 88 characters
- **Linter Checks:** `ruff` (selecting E and F rules)

Before committing, please ensure your code conforms to these standards:
```bash
black .
ruff check .
```
