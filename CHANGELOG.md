# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Streamlit web UI for interactive querying
- FastAPI REST endpoint (`/research`)
- LangSmith tracing integration
- Support for OpenAI models as an alternative LLM backend

---

## [0.3.0] - 2025-08-18

### Added
- **Router Agent** — Smart gatekeeper that classifies queries as casual vs. research, detects tone, and short-circuits the pipeline for simple conversational queries.
- **Tone detection** — The Router now extracts the user's desired response style (e.g., `"academic"`, `"explain like I'm 5"`, `"funny"`) and passes it to the Reporter for a fully customized output.
- `is_casual` and `tone` fields added to `ResearchState`.
- Conditional edge `route_after_router` in the LangGraph graph.

### Changed
- Router uses `llama-3.1-8b-instant` (fast, low-cost) instead of the main 70B model for maximum speed.
- Tightened retry settings for the Router (2 attempts, 1–3s wait) vs. the standard policy.

---

## [0.2.0] - 2025-08-10

### Added
- **Hybrid Confidence Scoring** in the Critic (`core/scorer.py`):
  - 60% LLM subjective judgment
  - 40% objective signals: Qdrant cosine similarity, source count, result length
- `score_breakdown` and `qdrant_scores` fields added to `ResearchState` for full transparency.
- `calculate_objective_score()` and `calculate_hybrid_score()` utility functions.

### Changed
- Critic no longer relies solely on LLM self-reported scores — objective signals prevent score inflation.
- Confidence threshold tuned to `0.72` for better precision/recall balance.

---

## [0.1.0] - 2025-08-01

### Added
- Initial 5-agent LangGraph pipeline: Planner → Researcher → Retriever → Critic → Reporter.
- **Planner**: Classifies query intent into 10 task types and breaks it into 4–5 focused sub-tasks.
- **Researcher**: Parallel Tavily web search using `ThreadPoolExecutor` (5 workers).
- **Retriever**: Google Gemini embeddings + Qdrant Cloud vector store with session-scoped filtering.
- **Critic**: LLM-based quality evaluator with self-correction loop (max 3 iterations).
- **Reporter**: Dynamic markdown report writer with flexible format selection.
- `tenacity` exponential backoff retry logic on all LLM and API calls.
- Qdrant fallback to in-memory client if remote connection fails.
- CLI entry point via `python main.py "your query"`.
- `.env.example` with all required API key slots.
