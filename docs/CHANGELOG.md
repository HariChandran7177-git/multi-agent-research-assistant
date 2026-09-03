# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-08-31
### Added
- Initial documented version reflecting the true state of the repository.
- LangGraph orchestration with persistent `AsyncSqliteSaver` checkpoints.
- Human-in-the-Loop (HitL) pause implementation before the Reporter node.
- Smart routing (`agents/router.py`) to bypass research for casual queries and detect tone.
- Critic hybrid scoring logic: 93% objective signals (Qdrant cosine scores, coverage, depth) and 7% LLM judgment.
- FastAPI server with Server-Sent Events (SSE) streaming and vanilla HTML/JS frontend.
- SQLite-based caching and report history persistence.
