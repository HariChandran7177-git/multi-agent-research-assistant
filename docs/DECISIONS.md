# Architectural Decisions

This document outlines key technical decisions based strictly on the current implementation of the Multi-Agent Research Assistant.

### 1. Orchestration: LangGraph over hand-rolled loops
* **Decision:** We use LangGraph (`core/graph.py`) to manage the agent pipeline instead of a hand-rolled while-loop.
* **Alternatives considered:** AutoGen, CrewAI, or a custom Python loop. 
* **Reasoning:** Our pipeline requires a cyclic graph (Critic -> Researcher loop) rather than a linear chain. Crucially, LangGraph provides the `AsyncSqliteSaver` checkpointer. This allows us to easily pause execution mid-graph (`interrupt_before=["reporter"]`), serialize the state to disk, and wait for human approval before continuing. Rebuilding a reliable, asynchronous state machine with database checkpointing from scratch would introduce massive complexity.
* **Consequence / Known Limitation:** The LangGraph SQLite checkpointer requires a strict `config={"configurable": {"thread_id": ...}}` to run. If this is omitted, the graph crashes.

### 2. Hybrid Evaluation Score (93% Objective / 7% LLM)
* **Decision:** The Critic agent (`agents/critic.py`) evaluates research quality using a hybrid score implemented in `core/scorer.py` (`calculate_hybrid_score`). The weighting is strictly **93% objective signals** and **7% LLM judgment**.
* **Alternatives considered:** Relying 100% on an LLM to grade the research.
* **Reasoning:** LLMs are easily fooled by fluent-but-thin scraped text. We rely heavily on deterministic signals:
    * `retrieval_relevance` (35%): Qdrant cosine similarity scores (the most reliable signal of query-document match).
    * `plan_coverage` (20%): Verifies if results actually contain keywords from the Planner's sub-tasks.
    * `content_depth` (20%): Penalizes results under 150 characters (0.2 score); rewards over 500 characters (1.0 score).
    * `source_quality` (10%): Regex URL parser that rewards `.edu`/`.gov` and known publications (e.g. `nature.com`).
    * `duplicate_penalty` (10%): 80-character fingerprinting to penalize redundant results.
    * `diversity` (5%): Compares the first 50 characters of results to catch scraping loops.
* **Consequence:** The LLM acts only as a minor 7% semantic tie-breaker. If the underlying data is structurally poor (missing URLs, too short), the Critic will force a loop regardless of how eloquently the LLM praises the text.

### 3. Loop Thresholds & Forced Exits (`CONFIDENCE_THRESHOLD = 0.8`, `MAX_ITERATIONS = 3`)
* **Decision:** Set in `core/config.py`. The Critic loop terminates if the hybrid score hits `0.8` OR if `iteration_count >= 3`.
* **Reasoning:** We need a stringent bar (0.8) to ensure high-quality research, but we must cap loops to prevent infinite API spend and endless latency. 
* **Consequence / Mitigation:** In `core/graph.py`, the `should_loop` function routes to `"reporter"` if `iterations >= MAX_ITERATIONS`, *even if the score is terrible*. If the threshold isn't met after 3 passes, the pipeline proceeds to the Reporter with low-confidence/empty research. **This is now mitigated** by two safeguards in `agents/reporter.py`: (1) if no research data exists at all, the Reporter skips the LLM call entirely and returns an honest "limited research available" fallback message; (2) if some data exists but `confidence_score < 0.5`, a strict-grounding appendix is injected into the prompt instructing the LLM to only state facts present in the provided research and flag gaps explicitly.

### 4. Vector Database: Qdrant
* **Decision:** Used in `agents/retriever.py` for embedding storage and similarity search.
* **Alternatives considered:** FAISS (in-memory, no persistent multi-tenant filtering), Pinecone (expensive), pgvector (heavy dependency).
* **Reasoning:** Qdrant supports robust Payload filtering. We create a payload index on `session_id` and `user_id`, allowing us to perfectly isolate data between concurrent users while querying the exact same collection (`research_docs`). It also returns raw cosine distance scores, which feed directly into our objective scorer (35% weight).

### 5. Web Search: Tavily API
* **Decision:** Used by `agents/researcher.py` to execute parallel sub-task queries.
* **Reasoning:** Tavily is optimized for LLMs, stripping out HTML and returning clean markdown/text content natively.
* **Consequence / KNOWN LIMITATION:** There is **no fallback search provider**. If the Tavily API fails (e.g. 429 limit exceeded, timeout), the `_process_task` function catches the exception, logs a warning, and silently returns an empty list `[]`. The Researcher will then pass 0 results to the Retriever, causing the Critic to score it poorly, looping until `MAX_ITERATIONS` is hit. When this happens, the Reporter's grounding safeguard (`_no_research_fallback`) detects the empty research data and returns an honest "limited research available" message instead of hallucinating a report.

### 6. The Router / Planner Split
* **Decision:** We separate initial routing (`agents/router.py`) from task planning (`agents/planner.py`).
* **Reasoning:** `router.py` uses a cheap, fast model (`groq/compound-mini`, hardcoded) to determine if a query is `is_casual` and to detect the user's `tone`. If it's a casual greeting ("hello"), it bypasses the entire pipeline and returns instantly. `planner.py` uses `GROQ_MODEL` from `core/config.py` (currently `llama-3.1-8b-instant`) and is only invoked if real research is required. This drastically reduces latency and API costs for trivial queries.

### 7. Human-in-the-Loop Placement
* **Decision:** The graph is configured with `interrupt_before=["reporter"]` (`core/graph.py`).
* **Reasoning:** We pause execution exactly after the Critic is satisfied (or max iterations are reached), but *before* the Reporter generates the final output. This is the optimal checkpoint: the user can review the raw `research_results` and the Critic's `score_breakdown` via the API, and decide whether to approve it or abandon it *before* we spend tokens on the heavy report generation prompt.
