# Evaluation Results

*Based on the most recent run of `eval.py` in the workspace.*

| Query | Status | Time (s) | Iterations | Loop Required? | Confidence |
|---|---|---|---|---|---|
| AWS vs GCP for startups in 2025... | Error (Qdrant missing `user_id` index) | 35.11 | 0 | No | 0.00 |
| Latest advancements in quantum computing | Success | 174.24 | 3 | Yes | 0.00 |
| How does transformer attention mechanism work? | Error (Qdrant missing `user_id` index) | 22.41 | 0 | No | 0.00 |
| Best practices for building production RAG... | Error (Qdrant missing `user_id` index) | 77.95 | 0 | No | 0.00 |
| State of AI regulation globally in 2025 | Success | 152.99 | 3 | Yes | 0.00 |

## Summary Statistics
- **Success Rate:** 2/5 (40%)
- **Average Time:** 163.61s (on successful runs)
- **Average Confidence:** 0.00 (Due to missing retrieval data/errors during evaluation)
- **Loop Required:** 100.0% (All successful runs hit the 3-iteration max limit)

**Note/TODO:** The current Qdrant cloud collection is missing the required keyword index for `user_id`. This causes 400 Bad Request errors in the Retriever agent on fresh queries.
