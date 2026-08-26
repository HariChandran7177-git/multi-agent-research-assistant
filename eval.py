"""
eval.py
-------
Mini evaluation harness for the NeuralDesk multi-agent research pipeline.
Runs 5 predefined complex queries, records metrics, and saves results to eval_results.md.
"""

import time
import asyncio
from core.graph import build_graph
from core.state import ResearchState
from core.logger import get_logger

logger = get_logger(__name__)

TEST_QUERIES = [
    "AWS vs GCP for startups in 2025 explain like a senior engineer",
    "Latest advancements in quantum computing",
    "How does transformer attention mechanism work? Explain simply",
    "Best practices for building production RAG systems in 2025",
    "State of AI regulation globally in 2025"
]

async def run_evaluation():
    logger.info("Building evaluation graph...")
    graph = await build_graph()

    results = []

    for i, query in enumerate(TEST_QUERIES):
        logger.info(f"\n--- Running evaluation {i+1}/{len(TEST_QUERIES)} ---")
        logger.info(f"Query: {query}")

        initial_state = {
            "query": query,
            "user_id": "eval_user",
            "plan": [],
            "research_results": [],
            "retrieved_docs": [],
            "qdrant_scores": [],
            "retrieval_available": False,
            "critique": "",
            "confidence_score": 0.0,
            "iteration_count": 0,
            "final_report": "",
        }

        start_time = time.time()

        try:
            # Provide thread_id since we added a checkpointer
            config = {"configurable": {"thread_id": f"eval-{i}"}}

            # We use ainvoke to run the full graph locally
            # This will run until it is interrupted before 'reporter'
            state_dict = await graph.ainvoke(initial_state, config=config)

            # Since the graph pauses before 'reporter' (HitL), we need to resume it
            final_state = await graph.ainvoke(None, config=config)

            duration = time.time() - start_time

            is_casual = final_state.get("is_casual", False)
            confidence = final_state.get("confidence_score", 0.0)
            iterations = final_state.get("iteration_count", 0)

            if is_casual:
                confidence = 1.0
                iterations = 0

            logger.info(f"Finished in {duration:.2f}s | Iterations: {iterations} | Confidence: {confidence:.2f}")

            results.append({
                "query": query,
                "duration": duration,
                "iterations": iterations,
                "confidence": confidence,
                "is_casual": is_casual,
                "status": "Success"
            })

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Failed after {duration:.2f}s: {e}")
            results.append({
                "query": query,
                "duration": duration,
                "iterations": 0,
                "confidence": 0.0,
                "is_casual": False,
                "status": f"Error: {e}"
            })

        # Give the API a brief rest to avoid rate limits
        await asyncio.sleep(5)

    # Generate Markdown Report
    report = "# NeuralDesk Evaluation Results\n\n"
    report += "| Query | Status | Time (s) | Iterations | Loop Required? | Confidence |\n"
    report += "|---|---|---|---|---|---|\n"

    total_time = 0
    total_conf = 0
    loop_count = 0
    success_count = 0

    for r in results:
        loop_req = "Yes" if r["iterations"] > 1 else "No"
        report += f"| {r['query']} | {r['status']} | {r['duration']:.2f} | {r['iterations']} | {loop_req} | {r['confidence']:.2f} |\n"

        if r["status"] == "Success":
            total_time += r["duration"]
            total_conf += r["confidence"]
            if r["iterations"] > 1:
                loop_count += 1
            success_count += 1

    if success_count > 0:
        avg_time = total_time / success_count
        avg_conf = total_conf / success_count
        loop_pct = (loop_count / success_count) * 100

        report += f"\n## Summary Statistics\n"
        report += f"- **Success Rate:** {success_count}/{len(TEST_QUERIES)}\n"
        report += f"- **Average Time:** {avg_time:.2f}s\n"
        report += f"- **Average Confidence:** {avg_conf:.2f}\n"
        report += f"- **Loop Required:** {loop_pct:.1f}%\n"

    with open("eval_results.md", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Evaluation complete! Results saved to eval_results.md")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
