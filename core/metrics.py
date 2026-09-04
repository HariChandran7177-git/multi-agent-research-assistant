"""
Metrics and observability for the research pipeline.
Tracks timing, errors, token usage, and cost signals per agent and per request.
"""

import os
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from core.logger import get_logger

logger = get_logger(__name__)

# ── LangSmith Tracing ─────────────────────────────────────────────────────────
# Enable via LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in your .env
_ls_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
if _ls_enabled and os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "multi-agent-research-assistant")
    logger.info("[LangSmith] Tracing enabled — project: " + os.environ["LANGCHAIN_PROJECT"])
else:
    logger.debug("[LangSmith] Tracing disabled (set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY to enable)")

# Groq token pricing (approximate, per 1M tokens, USD)
_GROQ_PRICING = {
    "openai/gpt-oss-20b": {"input": 0.59, "output": 0.79},
    "groq/compound-mini": {"input": 0.05, "output": 0.08},
}
_TAVILY_COST_PER_CALL = 0.001  # ~$0.001 per Tavily search call


@dataclass
class AgentMetrics:
    """Metrics for a single agent execution."""
    agent_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    error: Optional[str] = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    retry_count: int = 0
    model: str = ""               # which LLM model was used
    estimated_cost_usd: float = 0.0  # estimated cost in USD


@dataclass
class RequestCost:
    """Tracks total cost for one research request."""
    query_hash: str = ""
    groq_input_tokens: int = 0
    groq_output_tokens: int = 0
    tavily_calls: int = 0
    embedding_calls: int = 0
    estimated_usd: float = 0.0
    started_at: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and aggregates metrics across the pipeline."""

    def __init__(self):
        self.current_metrics: Dict[str, AgentMetrics] = {}
        self.history: list = []
        self.total_requests = 0
        self.total_errors = 0
        self.current_request: Optional[RequestCost] = None
        self.request_history: list = []  # list of RequestCost

    def start_agent(self, agent_name: str):
        """Start timing for an agent."""
        self.current_metrics[agent_name] = AgentMetrics(agent_name=agent_name)
        self.current_metrics[agent_name].start_time = time.perf_counter()

    def start_request(self, query: str):
        """Begin tracking cost for a new research request."""
        import hashlib
        self.current_request = RequestCost(
            query_hash=hashlib.md5(query.encode()).hexdigest()[:8]
        )

    def record_groq_tokens(self, model: str, input_tokens: int, output_tokens: int):
        """Record Groq token usage and estimate cost."""
        if self.current_request is None:
            return
        self.current_request.groq_input_tokens += input_tokens
        self.current_request.groq_output_tokens += output_tokens
        pricing = _GROQ_PRICING.get(model, {"input": 0.30, "output": 0.30})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
        self.current_request.estimated_usd += cost

    def record_tavily_call(self, num_calls: int = 1):
        """Record Tavily search API call(s)."""
        if self.current_request is None:
            return
        self.current_request.tavily_calls += num_calls
        self.current_request.estimated_usd += num_calls * _TAVILY_COST_PER_CALL

    def record_embedding_call(self, num_texts: int = 1):
        """Record local SentenceTransformers embedding call(s)."""
        if self.current_request is None:
            return
        self.current_request.embedding_calls += num_texts

    def end_request(self):
        """Finalize and store the current request cost."""
        if self.current_request is not None:
            self.request_history.append(self.current_request)
            # Keep last 100 requests only
            if len(self.request_history) > 100:
                self.request_history = self.request_history[-100:]
            self.current_request = None

    def end_agent(self, agent_name: str, error: str = None, **kwargs):
        """End timing for an agent and record metrics."""
        if agent_name in self.current_metrics:
            m = self.current_metrics[agent_name]
            m.end_time = time.perf_counter()
            m.latency_ms = (m.end_time - m.start_time) * 1000
            m.error = error
            m.input_tokens = kwargs.get("input_tokens", 0)
            m.output_tokens = kwargs.get("output_tokens", 0)
            m.retry_count = kwargs.get("retry_count", 0)
            self.history.append(m)
            del self.current_metrics[agent_name]

    def record_request(self, success: bool = True):
        """Record a complete request."""
        self.total_requests += 1
        if not success:
            self.total_errors += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get current metrics summary including cost tracking."""
        if not self.history:
            base = {"status": "no_data"}
        else:
            # Aggregate by agent
            agent_stats = defaultdict(lambda: {"count": 0, "total_ms": 0, "errors": 0, "tokens_in": 0, "tokens_out": 0})
            for m in self.history:
                stats = agent_stats[m.agent_name]
                stats["count"] += 1
                stats["total_ms"] += m.latency_ms
                if m.error:
                    stats["errors"] += 1
                stats["tokens_in"] += m.input_tokens
                stats["tokens_out"] += m.output_tokens

            summary = {}
            for agent, stats in agent_stats.items():
                avg_latency = stats["total_ms"] / stats["count"] if stats["count"] > 0 else 0
                summary[agent] = {
                    "count": stats["count"],
                    "avg_latency_ms": round(avg_latency, 2),
                    "error_rate": round(stats["errors"] / stats["count"], 3) if stats["count"] > 0 else 0,
                    "tokens_in": stats["tokens_in"],
                    "tokens_out": stats["tokens_out"],
                }
            base = {
                "summary": summary,
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": round(self.total_errors / self.total_requests, 3) if self.total_requests > 0 else 0,
            }

        # Add cost summary
        total_cost = sum(r.estimated_usd for r in self.request_history)
        avg_cost = total_cost / len(self.request_history) if self.request_history else 0
        base["cost_tracking"] = {
            "langsmith_enabled": _ls_enabled,
            "total_requests_tracked": len(self.request_history),
            "total_estimated_usd": round(total_cost, 6),
            "avg_cost_per_request_usd": round(avg_cost, 6),
            "last_request": {
                "groq_input_tokens": self.request_history[-1].groq_input_tokens,
                "groq_output_tokens": self.request_history[-1].groq_output_tokens,
                "tavily_calls": self.request_history[-1].tavily_calls,
                "estimated_usd": round(self.request_history[-1].estimated_usd, 6),
            } if self.request_history else None,
        }
        return base

    def reset(self):
        """Reset all metrics."""
        self.current_metrics.clear()
        self.history.clear()
        self.total_requests = 0
        self.total_errors = 0


# Global metrics collector
metrics = MetricsCollector()


def track_agent(agent_name: str):
    """Decorator to track agent execution metrics."""
    def decorator(func):
        async def wrapper(state, *args, **kwargs):
            from core.metrics import metrics
            metrics.start_agent(agent_name)
            try:
                result = await func(state, *args, **kwargs)
                metrics.end_agent(agent_name)
                return result
            except Exception as e:
                metrics.end_agent(agent_name, error=str(e))
                raise
        return wrapper
    return decorator
