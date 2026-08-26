"""
Metrics and observability for the research pipeline.
Tracks timing, errors, and quality signals per agent.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from core.logger import get_logger

logger = get_logger(__name__)


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


class MetricsCollector:
    """Collects and aggregates metrics across the pipeline."""

    def __init__(self):
        self.current_metrics: Dict[str, AgentMetrics] = {}
        self.history: list = []
        self.total_requests = 0
        self.total_errors = 0

    def start_agent(self, agent_name: str):
        """Start timing for an agent."""
        self.current_metrics[agent_name] = AgentMetrics(agent_name=agent_name)
        self.current_metrics[agent_name].start_time = time.perf_counter()

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
        """Get current metrics summary."""
        if not self.history:
            return {"status": "no_data"}

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

        # Calculate averages
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

        return {
            "summary": summary,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": round(self.total_errors / self.total_requests, 3) if self.total_requests > 0 else 0,
        }

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
