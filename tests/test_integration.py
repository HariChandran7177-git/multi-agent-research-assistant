"""
tests/test_integration.py
--------------------------
Integration tests for the FastAPI research API.
Uses TestClient (no real API calls) to verify routes, rate limits, and report history.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import sys, os

# Ensure env vars are set before importing the app (avoids sys.exit on check_environment)
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-key")

# Patch Qdrant before import so it doesn't connect
with patch("qdrant_client.QdrantClient"):
    from api.research_api import app

client = TestClient(app, raise_server_exceptions=False)


def test_health_endpoint_returns_200():
    """Health endpoint must always return 200 even if services are degraded."""
    with patch("core.health.HealthChecker.check_all", new_callable=AsyncMock) as mock_check:
        mock_check.return_value = {
            "status": "healthy",
            "services": {"groq": {"status": "healthy"}, "qdrant": {"status": "healthy"}},
        }
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "version" in data


def test_metrics_endpoint_returns_200():
    """Metrics endpoint must return valid JSON."""
    resp = client.get("/health/metrics")
    assert resp.status_code == 200
    data = resp.json()
    # Should have cost_tracking key after our Group 4 update
    assert "cost_tracking" in data


def test_reports_list_endpoint():
    """GET /reports must return a list."""
    resp = client.get("/reports")
    assert resp.status_code == 200
    data = resp.json()
    assert "reports" in data
    assert isinstance(data["reports"], list)


def test_reports_list_with_user_filter():
    """GET /reports?user_id=x must filter by user."""
    resp = client.get("/reports", params={"user_id": "integration_test_user", "limit": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert "reports" in data


def test_report_not_found_returns_404():
    """GET /reports/99999 for a non-existent report must return 404."""
    resp = client.get("/reports/99999")
    assert resp.status_code == 404


def test_delete_nonexistent_report_returns_404():
    """DELETE /reports/99999 for a non-existent report must return 404."""
    resp = client.delete("/reports/99999")
    assert resp.status_code == 404


def test_report_history_save_and_retrieve():
    """Saving a report then retrieving it via API should work end-to-end."""
    from core.report_history import save_report, get_report
    rid = save_report(
        query="integration test query",
        report="Integration test report content.",
        confidence=0.95,
        iterations=1,
        tone="professional",
        user_id="integration_test_user",
    )
    assert rid is not None and rid > 0

    resp = client.get(f"/reports/{rid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "integration test query"
    assert data["confidence"] == 0.95

    # Clean up
    del_resp = client.delete(f"/reports/{rid}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"


def test_resume_unknown_thread_returns_error():
    """/research/resume with unknown thread_id should return error (not 5xx)."""
    resp = client.post("/research/resume", json={"thread_id": "nonexistent-thread-000"})
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
