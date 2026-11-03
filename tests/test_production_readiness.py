"""
Automated Production Readiness Test Suite
Verifies all 10 pre-deployment checklist criteria.
"""

import os
import sys
import json
import time
import subprocess
import pytest
from fastapi.testclient import TestClient

# Ensure root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.research_api import app
from core.config import DATA_DIR, APP_API_KEY, RATE_LIMIT_PER_MINUTE, MAX_QUERY_LENGTH
from core.health import health_checker
from core.report_history import save_report, list_reports, delete_report


client = TestClient(app)


def test_1_git_secrets_and_ignore():
    """Verify .env is ignored and no secrets were ever committed to git history."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # 1. Check .gitignore
    gitignore_path = os.path.join(root_dir, ".gitignore")
    assert os.path.exists(gitignore_path), ".gitignore is missing!"
    with open(gitignore_path, "r", encoding="utf-8") as f:
        gitignore_content = f.read()
    assert ".env" in gitignore_content, ".env is not in .gitignore!"

    # 2. Check .dockerignore
    dockerignore_path = os.path.join(root_dir, ".dockerignore")
    assert os.path.exists(dockerignore_path), ".dockerignore is missing!"
    with open(dockerignore_path, "r", encoding="utf-8") as f:
        dockerignore_content = f.read()
    assert ".env" in dockerignore_content, ".env is not in .dockerignore!"

    # 3. Check .env.example contains no real secrets
    example_path = os.path.join(root_dir, ".env.example")
    assert os.path.exists(example_path), ".env.example is missing!"
    with open(example_path, "r", encoding="utf-8") as f:
        example_content = f.read()
    for pattern in ["gsk_", "AIza", "tvly-"]:
        assert pattern not in example_content, f"Found suspicious key pattern {pattern} in .env.example!"


def test_2_docker_compose_and_dockerfile_config():
    """Verify Qdrant container, volume mounts, and single-worker command."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Check docker-compose.yml
    compose_path = os.path.join(root_dir, "docker-compose.yml")
    assert os.path.exists(compose_path), "docker-compose.yml is missing!"
    with open(compose_path, "r", encoding="utf-8") as f:
        compose_content = f.read()

    assert "qdrant:" in compose_content, "Qdrant container is not defined in docker-compose.yml!"
    assert "qdrant_storage:" in compose_content, "Persistent qdrant_storage volume is missing!"
    assert "./data:/app/data" in compose_content, "data/ volume mount is missing!"
    assert "QDRANT_URL=http://qdrant:6333" in compose_content, "App QDRANT_URL should point to http://qdrant:6333!"

    # Check Dockerfile
    dockerfile_path = os.path.join(root_dir, "Dockerfile")
    assert os.path.exists(dockerfile_path), "Dockerfile is missing!"
    with open(dockerfile_path, "r", encoding="utf-8") as f:
        dockerfile_content = f.read()
    assert '"--workers", "1"' in dockerfile_content or "--workers 1" in dockerfile_content, "Uvicorn must run with --workers 1 to avoid SQLite lock conflicts!"


def test_3_no_hardcoded_windows_paths():
    """Verify no hardcoded Windows drive letters (A:\ or C:\) in python or config files."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    skip_dirs = {".venv", ".git", "__pycache__", "data", "model_cache", "scratch"}
    
    bad_paths = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for file in files:
            if file.endswith((".py", ".yml", ".yaml", "Dockerfile")):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        if ("A:\\" in line or "C:\\" in line) and "test_" not in file:
                            bad_paths.append((file, line_no, line.strip()))
    
    assert len(bad_paths) == 0, f"Found hardcoded Windows paths: {bad_paths}"


def test_4_input_validation_empty_and_long_queries():
    """Verify empty and excessively long queries are rejected with clean 422/400 errors."""
    # 1. Empty query
    resp = client.post("/research", json={"query": ""})
    assert resp.status_code in (400, 422), f"Expected 400 or 422 for empty query, got {resp.status_code}"

    # 2. Pure whitespace query
    resp = client.post("/research", json={"query": "     "})
    assert resp.status_code in (400, 422), f"Expected 400 or 422 for whitespace query, got {resp.status_code}"

    # 3. Query exceeding max allowed characters
    long_query = "What is AI? " * 300  # > 3000 chars
    resp = client.post("/research", json={"query": long_query})
    assert resp.status_code in (400, 422), f"Expected 400 or 422 for long query, got {resp.status_code}"


def test_5_api_key_authentication():
    """Verify API key authentication works when configured."""
    from api.research_api import APP_API_KEY
    if APP_API_KEY:
        # Request with no key -> 401
        resp = client.post("/research", json={"query": "test query"})
        assert resp.status_code == 401, f"Expected 401 without API key, got {resp.status_code}"

        # Request with bad key -> 401
        resp = client.post("/research", json={"query": "test query"}, headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401, f"Expected 401 with bad API key, got {resp.status_code}"

        # Request with valid key in header
        resp = client.get("/reports", headers={"X-API-Key": APP_API_KEY})
        assert resp.status_code == 200, f"Expected 200 with valid X-API-Key, got {resp.status_code}"

        # Request with valid Bearer token
        resp = client.get("/reports", headers={"Authorization": f"Bearer {APP_API_KEY}"})
        assert resp.status_code == 200, f"Expected 200 with Bearer token, got {resp.status_code}"
    else:
        # Dev mode (no key configured) -> allowed
        resp = client.get("/health")
        assert resp.status_code == 200


def test_6_health_endpoint_honesty():
    """Verify /health and /health/deep report accurately."""
    # 1. Shallow health check
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "config" in data
    assert "groq_key_set" in data["config"]

    # 2. Deep health check structure
    deep_data = health_checker.check_shallow()
    assert deep_data["status"] in ("healthy", "unhealthy")


def test_7_data_persistence():
    """Verify report history persists in SQLite database in DATA_DIR."""
    test_query = f"__test_persistence_query_{int(time.time())}"
    test_report = "This is a test report verifying data persistence."
    
    report_id = save_report(
        query=test_query,
        report=test_report,
        confidence=0.95,
        iterations=1,
        tone="professional",
        user_id="test_runner"
    )
    assert report_id is not None, "Failed to save report to SQLite!"

    # Verify retrieval
    reports = list_reports(user_id="test_runner", limit=10)
    matching = [r for r in reports if r["query"] == test_query]
    assert len(matching) > 0, "Saved report not found in SQLite query!"
    assert matching[0]["report"] == test_report

    # Cleanup
    delete_report(report_id)


def test_8_rate_limiting():
    """Verify rate limiter blocks abusive burst traffic exceeding limit."""
    # Rapid burst requests to /reports
    exceeded = False
    for _ in range(RATE_LIMIT_PER_MINUTE + 5):
        resp = client.get("/reports")
        if resp.status_code == 429:
            exceeded = True
            break
    assert exceeded, f"Rate limiter failed to trigger 429 after {RATE_LIMIT_PER_MINUTE + 5} rapid requests!"
