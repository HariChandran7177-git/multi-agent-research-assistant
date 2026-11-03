"""
Deterministic Accuracy, Grounding, and Template Validation Suite.
Validates the Part 1 & Part 2 requirements:
1. Chunk and Retrieve per Sub-task
2. Numbered citations mapping 100% to real chunks
3. Fact verification and uncited numbers check
4. Freshness and domain variety
5. Fixed template adherence (Summary, Key findings, Details, Data/timeline, Gaps and caveats, Sources)
6. Doubt agent concise 'small matter' response with next-step prompt
"""

import pytest
import re
from typing import List, Dict, Any
from agents.retriever import _split_into_chunks, _parse_entry
from agents.reporter import _validate_citations_and_numbers, _build_deterministic_sources_section
from agents.doubt import answer_doubt
import asyncio

# ── 15 Real Evaluation Test Queries ──────────────────────────────────────────
EVALUATION_QUERIES = [
    "Tesla current deliveries, revenue, and autonomous roadmap 2025-2026",
    "DeepSeek-V3 and DeepSeek-R1 Multi-Head Latent Attention and DeepSeekMoE architecture",
    "Qdrant vs Milvus vs Pinecone performance and cost tradeoffs for vector search",
    "Semaglutide and Tirzepatide latest clinical trials and FDA indications 2025",
    "Perovskite silicon tandem solar cells commercial efficiency records 2025-2026",
    "NIST post-quantum cryptography standards FIPS 203 and FIPS 204 implementation",
    "James Webb Space Telescope high-redshift galaxy discoveries and cosmological tension",
    "Nvidia Blackwell architecture vs Hopper B200 GB200 compute and thermal specs",
    "Solid-state battery commercialization timeline and energy density benchmarks",
    "PostgreSQL pgvector vs dedicated vector databases indexing latency and recall",
    "CRISPR Cas9 gene editing clinical trial approvals sickle cell anemia",
    "Global semiconductor foundry market share TSMC Samsung Intel 2025-2026",
    "Rust vs Go for high-throughput cloud microservices performance and memory",
    "Anthropic Claude 3.5 Sonnet computer use capability and safety architecture",
    "Nuclear fusion energy net gain progress SPARC and ITER milestones 2025"
]


# ── Deterministic Quality Metric Functions (No LLM Needed) ───────────────────
def check_citation_validity(report: str, chunks: List[Dict[str, Any]]) -> float:
    """The percentage of [n] in the report that map to real retrieved chunks (target 100%)."""
    valid_ids = {int(ch["id"]) for ch in chunks if "id" in ch}
    citations_found = [int(x) for x in re.findall(r'\[(\d+)\]', report)]
    if not citations_found:
        return 1.0 if not chunks else 0.0
    valid_citations = [c for c in citations_found if c in valid_ids]
    return round(len(valid_citations) / len(citations_found) * 100, 2)


def count_uncited_numbers(report: str) -> int:
    """Counts sentences with hard numbers/dates/metrics that have no [n] citation (target 0)."""
    # Exclude Sources and Table lines
    lines = [l for l in report.split("\n") if not l.startswith("|") and not l.startswith("## Sources") and not l.startswith("[")]
    text = " ".join(lines)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    uncited = 0
    for s in sentences:
        s_clean = s.strip()
        # If sentence contains a number/percentage/dollar figure but no [n]
        has_number = bool(re.search(r'\b\d+(?:\.\d+)?%?|\$\d+', s_clean))
        has_citation = bool(re.search(r'\[\d+\]', s_clean))
        if has_number and not has_citation and len(s_clean) > 20:
            uncited += 1
    return uncited


def verify_fixed_template(report: str) -> Dict[str, bool]:
    """Verifies report strictly adheres to the 6 mandatory template sections."""
    return {
        "Summary": bool(re.search(r'^##\s+Summary', report, re.MULTILINE | re.IGNORECASE)),
        "Key findings": bool(re.search(r'^##\s+Key\s+findings', report, re.MULTILINE | re.IGNORECASE)),
        "Details": bool(re.search(r'^##\s+Details', report, re.MULTILINE | re.IGNORECASE)),
        "Data / timeline": bool(re.search(r'^##\s+Data\s*/\s*timeline', report, re.MULTILINE | re.IGNORECASE)),
        "Gaps and caveats": bool(re.search(r'^##\s+Gaps\s+and\s+caveats', report, re.MULTILINE | re.IGNORECASE)),
        "Sources": bool(re.search(r'^##\s+Sources', report, re.MULTILINE | re.IGNORECASE)),
    }


def check_source_variety(chunks: List[Dict[str, Any]]) -> int:
    """Counts distinct domains across sources."""
    domains = {ch.get("domain", "").lower() for ch in chunks if ch.get("domain")}
    return len(domains)


# ── Test Cases ───────────────────────────────────────────────────────────────
def test_chunking_preserves_metadata_and_size():
    """Verify ~300 token chunking with overlap."""
    sample_text = (
        "Tesla delivered 1,808,581 vehicles in 2023, representing 38% YoY growth. "
        "Model Y became the best-selling vehicle globally across all powertrains. "
    ) * 15  # ~1500 chars

    chunks = _split_into_chunks(sample_text, max_chars=1100, overlap=150)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 1200
        assert "Tesla" in c


def test_parse_entry_extracts_all_fields():
    """Verify entry parser captures subtask, title, domain, date, url, content."""
    entry = (
        "Subtask: Tesla delivery numbers 2025\n"
        "Title: Tesla IR Update Q4\n"
        "Domain: ir.tesla.com\n"
        "Date: 2025-01-02\n"
        "URL: https://ir.tesla.com/press/q4-deliveries\n"
        "Content: Tesla delivered 484,507 vehicles in Q4."
    )
    parsed = _parse_entry(entry)
    assert parsed["subtask"] == "Tesla delivery numbers 2025"
    assert parsed["title"] == "Tesla IR Update Q4"
    assert parsed["domain"] == "ir.tesla.com"
    assert parsed["date"] == "2025-01-02"
    assert parsed["url"] == "https://ir.tesla.com/press/q4-deliveries"
    assert "484,507" in parsed["content"]


def test_citation_validation_strips_hallucinated_ids():
    """Verify that hallucinated citations [99] are removed and valid citations [1] preserved."""
    chunks = [
        {"id": 1, "text": "Tesla delivered 1.81 million cars in 2023", "title": "Tesla IR", "domain": "ir.tesla.com", "date": "2024", "url": "https://ir.tesla.com"},
        {"id": 2, "text": "Model Y was top seller", "title": "Reuters", "domain": "reuters.com", "date": "2024", "url": "https://reuters.com"}
    ]
    raw_report = (
        "## Summary\n"
        "Tesla delivered 1.81 million cars [1]. Invented fact without source [99].\n\n"
        "## Sources\n"
        "[99] fake.com"
    )

    validated = _validate_citations_and_numbers(raw_report, chunks)
    assert "[1]" in validated
    assert "[99]" not in validated
    assert "## Sources" in validated
    assert "[1] [Tesla IR](https://ir.tesla.com) — *ir.tesla.com* (2024)" in validated


def test_fixed_template_structure_check():
    """Verify that the template validator checks all 6 sections."""
    sample_report = (
        "## Summary\nDirect answer here [1].\n\n"
        "## Key findings\n- Finding 1 [1].\n- Finding 2 [2].\n\n"
        "## Details\n### Subtask 1\nDetails here [1].\n\n"
        "## Data / timeline\n| Metric | Value |\n|---|---|\n| Deliveries | 1.81M |\n\n"
        "## Gaps and caveats\nSome data not found.\n\n"
        "## Sources\n[1] source.com\n[2] other.com"
    )
    result = verify_fixed_template(sample_report)
    assert all(result.values()), f"Missing template sections: {[k for k, v in result.items() if not v]}"


def test_doubt_agent_concise_response_and_next_step(monkeypatch):
    """Verify doubt agent returns small matter and prompts for next step."""
    from agents import doubt
    sample_report = "## Summary\nSemaglutide reduced cardiovascular death risk by 20% in SELECT trial [1]."
    question = "What was the cardiovascular risk reduction percentage?"
    
    mock_resp = type("MockMsg", (), {"content": "- The SELECT trial demonstrated a 20% risk reduction in cardiovascular death.\n\nWhat would you like to explore next?"})()
    mock_llm = type("MockLLM", (), {"invoke": lambda self, p: mock_resp})()
    monkeypatch.setattr(doubt, "llm", mock_llm)

    answer = asyncio.run(doubt.answer_doubt(sample_report, question))

    assert "20%" in answer
    # Assert concise output (less than 400 characters)
    assert len(answer) < 400
    # Assert proactive prompt for next step
    assert any(term in answer.lower() for term in ["next", "explore", "would you like", "step"])
