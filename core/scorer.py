"""
core/scorer.py
--------------
Objective, rule-based quality signals for research evaluation.
Used by the Critic to complement (and ground-truth-check) its LLM score.

Signals
-------
1. Coverage          - Did we collect enough raw material?
2. Plan Coverage     - Did results address every planned sub-task?
3. Content Depth     - Are results detailed, or just thin snippets?
4. Diversity         - Are results varied, or all saying the same thing?
5. Duplicate Penalty - Are there near-identical results wasting slots?
6. Retrieval Relevance - How closely do retrieved docs match the query? (Qdrant scores)
7. Source Quality    - Are URLs from credible domains?

Final hybrid score blends LLM judgment (60%) with the objective signal (40%).
LLM is good at semantic quality; objective catches structural failures the LLM
can miss (e.g. verbose-but-thin content, duplicates, skipped sub-tasks).
"""

import re
from urllib.parse import urlparse
from core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Individual signals
# ---------------------------------------------------------------------------

def calculate_coverage_score(research_results: list, target_count: int = 8) -> float:
    """Did we gather enough raw material? Maxes out at target_count results."""
    score = min(len(research_results) / target_count, 1.0)
    logger.debug(f"[Scorer] Coverage: {len(research_results)}/{target_count} -> {score:.2f}")
    return score


def calculate_plan_coverage(plan: list, research_results: list) -> float:
    """
    For each planned sub-task, checks if at least one result addresses it
    by keyword overlap. Penalizes plans where sub-tasks were skipped entirely.
    """
    if not plan or not research_results:
        return 0.5  # neutral if no plan or no results

    combined = " ".join(research_results).lower()
    covered = 0
    for task in plan:
        # Extract meaningful keywords (length > 4 to skip stop words)
        keywords = [w for w in re.findall(r"[a-zA-Z]+", task.lower()) if len(w) > 4]
        if keywords and any(kw in combined for kw in keywords):
            covered += 1

    score = covered / len(plan)
    logger.debug(f"[Scorer] Plan coverage: {covered}/{len(plan)} tasks -> {score:.2f}")
    return score


def calculate_depth_score(research_results: list,
                           shallow_threshold: int = 150,
                           deep_threshold: int = 500) -> float:
    """
    Penalizes very short (thin/scraped) results; rewards detailed ones.
    shallow < 150 chars -> 0.2  |  150-500 chars -> 0.7  |  > 500 chars -> 1.0
    """
    if not research_results:
        return 0.0

    scores = []
    for r in research_results:
        length = len(r.strip())
        if length < shallow_threshold:
            scores.append(0.2)
        elif length < deep_threshold:
            scores.append(0.7)
        else:
            scores.append(1.0)

    score = sum(scores) / len(scores)
    logger.debug(f"[Scorer] Depth score: {score:.2f} (avg over {len(research_results)} results)")
    return score


def calculate_diversity_score(research_results: list) -> float:
    """
    Measures result variety by comparing the opening phrase of each result.
    Results sharing the same first 50 characters are likely near-duplicates.
    """
    if len(research_results) <= 1:
        return 0.5

    openings = [r[:50].lower().strip() for r in research_results]
    unique_openings = len(set(openings))
    score = unique_openings / len(openings)
    logger.debug(f"[Scorer] Diversity: {unique_openings}/{len(openings)} unique -> {score:.2f}")
    return score


def calculate_duplicate_penalty(research_results: list) -> float:
    """
    Detects near-identical results using an 80-char fingerprint.
    Returns 1.0 (no duplicates) -> lower values as duplicates increase.
    """
    if len(research_results) <= 1:
        return 1.0

    seen = set()
    duplicates = 0
    for r in research_results:
        fingerprint = re.sub(r"\s+", " ", r.lower().strip())[:80]
        if fingerprint in seen:
            duplicates += 1
        seen.add(fingerprint)

    score = 1.0 - (duplicates / len(research_results))
    logger.debug(f"[Scorer] Duplicate penalty: {duplicates} dupes -> score {score:.2f}")
    return score


def calculate_retrieval_relevance(qdrant_scores: list) -> float:
    """
    Uses Qdrant's own cosine similarity scores from the retrieval step.
    These are the most reliable signal - directly measure query-document match.
    """
    if not qdrant_scores:
        return 0.65  # neutral default when scores not available

    score = sum(qdrant_scores) / len(qdrant_scores)
    logger.debug(f"[Scorer] Retrieval relevance (Qdrant avg): {score:.2f} over {len(qdrant_scores)} docs")
    return score


def calculate_source_quality(research_results: list) -> float:
    """
    Extracts URLs from formatted research results and scores domain credibility.
    Trusted domains (.edu, .gov, .org, known publications) score 1.0.
    Unknown domains score 0.6. Missing URL scores 0.4.
    """
    trusted = {".edu", ".gov", ".org", ".ac.uk", ".ac.in"}
    known_publications = {
        "nature.com", "science.org", "arxiv.org", "pubmed.ncbi",
        "reuters.com", "bbc.com", "nytimes.com", "wired.com",
        "techcrunch.com", "forbes.com", "bloomberg.com"
    }

    url_pattern = re.compile(r"URL:\s*(https?://[^\s]+)")
    scores = []

    for result in research_results:
        match = url_pattern.search(result)
        if not match:
            scores.append(0.4)
            continue
        try:
            domain = urlparse(match.group(1)).netloc.lower().lstrip("www.")
            if any(domain.endswith(t) for t in trusted):
                scores.append(1.0)
            elif any(pub in domain for pub in known_publications):
                scores.append(0.9)
            else:
                scores.append(0.6)
        except Exception:
            scores.append(0.4)

    if not scores:
        return 0.5

    score = sum(scores) / len(scores)
    logger.debug(f"[Scorer] Source quality: {score:.2f} over {len(scores)} sources")
    return score


# ---------------------------------------------------------------------------
# Composite objective score
# ---------------------------------------------------------------------------

def calculate_objective_score(state: dict, qdrant_scores: list = None) -> dict:
    """
    Combines all signals into one objective quality score, independent of LLM judgment.

    Weights (quality-first design):
      Retrieval Relevance  35%  - most reliable; Qdrant directly measures query-doc match
      Plan Coverage        20%  - did we actually answer what was planned?
      Content Depth        20%  - is the content detailed, not just thin snippets?
      Source Quality       10%  - credibility of domains found
      Duplicate Penalty    10%  - penalize redundant results
      Diversity             5%  - variety of perspectives (least critical)

    Returns a dict with the composite score AND a breakdown for full transparency.
    """
    research_results = state.get("research_results", [])
    plan = state.get("plan", [])

    relevance       = calculate_retrieval_relevance(qdrant_scores)
    plan_coverage   = calculate_plan_coverage(plan, research_results)
    depth           = calculate_depth_score(research_results)
    source_quality  = calculate_source_quality(research_results)
    dup_penalty     = calculate_duplicate_penalty(research_results)
    diversity       = calculate_diversity_score(research_results)

    composite = (
        (relevance      * 0.35) +
        (plan_coverage  * 0.20) +
        (depth          * 0.20) +
        (source_quality * 0.10) +
        (dup_penalty    * 0.10) +
        (diversity      * 0.05)
    )
    composite = round(min(max(composite, 0.0), 1.0), 3)

    breakdown = {
        "retrieval_relevance": round(relevance,      3),
        "plan_coverage":       round(plan_coverage,  3),
        "content_depth":       round(depth,          3),
        "source_quality":      round(source_quality, 3),
        "duplicate_penalty":   round(dup_penalty,    3),
        "diversity":           round(diversity,      3),
        "objective_score":     composite,
    }

    logger.info(f"[Scorer] Objective breakdown: {breakdown}")
    return breakdown


# ---------------------------------------------------------------------------
# Hybrid score: LLM + Objective
# ---------------------------------------------------------------------------

def calculate_hybrid_score(llm_score: float, objective_breakdown: dict,
                            llm_weight: float = 0.07,
                            obj_weight: float = 0.93) -> float:
    """
    Blends objective rule-based score (93%) with a small LLM sanity-check (7%).

    The objective signals dominate because they are deterministic and cannot be
    fooled by fluent-but-shallow LLM output. The LLM contribution is kept at
    just 7% — enough to act as a mild semantic tiebreaker on edge cases, but
    not enough to override hard structural failures detected by the scorer.

    Objective catches:
        * Verbose-but-thin content that 'looks' good to an LLM
        * Duplicated results inflating apparent coverage
        * Sub-tasks that were silently skipped by the researcher
        * Low Qdrant similarity hidden behind fluent prose

    The hybrid score is what gets written to state['confidence_score'].
    """
    obj_score = objective_breakdown.get("objective_score", 0.5)
    hybrid = round((llm_score * llm_weight) + (obj_score * obj_weight), 3)
    logger.info(
        f"[Scorer] Hybrid score: LLM={llm_score:.3f} x {llm_weight} + "
        f"Objective={obj_score:.3f} x {obj_weight} = {hybrid:.3f}"
    )
    return hybrid
