from core.graph import should_loop
from core.config import CONFIDENCE_THRESHOLD, MAX_ITERATIONS


def test_high_confidence_goes_to_reporter():
    state = {"confidence_score": 0.9, "iteration_count": 1}
    result = should_loop(state)
    assert result == "reporter"


def test_low_confidence_under_cap_loops_to_researcher():
    state = {"confidence_score": 0.3, "iteration_count": 1}
    result = should_loop(state)
    assert result == "researcher"


def test_low_confidence_at_cap_forces_reporter():
    state = {"confidence_score": 0.3, "iteration_count": MAX_ITERATIONS}
    result = should_loop(state)
    assert result == "reporter"


def test_confidence_exactly_at_threshold_goes_to_reporter():
    state = {"confidence_score": CONFIDENCE_THRESHOLD, "iteration_count": 0}
    result = should_loop(state)
    assert result == "reporter"
