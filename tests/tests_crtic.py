import pytest
from agents.critic import critic_node


class FakeResponse:
    """Mimics the object Groq's llm.invoke() normally returns — just needs a .content attribute."""

    def __init__(self, content):
        self.content = content


def make_fake_state():
    return {
        "query": "test query",
        "research_results": ["some finding"],
        "retrieved_docs": ["some doc"],
        "iteration_count": 0,
    }


def test_valid_json_parses_correctly(monkeypatch):
    fake_json = '{"score": 0.85, "critique": "Solid coverage"}'

    def fake_invoke(llm, prompt):
        return FakeResponse(fake_json)

    monkeypatch.setattr("agents.critic.invoke_with_retry", fake_invoke)

    result = critic_node(make_fake_state())

    assert result["confidence_score"] == 0.85
    assert result["critique"] == "Solid coverage"


def test_malformed_json_falls_back_to_default_score(monkeypatch):
    broken_json = "this is not valid json at all"

    def fake_invoke(llm, prompt):
        return FakeResponse(broken_json)

    monkeypatch.setattr("agents.critic.invoke_with_retry", fake_invoke)

    result = critic_node(make_fake_state())

    assert result["confidence_score"] == 0.5  # the documented fallback value
    assert "Could not parse" in result["critique"]


def test_json_with_missing_score_field_falls_back(monkeypatch):
    incomplete_json = '{"critique": "Missing the score field entirely"}'

    def fake_invoke(llm, prompt):
        return FakeResponse(incomplete_json)

    monkeypatch.setattr("agents.critic.invoke_with_retry", fake_invoke)

    result = critic_node(make_fake_state())

    # falls back since "score" key missing
    assert result["confidence_score"] == 0.5


def test_markdown_fenced_json_still_parses(monkeypatch):
    fenced_json = '```json\n{"score": 0.7, "critique": "Decent"}\n```'

    def fake_invoke(llm, prompt):
        return FakeResponse(fenced_json)

    monkeypatch.setattr("agents.critic.invoke_with_retry", fake_invoke)

    result = critic_node(make_fake_state())

    assert result["confidence_score"] == 0.7


def test_iteration_count_increments(monkeypatch):
    def fake_invoke(llm, prompt):
        return FakeResponse('{"score": 0.9, "critique": "Good"}')

    monkeypatch.setattr("agents.critic.invoke_with_retry", fake_invoke)

    state = make_fake_state()
    state["iteration_count"] = 2
    result = critic_node(state)

    assert result["iteration_count"] == 3
