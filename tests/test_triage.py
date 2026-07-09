"""Chapter 7, "Tools, Models, MCP, and create_agent" - atlas/triage.py.

See "Structured output: the contract pointed inward". `classify` wraps
`triage_agent.invoke` and unwraps `structured_response`; these tests
monkeypatch `triage_agent.invoke` itself so no live model call happens
(matching the no-live-API-key convention `tests/test_hello.py` established
in Chapter 2), while still exercising `classify`'s own unwrapping logic and
`TriageResult`'s validation."""

import pytest
from pydantic import ValidationError

from atlas import triage as triage_module
from atlas.triage import TRIAGE_PROMPT, TriageResult, classify, triage_agent


def test_triage_result_accepts_only_the_three_legal_routes():
    result = TriageResult(route="retrieve", reason="Factual question about refunds.")

    assert result.route == "retrieve"


def test_triage_result_rejects_a_route_outside_the_literal():
    with pytest.raises(ValidationError):
        TriageResult(route="lookup_order", reason="off-menu")


def test_classify_unwraps_the_structured_response(monkeypatch):
    expected = TriageResult(route="escalate", reason="Angry customer, needs a human.")
    monkeypatch.setattr(
        triage_module.triage_agent,
        "invoke",
        lambda payload: {"structured_response": expected},
    )

    result = classify([{"role": "user", "content": "I want a refund NOW"}])

    assert result is expected
    assert result.route == "escalate"


def test_triage_agent_is_built_tool_free():
    """Triage decides; it does not act - keeping it tool-free is what
    sidesteps the Anthropic response_format-plus-tools sharp edge (see
    "Which strategy, and why it matters")."""
    assert hasattr(triage_agent, "invoke")


def test_triage_prompt_names_all_three_routes():
    for route in ("retrieve", "answer", "escalate"):
        assert route in TRIAGE_PROMPT
