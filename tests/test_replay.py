"""Chapter 21, "Evaluation and Testing" - atlas/replay.py.

See "Testing non-determinism: replaying a checkpoint". No live model, no
LangSmith connection - `replay_fixture` runs entirely against Chapter 1's
`ScriptedModel`, exercising `atlas.graph.build_graph`'s model-injection seam
end to end, then replays the run from its saved checkpoint."""

import pytest
from langchain_core.messages import AIMessage

from atlas.breaks import ScriptedModel
from atlas.graph import build_graph
from atlas.replay import TICKET, config, replay_fixture


def test_replay_fixture_asserts_the_refund_route_with_zero_real_model_calls():
    """The chapter's own assertions, run for real: a scripted "refund"
    response drives triage to the refund route, and the replay from the
    checkpoint before triage takes the same route."""
    replay_fixture([AIMessage(content="refund")])


def test_the_replay_re_runs_triage_from_the_saved_checkpoint():
    """Two triage calls on one scripted model: the first run, then the
    replay - the replay executes triage again rather than reading a result."""
    model = replay_fixture([AIMessage(content="refund")])
    assert model.calls == 2


def test_the_replay_fails_when_the_replayed_route_changes():
    """The replay is a real re-execution: script a different second reply
    and the fixture's second assertion catches it."""
    with pytest.raises(AssertionError):
        replay_fixture([AIMessage(content="refund"), AIMessage(content="escalate")])


def test_replay_fixture_raises_when_the_scripted_route_is_not_refund():
    with pytest.raises(AssertionError):
        replay_fixture([AIMessage(content="escalate")])


def test_the_first_run_pauses_at_the_approval_gate():
    """What get_state reads: a run paused at approval_gate's interrupt,
    with the route already set."""
    fixture_graph = build_graph(model=ScriptedModel([AIMessage(content="refund")]))
    fixture_graph.invoke(
        {"messages": [{"role": "user", "content": "refund"}], "ticket": TICKET},
        config,
    )
    snapshot = fixture_graph.get_state(config)
    assert snapshot.next == ("approval_gate",)
    assert snapshot.values["route"] == "refund"


def test_a_refund_fixture_without_a_ticket_raises_at_the_gate():
    fixture_graph = build_graph(model=ScriptedModel([AIMessage(content="refund")]))
    with pytest.raises(KeyError, match="ticket"):
        fixture_graph.invoke({"messages": [{"role": "user", "content": "x"}]}, config)
