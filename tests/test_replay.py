"""Chapter 21, "Evaluation and Testing" - atlas/replay.py.

See "Testing non-determinism: replaying a checkpoint". No live model, no
LangSmith connection - `replay_fixture` runs entirely against Chapter 1's
`ScriptedModel`, exercising `atlas.graph.build_graph`'s model-injection seam
end to end."""

from langchain_core.messages import AIMessage

from atlas.replay import replay_fixture


def test_replay_fixture_asserts_the_refund_route_with_zero_real_model_calls():
    """The chapter's own assertion, run for real: a scripted "refund"
    response drives triage to the refund route, and the fixture's own
    assert (inside replay_fixture) must not raise."""
    replay_fixture([AIMessage(content="refund")])


def test_replay_fixture_raises_when_the_scripted_route_is_not_refund():
    """The flip side of the same assertion: a scripted response that routes
    somewhere else (here, straight to "escalate") fails the fixture's own
    `route == "refund"` check."""
    import pytest

    with pytest.raises(AssertionError):
        replay_fixture([AIMessage(content="escalate")])
