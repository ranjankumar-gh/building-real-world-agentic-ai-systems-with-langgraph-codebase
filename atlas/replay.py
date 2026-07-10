"""Chapter 21, "Evaluation and Testing" - checkpoint-replay fixtures.

See "Testing non-determinism: replaying a checkpoint". CI (`atlas/evals.py`)
and the online monitor (`atlas/monitor.py`) both call a real model. A third
kind of test needs to run without one - asserting on Atlas's *deterministic*
machinery (routing logic, handoff bounds, state transitions) without paying
for a model call or risking a flaky non-deterministic answer failing the
build. Chapter 1's `ScriptedModel` (`atlas.breaks`) and `atlas.graph`'s
`build_graph(model=...)` (this chapter's small, additive refactor to
`atlas/graph.py`) combine to make this a fixture instead of a live call.

`approval_gate` (Chapter 11) reads `state["ticket"]` before the scripted
model's route decision ever reaches it, so a refund-routed fixture needs a
`ticket` on the input the same way the real refund flow does - the id and
amount below echo `atlas/evals.py`'s own refund dataset example, for the
same order."""

from atlas.breaks import ScriptedModel
from atlas.graph import build_graph

config = {"configurable": {"thread_id": "regression-fixture-1"}}


def replay_fixture(scripted_responses: list) -> None:
    """Replay a captured conversation shape against a scripted model -
    deterministic assertions, zero real model calls."""
    fixture_graph = build_graph(model=ScriptedModel(scripted_responses))
    fixture_graph.invoke(
        {
            "messages": [{"role": "user", "content": "I want a refund for order 4471."}],
            "ticket": {"id": "4471", "amount": 340.0},
        },
        config,
    )
    snapshot = fixture_graph.get_state(config)
    assert snapshot.values["route"] == "refund"
