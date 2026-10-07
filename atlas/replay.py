"""Chapter 21, "Evaluation and Testing" - checkpoint-replay fixtures.

See "Testing non-determinism: replaying a checkpoint". CI (`atlas/evals.py`)
and the online monitor (`atlas/monitor.py`) both call a real model. A third
kind of test needs to run without one - asserting on Atlas's *deterministic*
machinery (routing logic, state transitions, what a resumed or replayed run
does) without paying for a model call or risking a flaky answer failing the
build. Chapter 1's `ScriptedModel` (`atlas.breaks`) and Chapter 17's
`build_graph(model=...)` seam, which this chapter gives its use, make the run
deterministic; Chapter 9's checkpoint history makes it replayable.

`replay_fixture` runs a refund request once, against a scripted triage, and
asserts the route. The scripted triage routes to `approval_gate` (Chapter
11), which reads `state["ticket"]` and pauses at its interrupt; `get_state`
reads that paused state, where `route` is already set. It then replays the
run from the checkpoint saved just before `triage` - `get_state_history`
finds it, and `invoke(None, <that checkpoint's config>)` re-executes from it
(Chapter 9) - and asserts the replayed run takes the same route. The ticket
is the book-wide refund input: Chapter 10's seeded `T-1001`, with the
`customer_id` Chapter 13's `recall` reads."""

from langchain_core.messages import AIMessage

from atlas.breaks import ScriptedModel
from atlas.graph import build_graph

config = {"configurable": {"thread_id": "regression-fixture-1"}}
TICKET = {"id": "T-1001", "amount": 49.0, "customer_id": "C-1"}


def replay_fixture(scripted_responses: list[AIMessage]) -> ScriptedModel:
    """Run a captured conversation shape against a scripted model, then
    replay it from its saved checkpoint - zero real model calls."""
    model = ScriptedModel(scripted_responses)
    fixture_graph = build_graph(model=model)
    fixture_graph.invoke(
        {
            "messages": [{"role": "user", "content": "I need a refund for T-1001."}],
            "ticket": TICKET,
        },
        config,
    )
    assert fixture_graph.get_state(config).values["route"] == "refund"

    before_triage = next(
        s for s in fixture_graph.get_state_history(config) if s.next == ("triage",)
    )
    fixture_graph.invoke(None, before_triage.config)
    assert fixture_graph.get_state(config).values["route"] == "refund"
    return model
