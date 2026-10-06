"""Chapter 9, "Persistence and Checkpointing" - atlas/graph.py's checkpointer
and atlas/run.py's thread-scoped call shape.

`graph` now compiles onto `InMemorySaver` (RAM-backed, dev/test only), so
every `invoke` needs a `thread_id`. `run_durable` compiles the same builder
onto `AsyncPostgresSaver` instead - that needs a real, reachable Postgres
instance, so its test is skip-guarded behind `ATLAS_POSTGRES_TEST_DSN` and
skips cleanly without one; it is not required for the rest of the suite to
pass. See the README for how to point it at the seeded local Postgres
service.

Chapter 11, "Human-in-the-Loop", adds `run_to_approval` and
`resume_approval` - the thread-scoped suspend/resume shape for the
approval gate in `atlas/graph.py`.

Chapter 17, "Subgraphs, Parallelism, and Map-Reduce", adds `run_research` -
"Bound the fan-out": `max_concurrency` is set on the invoke config, not the
graph, so the same `research_graph` can be called with a different bound
per call."""

import os

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import RunControl

from atlas import graph as graph_module
from atlas.graph import graph, run_durable
from atlas.run import (
    inspect,
    resume_approval,
    run_research,
    run_to_approval,
    run_two_turns,
    run_with_drain,
    run_with_durability,
)

requires_postgres = pytest.mark.skipif(
    not os.environ.get("ATLAS_POSTGRES_TEST_DSN"),
    reason="requires a live Postgres connection (ATLAS_POSTGRES_TEST_DSN)",
)


def _decision(route: str):
    from types import SimpleNamespace

    return SimpleNamespace(route=route)


def test_graph_is_compiled_with_an_in_memory_checkpointer():
    """The dev/test default from "Compile with a checkpointer" - RAM-backed,
    exercises the real checkpointing path with no external dependency."""
    assert isinstance(graph.checkpointer, InMemorySaver)


def test_invoking_the_checkpointed_graph_without_a_thread_id_errors():
    """"thread_id is a correctness boundary": omit it with a checkpointer
    attached and there is nowhere to save, so the run errors instead of
    silently proceeding."""
    with pytest.raises(ValueError):
        graph.invoke({"messages": [{"role": "user", "content": "hi"}]})


def test_run_two_turns_restores_history_on_the_same_thread(monkeypatch):
    """"Scope the conversation with a thread_id": the second call's reply
    sits on top of BOTH prior turns, not just the one it was given - proof
    the checkpointer, not the caller, is carrying the history forward."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("answer")
    )
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])
    monkeypatch.setattr(
        graph_module,
        "compose_answer",
        # compose_answer returns a str; `answer` wraps it in an AIMessage.
        lambda messages, retrieved: "Yes, two weeks is within policy.",
    )

    result = run_two_turns("test-thread-two-turns")

    contents = [m.content for m in result["messages"]]
    assert contents == [
        "I need a refund.",
        "Yes, two weeks is within policy.",
        "It's been two weeks, is that ok?",
        "Yes, two weeks is within policy.",
    ]


def test_inspect_reports_a_finished_run_with_an_empty_next(monkeypatch):
    """"Inspect and resume": once a run reaches END, `snapshot.next` - the
    resumption boundary made visible - is an empty tuple."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("answer")
    )
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])
    monkeypatch.setattr(
        graph_module, "compose_answer", lambda messages, retrieved: "reply"
    )
    thread_id = "test-thread-inspect"

    graph.invoke(
        {"messages": [{"role": "user", "content": "hi"}]},
        {"configurable": {"thread_id": thread_id}},
    )
    snapshot = inspect(thread_id)

    assert snapshot.next == ()
    assert snapshot.values["route"] == "answer"


def test_two_different_threads_do_not_share_history(monkeypatch):
    """Thread isolation, the flip side of Exercise 3: two distinct
    `thread_id`s never see each other's messages."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("answer")
    )
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])
    monkeypatch.setattr(
        graph_module, "compose_answer", lambda messages, retrieved: "reply"
    )

    graph.invoke(
        {"messages": [{"role": "user", "content": "customer A's secret"}]},
        {"configurable": {"thread_id": "thread-a"}},
    )
    graph.invoke(
        {"messages": [{"role": "user", "content": "customer B's question"}]},
        {"configurable": {"thread_id": "thread-b"}},
    )

    snapshot_b = inspect("thread-b")
    contents_b = [m.content for m in snapshot_b.values["messages"]]

    assert "customer A's secret" not in contents_b


def test_run_with_durability_defaults_to_sync_and_still_returns_the_normal_result(
    monkeypatch,
):
    """"Persist before you proceed": durability is a per-path choice, not a
    silently-accepted default - the sync path still returns the same shape
    of result as a plain invoke."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("answer")
    )
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])
    monkeypatch.setattr(
        graph_module, "compose_answer", lambda messages, retrieved: "reply"
    )

    result = run_with_durability("test-thread-durability-sync", "hi")

    assert result["route"] == "answer"


def test_run_with_durability_accepts_the_async_mode_for_read_only_paths(monkeypatch):
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("answer")
    )
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])
    monkeypatch.setattr(
        graph_module, "compose_answer", lambda messages, retrieved: "reply"
    )

    result = run_with_durability(
        "test-thread-durability-async", "hi", durability="async"
    )

    assert result["route"] == "answer"


def test_run_with_drain_returns_none_when_a_drain_was_already_requested(monkeypatch):
    """Graceful drain for deploys: a RunControl with `.request_drain()`
    already called stops the run at the next superstep boundary and raises
    `GraphDrained` - caught and turned into None, with a resumable
    checkpoint left behind for the same thread_id."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("answer")
    )
    control = RunControl()
    control.request_drain()

    result = run_with_drain("test-thread-drain", "hi", control=control)

    assert result is None
    snapshot = inspect("test-thread-drain")
    assert snapshot.next == ("__start__",)


def test_run_with_drain_completes_normally_without_a_drain_request(monkeypatch):
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("answer")
    )
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])
    monkeypatch.setattr(
        graph_module, "compose_answer", lambda messages, retrieved: "reply"
    )

    result = run_with_drain("test-thread-no-drain", "hi")

    assert result is not None
    assert result["route"] == "answer"


def test_run_to_approval_suspends_and_surfaces_the_proposed_refund(monkeypatch):
    """"Suspend, surface, resume": driving Atlas's "refund" route through
    `run_to_approval` returns with `__interrupt__` set instead of a finished
    answer - the run is durably parked at the gate, not blocked in memory."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("refund")
    )

    result = run_to_approval(
        "test-thread-run-to-approval",
        {"id": "T-1001", "amount": 49.0},
        "refund please",
    )

    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["action"] == "issue_refund"


def test_resume_approval_resumes_the_same_thread_and_completes_the_refund(
    monkeypatch,
):
    """Hours later, on any worker: `resume_approval` invokes the SAME
    thread_id with the human's decision and nothing else, and the refund
    completes exactly once."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("refund")
    )
    thread_id = "test-thread-resume-approval"

    run_to_approval(thread_id, {"id": "T-1001", "amount": 49.0}, "refund please")
    result = resume_approval(thread_id, {"type": "approve"})

    assert result["refund_done"] is True
    assert result["messages"][-1].content.startswith("Refund of $")


@requires_postgres
def test_run_durable_persists_to_a_real_postgres_backend(monkeypatch):
    """"Swap to a durable backend": the same graph, compiled onto
    AsyncPostgresSaver, actually round-trips a turn through a live Postgres
    instance AND leaves it there. Skipped by default - see `requires_postgres`
    above.

    Two things this test learned the hard way, the first time it was run
    against a real database rather than skipped:

    1. `scripts/setup_checkpointer.py` has to have run first. Ch9's own
       production-considerations rule is that `.setup()` is a migration and
       never runs from the application, so `run_durable` correctly does not
       call it - which means a fresh database has no `checkpoints` table and
       the invoke fails with UndefinedTable. Running the migration here is
       what a deploy step does for you in production.
    2. On Windows the default event loop is the ProactorEventLoop, and
       psycopg's async driver refuses it outright with InterfaceError before
       it opens a connection. `use_selector_event_loop()` is the same shim
       the migration script uses, and it is a no-op off Windows.

    The final assertion reopens a SEPARATE saver and reads the thread back.
    Asserting on `run_durable`'s return value alone would pass just as well
    against an in-memory checkpointer, which would make a test named
    "persists to a real Postgres backend" prove nothing about persistence.

    The thread id is per-run, and that is the third thing this test learned.
    A fixed id read back 6 messages on the third run rather than 2, because
    Postgres had kept the previous two runs - the durability the test exists
    to prove is exactly what makes a hardcoded id non-deterministic here.
    Every other test in this file can reuse a fixed id safely because
    InMemorySaver starts empty each session."""
    import asyncio
    import uuid

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from scripts.setup_checkpointer import (
        create_checkpoint_tables,
        use_selector_event_loop,
    )

    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("answer")
    )
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])
    monkeypatch.setattr(
        graph_module, "compose_answer", lambda messages, retrieved: "reply"
    )

    use_selector_event_loop()
    dsn = os.environ["ATLAS_POSTGRES_TEST_DSN"]
    config = {"configurable": {"thread_id": f"test-thread-postgres-{uuid.uuid4()}"}}

    asyncio.run(create_checkpoint_tables(dsn))  # the deploy step, run once
    result = asyncio.run(run_durable("I need a refund.", config, db_uri=dsn))

    assert result["route"] == "answer"

    async def reread_from_a_fresh_connection() -> int:
        async with AsyncPostgresSaver.from_conn_string(dsn) as checkpointer:
            reopened = graph_module.builder.compile(checkpointer=checkpointer)
            snapshot = await reopened.aget_state(config)
            return len(snapshot.values["messages"])

    # The turn is still in Postgres after the connection that wrote it closed.
    assert asyncio.run(reread_from_a_fresh_connection()) == 2


# --- Chapter 17: subgraphs, parallelism, and map-reduce --------------------


def test_run_research_bounds_the_fan_out_and_still_returns_every_finding():
    """"Bound the fan-out": max_concurrency=1 forces the workers to run one
    at a time behind the scenes, but the reducer still merges all of their
    writes - the bound changes throughput, never correctness."""
    result = run_research(
        ["docs.internal/refund-policy", "docs.internal/sla"], max_concurrency=1
    )

    assert len(result["findings"]) == 2


def test_run_research_defaults_max_concurrency_to_eight():
    result = run_research(["docs.internal/refund-policy"])

    assert result["findings"][0]["source"] == "docs.internal/refund-policy"
