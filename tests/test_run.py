"""Chapter 9, "Persistence and Checkpointing" - atlas/graph.py's checkpointer
and atlas/run.py's thread-scoped call shape.

`graph` now compiles onto `InMemorySaver` (RAM-backed, dev/test only), so
every `invoke` needs a `thread_id`. `run_durable` compiles the same builder
onto `AsyncPostgresSaver` instead - that needs a real, reachable Postgres
instance, so its test is skip-guarded behind `ATLAS_POSTGRES_TEST_DSN` and
skips cleanly without one; it is not required for the rest of the suite to
pass. See the README for how to point it at the seeded local Postgres
service.
"""

import os

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from atlas import graph as graph_module
from atlas.graph import graph, run_durable
from atlas.run import inspect, run_two_turns

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
        lambda messages, retrieved: {
            "role": "assistant",
            "content": "Yes, two weeks is within policy.",
        },
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


@requires_postgres
def test_run_durable_persists_to_a_real_postgres_backend(monkeypatch):
    """"Swap to a durable backend": the same graph, compiled onto
    AsyncPostgresSaver, actually round-trips a turn through a live Postgres
    instance. Skipped by default - see `requires_postgres` above."""
    import asyncio

    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("answer")
    )
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])
    monkeypatch.setattr(
        graph_module, "compose_answer", lambda messages, retrieved: "reply"
    )

    dsn = os.environ["ATLAS_POSTGRES_TEST_DSN"]
    config = {"configurable": {"thread_id": "test-thread-postgres"}}

    result = asyncio.run(run_durable("I need a refund.", config, db_uri=dsn))

    assert result["route"] == "answer"
