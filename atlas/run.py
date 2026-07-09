"""Chapter 9, "Persistence and Checkpointing" - scope a conversation with a
`thread_id` and inspect a live run.

`atlas/graph.py` now compiles `graph` onto `InMemorySaver` (dev/test) or, via
`run_durable`, onto `AsyncPostgresSaver` (production). Either way, every
`invoke`/`ainvoke` call must carry a `thread_id` in `config["configurable"]` -
that id *is* the conversation's identity. Reuse one across customers and
their histories merge (a privacy incident); mint a fresh one every turn and
you lose continuity. Derive it from a stable identity, like a ticket id.
"""

from langgraph.types import StateSnapshot

from atlas.graph import graph


def run_two_turns(thread_id: str) -> dict:
    """"Scope the conversation with a thread_id": two invocations on the
    SAME thread. The second call does not resend the first turn - the
    checkpointer loads the thread's saved state, the new message merges in
    through the `add_messages` reducer (Chapter 5), and the run continues."""
    config = {"configurable": {"thread_id": thread_id}}

    graph.invoke(
        {"messages": [{"role": "user", "content": "I need a refund."}]},
        config,
    )

    # A later turn on the SAME thread - the history is restored automatically.
    return graph.invoke(
        {"messages": [{"role": "user", "content": "It's been two weeks, is that ok?"}]},
        config,
    )


def inspect(thread_id: str) -> StateSnapshot:
    """"Inspect and resume": `get_state` returns the current snapshot for a
    thread - `snapshot.values` is the AtlasState dict (messages, route,
    retrieved, ...); `snapshot.next` names the pending node(s), and an empty
    tuple means the run finished. `snapshot.next` is the resumption boundary
    made visible: a new process, same thread_id, picks up from here."""
    config = {"configurable": {"thread_id": thread_id}}
    return graph.get_state(config)
