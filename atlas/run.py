"""Chapter 9, "Persistence and Checkpointing" - scope a conversation with a
`thread_id` and inspect a live run.

`atlas/graph.py` now compiles `graph` onto `InMemorySaver` (dev/test) or, via
`run_durable`, onto `AsyncPostgresSaver` (production). Either way, every
`invoke`/`ainvoke` call must carry a `thread_id` in `config["configurable"]` -
that id *is* the conversation's identity. Reuse one across customers and
their histories merge (a privacy incident); mint a fresh one every turn and
you lose continuity. Derive it from a stable identity, like a ticket id.

Chapter 10, "Durable Execution, Long-Running Workflows, and State
Migration", adds `run_with_durability` - "Persist before you proceed":
durability is a deliberate per-path choice (`sync` for a run that crosses
the checkpoint membrane, `async` for a read-only one, `exit` dev-only) - and
`run_with_drain`, the graceful-drain shape for deploys: pass a `RunControl`,
call `.request_drain()` from a shutdown handler (any thread), and the run
stops at the next superstep boundary with a resumable checkpoint instead of
losing an in-flight conversation to a hard kill.

Chapter 11, "Human-in-the-Loop", adds `run_to_approval` and
`resume_approval` - "Suspend, surface, resume": driving Atlas's "refund"
route now suspends at `atlas/graph.py`'s `approval_gate` instead of
finishing. `run_to_approval` returns with `result["__interrupt__"]` set (the
surfaced proposed refund); `resume_approval` invokes the SAME thread_id with
a `Command(resume=decision)` carrying the human's decision - no new input,
just the answer to the question the gate asked.
"""

from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl
from langgraph.types import Command, StateSnapshot

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


def run_with_durability(
    thread_id: str, message: str, durability: str = "sync"
) -> dict:
    """"Persist before you proceed": durability is a per-path choice, not a
    default to accept. `"sync"` for a run that crosses the checkpoint
    membrane (the completion record must be durable before anything
    downstream depends on it); `"async"` for a read-only path; `"exit"` is a
    development convenience, never a production setting for side-effecting
    work."""
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config,
        durability=durability,
    )


def run_with_drain(
    thread_id: str, message: str, control: RunControl | None = None
) -> dict | None:
    """Graceful drain for deploys: a `RunControl` passed as `control=` can
    have `.request_drain()` called on it from any thread (a SIGTERM handler,
    an orchestrator preStop hook). The run stops at the next superstep
    boundary, not mid-node, and raises `GraphDrained` - caught here and
    turned into `None`, since the checkpoint it left is a normal resumption
    boundary: the same `thread_id` resumes it once the new version is up."""
    config = {"configurable": {"thread_id": thread_id}}
    control = control if control is not None else RunControl()
    try:
        return graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config,
            control=control,
            durability="sync",
        )
    except GraphDrained:
        return None


def run_to_approval(thread_id: str, ticket: dict, message: str) -> dict:
    """"Suspend, surface, resume": drive Atlas up to the approval gate. If
    triage routes to "refund", the run now suspends at `approval_gate`
    instead of crossing the membrane - the returned dict carries
    `result["__interrupt__"]`, the surfaced payload awaiting a decision."""
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(
        {"messages": [{"role": "user", "content": message}], "ticket": ticket},
        config,
    )


def resume_approval(thread_id: str, decision: dict) -> dict:
    """Resume a suspended approval gate on the SAME thread - no new input,
    just the human's decision. `decision` becomes the return value of the
    `interrupt()` call inside `approval_gate`, hours or a restart later, on
    any worker."""
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(Command(resume=decision), config)
