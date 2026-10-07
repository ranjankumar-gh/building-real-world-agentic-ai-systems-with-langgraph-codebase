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
losing an in-flight conversation to a hard kill. `inputs` is the shape of a
refund request: Atlas does not look tickets up, so the support system that
opened the ticket passes it in beside the customer's message, and the
`refund` node reads `id` and `amount` from it. The ticket also names the
customer it belongs to (`customer_id`), which Chapter 13's `recall` and
`remember` read to reach that customer's long-term profile.

Chapter 11, "Human-in-the-Loop", adds `run_to_approval` and
`resume_approval` - "Suspend, surface, resume": driving Atlas's "refund"
route now suspends at `atlas/graph.py`'s `approval_gate` instead of
finishing. `run_to_approval` returns with `result["__interrupt__"]` set (the
surfaced proposed refund); `resume_approval` invokes the SAME thread_id with
a `Command(resume=decision)` carrying the human's decision - no new input,
just the answer to the question the gate asked.

Chapter 14, "Advanced Memory: Extraction, Compaction, and LangMem", adds
`run_and_reflect` - background reflection, off the hot path: the graph
answers, the function returns the result at once, and `atlas/memory.py`'s
`reflect` runs afterwards on `reflection_pool`, writing into the store the
graph was compiled with (`graph.store`). One worker on purpose: reflections
run one at a time, so `compact`'s read-then-write never races itself (the
lost update Chapter 13 warns about). A real deployment keys a durable queue
by customer instead; this pool dies with the process, and a failure shows
up only on the Future, which `_log_failure` logs.

Chapter 17, "Subgraphs, Parallelism, and Map-Reduce", adds `run_research` -
"Bound the fan-out": ten concurrent workers is fine, a hundred is a
rate-limit outage, so `max_concurrency` caps how many fanned-out branches
run at once. Set on `invoke`'s `config`, not on the graph itself, so each
call can take its own bound. `research_runner` compiles the research
builder onto a checkpointer and `run_research` runs on a `thread_id`, so a
failed superstep keeps the workers that finished and `resume_research`
re-runs only the unfinished task. The module-level `research_graph` stays
checkpointer-free for `langgraph.json` and its other importers.
"""

import logging
from concurrent.futures import Future, ThreadPoolExecutor

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl
from langgraph.types import Command, StateSnapshot

from atlas.graph import graph
from atlas.memory import reflect
from atlas.research import research_builder

logger = logging.getLogger(__name__)


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


# A refund request as the support system sends it: the customer's message
# plus the ticket it is about. The id and amount match the seeded backend.
inputs = {
    "messages": [{"role": "user", "content": "I need a refund."}],
    "ticket": {"id": "T-1001", "amount": 49.0, "customer_id": "C-1"},
}


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
    any worker. Put the decider in it (`{"type": "approve", "by": ...}`):
    the gate copies `by` into the `approval` audit record."""
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(Command(resume=decision), config)


# Chapter 17: the same map-reduce, compiled onto a checkpointer, so every
# finished worker's writes are saved and a resume re-runs only the unfinished
# task. InMemorySaver keeps it offline; production uses a durable saver
# (Chapter 9).
research_runner = research_builder.compile(checkpointer=InMemorySaver())


def _research_config(thread_id: str, max_concurrency: int) -> RunnableConfig:
    return {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": max_concurrency,
    }


def run_research(sources: list[str], thread_id: str, max_concurrency: int = 8) -> dict:
    """Run the research fan-out on a fresh thread, at most `max_concurrency`
    workers at once; the rest queue and fill in as slots free. One
    thread_id per run: `findings` is an add channel, so a second run on
    the same thread appends to the first run's findings."""
    config = _research_config(thread_id, max_concurrency)
    return research_runner.invoke({"sources": sources}, config)


def resume_research(thread_id: str, max_concurrency: int = 8) -> dict:
    """Resume a failed run: only the tasks that did not finish run again."""
    return research_runner.invoke(None, _research_config(thread_id, max_concurrency))


reflection_pool = ThreadPoolExecutor(max_workers=1)  # one reflection at a time


def _log_failure(future: Future) -> None:
    if future.exception() is not None:
        logger.error("reflection failed", exc_info=future.exception())


def run_and_reflect(thread_id: str, inputs: dict) -> dict:
    """Answer now, learn afterwards: the reply never waits on extraction."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(inputs, config)
    customer_id = (inputs.get("ticket") or {}).get("customer_id")
    if customer_id is None:
        return result  # no customer on the ticket: nothing to learn, reply intact
    future = reflection_pool.submit(
        reflect, graph.store, customer_id, result["messages"]
    )
    future.add_done_callback(_log_failure)
    return result
