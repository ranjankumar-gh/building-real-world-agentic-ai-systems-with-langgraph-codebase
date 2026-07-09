"""The Chapter 3 whiteboard, wired into a real, running StateGraph.

See Chapter 4, "StateGraph, Nodes, and Edges". This module supersedes
atlas/graph_sketch.py's role as Atlas's *executable* graph - graph_sketch.py
stays in the repo unchanged as the Chapter 3 node-SHAPE artifact (no
StateGraph, no compile()); this is the transcription of that whiteboard into
something `compile()` validates and the runtime actually runs.

Chapter 5 moved the state schema out to atlas/state.py and gave every channel
its own deliberately chosen reducer - `AtlasState` is imported from there now,
not defined inline.

Chapter 6, "Conditional Edges and Dynamic Control Flow", turns the linear
START -> triage -> retrieve -> answer -> END chain into the Figure 3.1
branching topology: `triage` validates the model's proposed route against
`ALLOWED_ROUTES` (off-menu -> "escalate", the routing boundary from that
chapter), `add_conditional_edges` replaces the fixed triage->retrieve and
retrieve->answer edges, and a new `escalate` node gives the graph a graceful
exit. `retrieve` is now a bounded retry cycle guarded by the explicit
`retrieve_attempts` state counter - not by LangGraph's `recursion_limit` -
and records a `KnowledgeBaseUnavailable` failure in state instead of crashing
or answering on top of it. `triage_with_command` is the chapter's `Command`
alternative: update state and route in one move. It is intentionally not
wired into `builder` below, the same way Chapter 4's `retrieve_async` is
defined but unused - the chapter's own guidance is to default to conditional
edges and reach for `Command` only when the update and the route are
genuinely one decision.

The node bodies originally called stubs in atlas.helpers
(classify/search_kb/compose_answer), which raised NotImplementedError.

Chapter 7, "Tools, Models, MCP, and create_agent", replaces `classify` with
the real, structured-output version in `atlas/triage.py` - `triage` below
now reads a validated `TriageResult` instead of parsing a raw string, but
still validates `.route` against `ALLOWED_ROUTES` (the routing boundary
holds; structured output narrows the input, it does not dissolve the
boundary). `search_kb` also went real in that chapter, as a `@tool` in
`atlas/tools.py` - along with `KnowledgeBaseUnavailable`, imported from
there now instead of atlas.helpers - but `retrieve` and `answer` below are
not yet rewired to call it: that integration (folding a tool-calling agent
into this graph) is deferred to a later chapter, so `retrieve`/`answer`
still call the atlas.helpers stubs for `search_kb`/`compose_answer` shape
continuity until then. See atlas/helpers.py's module docstring.

Chapter 9, "Persistence and Checkpointing", compiles `graph` onto a
checkpointer so state survives past a single `invoke` call. `InMemorySaver`
is the dev/test backend - RAM only, gone on process restart, but it
exercises the exact checkpointing code path with no external dependency.
With a checkpointer attached, every `invoke`/`ainvoke` now requires a
`thread_id` in `config["configurable"]` - see atlas/run.py for the
thread-scoped call shape and `get_state` inspection. `run_durable` below is
the production seam: the same graph, compiled for the lifetime of one call
onto `AsyncPostgresSaver` instead, so checkpoints outlive the process. Select
between them by environment behind one factory, per the chapter's "one seam
for dev and prod" - the graph-building code itself never branches on which
backend is live.

Chapter 10, "Durable Execution, Long-Running Workflows, and State
Migration", is Atlas's first crossing of the checkpoint membrane: `refund`
is a real side-effecting node, so it earns the `retry_policy` Chapter 4
forbade side-effecting nodes ("don't retry side-effecting nodes" meant "earn
the retry via idempotency first," not "never retry") plus an `error_handler`
(`refund_failed`) that compensates by routing to `escalate` once retries are
exhausted. `ALLOWED_ROUTES` grows to include `"refund"`. The idempotency key
and the idempotent operation itself live in `atlas/effects.py`, isolated
from the graph so the side-effecting code is testable on its own -
`refund_already_done` is the additive-migration-safe read
(`state.get("refund_done", False)`) that lets a checkpoint written before
this chapter keep resuming without raising. `TimeoutPolicy` is async-only
(sync-node timeouts are rejected at compile); Atlas has no async node in its
compiled topology yet, so `retrieve_async` - already defined-but-unused
since Chapter 6 - is what the chapter's `TimeoutPolicy` example is exercised
against in tests, rather than inventing a node Atlas does not have."""

import asyncio
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, TimeoutPolicy

from atlas.effects import RefundError, charge_refund, idempotency_key
from atlas.helpers import compose_answer, search_kb
from atlas.state import AtlasState
from atlas.tools import KnowledgeBaseUnavailable
from atlas.triage import classify

ALLOWED_ROUTES = ("answer", "retrieve", "escalate", "refund")
MAX_RETRIEVE_ATTEMPTS = 3

# Chapter 10's TimeoutPolicy example ("research", an async node bounded by a
# hard wall clock plus an idle timeout). Atlas's real async node candidate is
# retrieve_async, below - see tests/test_graph.py for the async-only
# behavior this policy demonstrates.
RETRIEVE_TIMEOUT = TimeoutPolicy(run_timeout=30.0, idle_timeout=10.0)


def triage(state: AtlasState) -> dict:
    decision = classify(state["messages"])  # typed; route is already constrained
    route = decision.route if decision.route in ALLOWED_ROUTES else "escalate"
    return {"route": route}


def triage_with_command(
    state: AtlasState,
) -> Command[Literal["answer", "retrieve", "escalate", "refund"]]:
    """The `Command` alternative to `triage` + `route_from_triage`: update
    state and name the next node in one move, for when the two are genuinely
    the same decision. Not wired into `builder` - `triage` above stays the
    default, since it keeps routing visible on a separate edge."""
    decision = classify(state["messages"])
    route = decision.route if decision.route in ALLOWED_ROUTES else "escalate"
    return Command(update={"route": route}, goto=route)


def route_from_triage(
    state: AtlasState,
) -> Literal["answer", "retrieve", "escalate", "refund"]:
    return state["route"]


def retrieve(state: AtlasState) -> dict:
    """Plain `def`, on purpose. `retrieve` does network I/O (it hits the
    knowledge base); under `ainvoke` the runtime offloads a sync node like
    this one to a worker thread, so a blocking call inside it never touches
    the event loop. See "Making it correct under load" - marking this
    `async def` while still calling `search_kb` directly is the mistake
    behind the chapter's opening incident (every concurrent run stalls
    behind one blocking request).

    Chapter 6 adds the bounded-retry counter and the failure path: a
    `KnowledgeBaseUnavailable` is caught and recorded in `error` rather than
    left to crash the run, and every successful call increments
    `retrieve_attempts` so `route_after_retrieve` can cap the retry loop."""
    try:
        hits = search_kb(state["messages"])
    except KnowledgeBaseUnavailable as exc:
        return {"error": str(exc)}  # record the failure - do not pretend it worked
    attempts = state.get("retrieve_attempts", 0) + 1
    return {"retrieved": hits, "retrieve_attempts": attempts}


async def retrieve_async(state: AtlasState) -> dict:
    """The correct shape if `retrieve` genuinely needed `async` (say, to
    `await` several I/O calls): offload the blocking call explicitly with
    `asyncio.to_thread` instead of calling it directly on the event loop.
    Not wired into `builder` below - Atlas's `retrieve` has no other await
    work yet, so the plain `def` version above is the right default."""
    hits = await asyncio.to_thread(search_kb, state["messages"])
    return {"retrieved": hits}


def route_after_retrieve(
    state: AtlasState,
) -> Literal["answer", "retrieve", "escalate"]:
    if state.get("error"):
        return "escalate"  # tool failure -> human, not a fake answer
    if state["retrieved"]:
        return "answer"  # got results -> answer
    if state["retrieve_attempts"] >= MAX_RETRIEVE_ATTEMPTS:
        return "escalate"  # gave up -> human, gracefully
    return "retrieve"  # bounded retry


def answer(state: AtlasState) -> dict:
    reply = compose_answer(state["messages"], state["retrieved"])
    return {"messages": [reply]}


def escalate(state: AtlasState) -> dict:
    """The graceful exit the retry loop and the model's off-menu routes both
    fall back to. Same shape as the Chapter 3 whiteboard's escalate node
    (atlas/graph_sketch.py) - Chapter 6 is what finally wires it in."""
    return {"ticket": {"status": "escalated"}}


def refund_already_done(state: AtlasState) -> bool:
    """Chapter 10, "State migration without downtime": the additive-
    migration-safe read. A checkpoint written before this chapter has no
    `refund_done` key at all - `state["refund_done"]` would raise KeyError
    resuming one of those; `.get` with a default does not."""
    return state.get("refund_done", False)  # resumes old checkpoints safely


def refund(state: AtlasState, config: RunnableConfig) -> dict:
    """Atlas's first crossing of the checkpoint membrane. The idempotency
    key is derived from durable state (`thread_id` + `ticket_id`), so a
    retry or a resume recomputes the SAME key and `charge_refund` dedupes at
    the backend instead of charging twice."""
    ticket_id = state["ticket"]["id"]
    thread_id = config["configurable"]["thread_id"]
    key = idempotency_key(thread_id, ticket_id)
    result = charge_refund(key, ticket_id)  # safe to re-run: keyed
    return {"messages": [AIMessage(result)], "refund_done": True}


def refund_failed(state: AtlasState) -> Command:
    # Compensation routes to a human and records the failure. If this
    # reversed a prior action, that reversal would need its own
    # idempotency key - compensation is a side effect too.
    return Command(
        update={"error": "refund failed after retries; needs manual review"},
        goto="escalate",
    )


builder = StateGraph(AtlasState)
builder.add_node("triage", triage)
builder.add_node(
    "retrieve",
    retrieve,
    # First look at durable execution (full treatment: Chapter 10). Safe here
    # because retrieve is a read; do not add a retry_policy to a
    # side-effecting node without the discipline Chapter 10 covers.
    retry_policy=RetryPolicy(max_attempts=3, retry_on=(ConnectionError,)),
)
builder.add_node("answer", answer)
builder.add_node("escalate", escalate)
builder.add_node(
    "refund",
    refund,
    # Chapter 10: safe now that refund is idempotent (earns the retry
    # Chapter 4 forbade on side-effecting nodes). error_handler runs only
    # after retries are exhausted and compensates by routing to escalate.
    retry_policy=RetryPolicy(max_attempts=3, retry_on=(RefundError,)),
    error_handler=refund_failed,
)

builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", route_from_triage)  # the branch
builder.add_conditional_edges("retrieve", route_after_retrieve)  # the cycle + its exit
builder.add_edge("answer", END)
builder.add_edge("escalate", END)
builder.add_edge("refund", END)

# Chapter 9: compiled onto a checkpointer, so every superstep is saved. The
# dev/test default - RAM-backed, gone on restart, but the right tool for
# tests: it exercises the real checkpointing path with no external service.
graph = builder.compile(checkpointer=InMemorySaver())

DB_URI = "postgresql://atlas:atlas@localhost:5432/atlas"


async def run_durable(message: str, config: dict, db_uri: str = DB_URI) -> AtlasState:
    """The production seam from "Swap to a durable backend": compile this
    same `builder` onto `AsyncPostgresSaver` instead of `InMemorySaver`, so
    checkpoints outlive the process. The Postgres instance is a seeded local
    service in the companion repo - see README.md for how to point `db_uri`
    at it; no cloud account is needed.

    `AsyncPostgresSaver.from_conn_string` is an async context manager, so the
    compiled graph (and its checkpointer connection) only lives for the
    duration of this call - a real deployment keeps that context open for
    the life of the process instead of opening and closing it per call."""
    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
        durable_graph = builder.compile(checkpointer=checkpointer)
        return await durable_graph.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config,
        )
