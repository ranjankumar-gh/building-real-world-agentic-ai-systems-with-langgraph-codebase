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
which `triage` resets (with `error` and `retrieved`) at the start of every
question, and
records a `KnowledgeBaseUnavailable` failure in state instead of crashing
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
not yet rewired to call it: `answer` is an injectable seam via
`build_graph(resolve_node=...)` so a middleware-equipped agent can be mounted
there; see Chapter 17's mounting section for the pattern. Until a custom node
is mounted, `answer` calls the atlas.helpers stub for `compose_answer` shape
continuity. See atlas/helpers.py's module docstring.

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
against in tests, rather than inventing a node Atlas does not have.

Chapter 11, "Human-in-the-Loop", places the `approval_gate` node BEFORE the
checkpoint membrane the `refund` node crosses. Triage's "refund" route no
longer goes straight to `refund` - it goes to `approval_gate` first (see the
`path_map` on `triage`'s conditional edges below), and `approval_gate` is
the only node that decides whether `refund` ever runs. It calls
`interrupt()` with the proposed action, which suspends the run to the
checkpointer, and resumes with the human's decision as the return value of
that same call. Approve and a re-validated edit route onward to `refund`
via `Command(goto=...)`; reject routes to `escalate`. Because a resumed node
re-runs from the top (see the chapter's "gotcha" callout), `approval_gate`
does nothing but interrupt and route - no side effect lives here, the same
membrane discipline Chapter 10 established for `refund` itself.

Chapter 13, "Short-Term vs Long-Term Memory", adds `remember` and `recall` -
the node-facing side of the cross-thread store (`atlas/memory.py` holds the
store's own shape: `profile_ns`, `relevant_memories`, the dev/prod backend
swap). Both reach the store through `runtime.store`, the same `Runtime`
handle every node and middleware already receives - `remember` persists a
durable customer fact that outlives the thread; `recall` reads it back at
the start of a fresh one. Like `retrieve_async` and `triage_with_command`
before them, they are defined and unit-tested here but not wired into
`builder` below: the chapter's point is the mechanism itself (checkpointer
vs. store, thread-scoped vs. namespace-scoped), not a specific place in
Atlas's routing topology to call it - that integration is left to
Chapter 14, once extraction decides *what* is worth remembering.

Chapter 17, "Subgraphs, Parallelism, and Map-Reduce", mounts
`atlas/research.py`'s compiled `research_graph` as a node. `AtlasState` and
`ResearchState` share no keys (Atlas has no native notion of "sources"), so
per "Subgraphs: the research pipeline as a reusable unit" this cannot be
mounted directly - `research` below is the wrapping node the chapter shows
for exactly that case, adapting in with `derive_sources` and back out with
`summarize_findings`. `derive_sources` reads from `ticket`, the same
free-form per-request dict `approval_gate`/`refund` already read
ticket-scoped data from. Like `remember`/`recall` before it, `research` is
added to `builder` (the chapter's own code calls `add_node`) but is not
wired into `route_from_triage`'s edges - the chapter names the node, not a
place in the routing topology to reach it from.

Chapter 21, "Evaluation and Testing", factors the builder-assembly wiring out
into `_make_builder` and adds `build_graph(model=...)` on top of it - a
small, additive refactor so `atlas/replay.py`'s checkpoint-replay fixture can
substitute Chapter 1's `ScriptedModel` for the triage step's decision source
without touching `builder`/`graph` (still built exactly as before, via
`_make_builder(triage)`), which `tests/test_graph.py` and every other
chapter's tests continue to depend on unchanged."""

import asyncio
from collections.abc import Callable
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Overwrite, RetryPolicy, TimeoutPolicy, interrupt

from atlas.effects import RefundError, charge_refund, idempotency_key
from atlas.helpers import compose_answer, search_kb
from atlas.memory import profile_ns
from atlas.research import research_graph
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
    """Chapter 6, "The bounded retry": triage runs once at the start of every
    question, so it is where the per-question loop guard starts over. Without
    the reset, a checkpointed thread (Chapter 9 onward) carries
    `retrieve_attempts`, `error`, and `retrieved` into the next question:
    that question starts at the cap and gets no retries, a stale error
    escalates it outright, and an empty search "answers" on the previous
    question's documents. `retrieved` has a reducer (dedup_by_id), which
    would merge an empty list into the old one, so its reset is an
    `Overwrite` - one per channel per superstep, and triage is the only
    writer in its superstep."""
    decision = classify(state["messages"])  # typed; route is already constrained
    route = decision.route if decision.route in ALLOWED_ROUTES else "escalate"
    # a new question gets a fresh guard: no attempts yet, no recorded failure
    return {
        "route": route,
        "retrieve_attempts": 0,
        "error": None,
        "retrieved": Overwrite([]),  # bypass dedup_by_id: start empty
    }


def triage_with_command(
    state: AtlasState,
) -> Command[Literal["answer", "retrieve", "escalate", "refund"]]:
    """The `Command` alternative to `triage` + `route_from_triage`: update
    state and name the next node in one move, for when the two are genuinely
    the same decision. Not wired into `builder` - `triage` above stays the
    default, since it keeps routing visible on a separate edge."""
    decision = classify(state["messages"])
    route = decision.route if decision.route in ALLOWED_ROUTES else "escalate"
    update = {
        "route": route,
        "retrieve_attempts": 0,
        "error": None,
        "retrieved": Overwrite([]),
    }
    return Command(update=update, goto=route)


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
    # compose_answer returns a str, and add_messages coerces a bare str into a
    # HumanMessage - wrap it so the reply is recorded as the assistant's turn.
    return {"messages": [AIMessage(content=reply)]}


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


def approval_gate(state: AtlasState) -> Command:
    """Chapter 11: the approval gate, sitting BEFORE the membrane `refund`
    crosses. `interrupt()` suspends the run to the checkpointer and surfaces
    the proposed refund; the human's decision comes back as `decision`, the
    return value of that same `interrupt()` call, once someone resumes the
    thread with `Command(resume=...)`.

    Approve and a re-validated edit route onward to `refund`; reject routes
    to `escalate`. This node does nothing else - on resume, LangGraph
    re-runs it from the top, so any side effect placed before `interrupt()`
    would fire again on every resume. See "A resumed node re-runs from the
    top"."""
    ticket = state["ticket"]
    decision = interrupt(
        {
            "action": "issue_refund",
            "ticket_id": ticket["id"],
            "amount": ticket["amount"],
        }
    )
    if decision["type"] == "approve":
        return Command(goto="refund")
    if decision["type"] == "edit":
        amount = decision["amount"]
        if not 0 < amount <= ticket["amount"]:  # re-validate the human's edit
            return Command(
                update={"error": f"edited amount {amount} out of policy"},
                goto="escalate",
            )
        return Command(
            update={"ticket": {**ticket, "amount": amount}},
            goto="refund",
        )
    if decision["type"] == "reject":
        return Command(
            update={"error": f"refund rejected: {decision.get('reason', '')}"},
            goto="escalate",
        )
    raise ValueError(f"unknown decision: {decision['type']}")


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


def remember(state: AtlasState, runtime: Runtime) -> dict:
    """Chapter 13: persist a durable fact about the customer - crosses no
    membrane, but outlives the thread. Not wired into `builder` below (see
    the module docstring's Chapter 13 paragraph) - unit-tested directly in
    tests/test_graph.py against a `Runtime` built on `InMemoryStore`."""
    cid = state["ticket"]["customer_id"]
    runtime.store.put(profile_ns(cid), "plan", {"tier": "enterprise"})
    return {}


def recall(state: AtlasState, runtime: Runtime) -> dict:
    """Chapter 13: load the customer's profile into state at the start of a
    thread, so a returning customer is not a stranger. See `remember`
    above."""
    cid = state["ticket"]["customer_id"]
    item = runtime.store.get(profile_ns(cid), "plan")
    plan = item.value["tier"] if item else "unknown"
    return {"customer_plan": plan}


def derive_sources(state: AtlasState) -> list[str]:
    """Chapter 17: the parent-to-subgraph input adapter. `AtlasState` has no
    native "sources" concept, so pull the list from `ticket` - the same
    free-form, per-request dict `approval_gate`/`refund` already read
    ticket-scoped fields from."""
    ticket = state.get("ticket") or {}
    return ticket.get("sources", [])


def summarize_findings(findings: list[dict]) -> AIMessage:
    """Chapter 17: the subgraph-to-parent output adapter - fold N findings
    (each a result or a partial-failure error) into one message for the
    transcript."""
    lines = [
        f"- {f['source']}: {f['result']}"
        if "result" in f
        else f"- {f['source']}: unavailable ({f['error']})"
        for f in findings
    ]
    return AIMessage("Research findings:\n" + "\n".join(lines))


def research(state: AtlasState) -> dict:
    """Adapt Atlas state to the research subgraph and back. `research_graph`
    is the Chapter 17 map-reduce pipeline - `plan` -> `Send`-fanned
    `research_worker`s -> `END` - independently testable and internally
    parallel; this node is the only place Atlas's own state touches it."""
    out = research_graph.invoke({"sources": derive_sources(state)})
    return {"messages": [summarize_findings(out["findings"])]}


def _make_builder(
    triage_node, resolve_node: Callable[[AtlasState], dict] = answer
) -> StateGraph:
    """Chapter 21, "Testing non-determinism": the wiring shared by the
    module-level `builder` below and every fixture graph `build_graph`
    constructs - the exact same Chapter 6-17 topology, parameterized only on
    which triage callable is wired in for the "triage" node. Chapter 21 adds
    the `resolve_node` seam for mounting a middleware-equipped agent in the
    answering position; see Chapter 17's mounting section for the pattern."""
    b = StateGraph(AtlasState)
    b.add_node(
        "triage",
        triage_node,
        # Chapter 4, "A first look at retries": triage makes the first live
        # model call in the graph, and a model call is a read - safe to
        # retry, and previously not retried at all. NOTE the absent
        # retry_on: the default predicate (langgraph.types.default_retry_on)
        # retries a provider rate-limit error and declines to retry a
        # ValueError. Naming retry_on=(ConnectionError,) here would REPLACE
        # that judgement with a whitelist of one, and a 429 arrives as
        # anthropic.RateLimitError, which is not a ConnectionError.
        retry_policy=RetryPolicy(max_attempts=3),
    )
    b.add_node(
        "retrieve",
        retrieve,
        # First look at durable execution (full treatment: Chapter 10). Safe
        # here because retrieve is a read; do not add a retry_policy to a
        # side-effecting node without the discipline Chapter 10 covers. The
        # chapter first shows retry_on=(ConnectionError,) as the narrowing to
        # avoid, then drops it: the default predicate also retries a
        # knowledge-base 5xx and a provider SDK error, which that whitelist
        # of one would decline.
        retry_policy=RetryPolicy(max_attempts=3),
    )
    b.add_node("answer", resolve_node)
    b.add_node("escalate", escalate)
    # Chapter 11: the approval gate - no retry_policy, no side effect. It
    # only interrupts and routes; retrying a suspended interrupt is not the
    # same kind of retry Chapter 10 earned for refund.
    b.add_node("approval_gate", approval_gate)
    b.add_node(
        "refund",
        refund,
        # Chapter 10: safe now that refund is idempotent (earns the retry
        # Chapter 4 forbade on side-effecting nodes). error_handler runs only
        # after retries are exhausted and compensates by routing to escalate.
        retry_policy=RetryPolicy(max_attempts=3, retry_on=(RefundError,)),
        error_handler=refund_failed,
    )
    # Chapter 17: the research subgraph, mounted like any other node -
    # reusable, internally parallel, independently testable. Not wired into
    # route_from_triage below; see the module docstring's Chapter 17
    # paragraph.
    b.add_node("research", research)

    b.add_edge(START, "triage")
    # Chapter 11: triage's "refund" route now lands on the approval gate, not
    # on refund directly - the gate decides whether refund ever runs.
    # ALLOWED_ROUTES and route_from_triage are unchanged; only the physical
    # destination for the "refund" route moves behind the gate.
    b.add_conditional_edges(
        "triage",
        route_from_triage,
        {
            "answer": "answer",
            "retrieve": "retrieve",
            "escalate": "escalate",
            "refund": "approval_gate",
        },
    )
    b.add_conditional_edges("retrieve", route_after_retrieve)  # cycle + exit
    b.add_edge("answer", END)
    b.add_edge("escalate", END)
    b.add_edge("refund", END)
    # approval_gate has no static outgoing edge - it always returns a
    # Command with goto="refund" or goto="escalate", the same
    # dynamic-routing shape triage_with_command uses above.
    return b


def build_graph(
    model=None, resolve_node: Callable[[AtlasState], dict] | None = None
) -> Pregel:
    """Chapter 21, "Testing non-determinism: replaying a checkpoint": factor
    the model out to a parameter, the way `create_agent` already takes one,
    instead of the module-level `classify` every node closes over. `model=
    None` reconstructs the exact same graph as the module-level `graph`
    below (the real, `create_agent`-backed `classify`); passing Chapter 1's
    `atlas.breaks.ScriptedModel` swaps ONLY the triage step's decision
    source - `triage_node` below reads the scripted model's next response
    directly, the same one-call-per-turn contract `ScriptedModel` already
    provides - so a replay fixture can force a specific route deterministically,
    with zero real model calls, without touching `builder`/`graph` any other
    test or CI depends on."""
    if model is None:
        triage_node = triage
    else:

        def triage_node(state: AtlasState) -> dict:
            ai = model.invoke(state["messages"])
            route = ai.content if ai.content in ALLOWED_ROUTES else "escalate"
            return {
                "route": route,
                "retrieve_attempts": 0,
                "error": None,
                "retrieved": Overwrite([]),
            }

    return _make_builder(
        triage_node, resolve_node=resolve_node or answer
    ).compile(
        checkpointer=InMemorySaver(), store=InMemoryStore()
    )


builder = _make_builder(triage)

# Chapter 9: compiled onto a checkpointer, so every superstep is saved. The
# dev/test default - RAM-backed, gone on restart, but the right tool for
# tests: it exercises the real checkpointing path with no external service.
#
# Chapter 13 configures the store BESIDE the checkpointer, not instead of it:
# they do different jobs (in-thread state vs cross-thread facts). Passing it
# here is what makes `runtime.store` non-None inside `remember`/`recall`.
graph = builder.compile(checkpointer=InMemorySaver(), store=InMemoryStore())

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
    # Imported HERE, not at module scope. atlas/memory.py's build_prod_store
    # already does this for the same class of dependency, and graph.py was
    # the odd one out. The cost of the module-level version was concrete:
    # importing atlas.graph required psycopg's binary extra even on paths
    # that never touch Postgres, so `langgraph dev` - the Docker-free server
    # this book points readers at - could not load the graph at all. It
    # failed with `no pq wrapper available`, which is precisely the import
    # failure Chapter 9 warns about two paragraphs before it prints this
    # function. A production-only dependency should not be a hard
    # requirement for loading the module.
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(db_uri) as checkpointer:
        durable_graph = builder.compile(
            checkpointer=checkpointer, store=InMemoryStore()
        )
        return await durable_graph.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config,
        )
