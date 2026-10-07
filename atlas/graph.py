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
START -> triage -> retrieve -> answer -> END chain into the branching
topology of Chapter 3's "The Atlas graph topology" figure: `triage`
validates the model's proposed route against `ALLOWED_ROUTES` (off-menu ->
"escalate", the routing boundary from that chapter), `add_conditional_edges`
replaces the fixed triage->retrieve and retrieve->answer edges, and a new
`escalate` node gives the graph a graceful exit. `retrieve` is now a bounded
retry cycle guarded by the explicit `retrieve_attempts` state counter - not
by LangGraph's `recursion_limit` - which `triage` resets (with `error` and
`retrieved`) at the start of every question, and records a
`KnowledgeBaseUnavailable` failure in state instead of crashing or answering
on top of it. `triage_with_command` is the chapter's `Command` alternative:
update state and route in one move. It is intentionally not wired into
`builder` below, the same way Chapter 4's `retrieve_async` is defined but
unused - the chapter's own guidance is to default to conditional edges and
reach for `Command` only when the update and the route are genuinely one
decision.

The node bodies originally called stubs in atlas.helpers
(classify/search_kb/compose_answer), which raised NotImplementedError.

Chapter 7, "Tools, Models, MCP, and create_agent", replaces `classify` with
the real, structured-output version in `atlas/triage.py` - `triage` below
now reads a validated `TriageResult` instead of parsing a raw string, but
still validates `.route` against `ALLOWED_ROUTES` (the routing boundary
holds; structured output narrows the input, it does not dissolve the
boundary). `search_kb` also went real in that chapter, as a `@tool` in
`atlas/tools.py` - along with `KnowledgeBaseUnavailable`, imported from
there now instead of atlas.helpers - but `retrieve` and `answer` below still
call the atlas.helpers functions, not the tool directly. At the chapter tags
those functions are stubs; in the finished repo they are thin, model-free
adapters onto the Chapter 7 tool (see atlas/helpers.py's module docstring), so
the retrieve/answer path runs end to end. `answer` is also an injectable seam
via `build_graph(resolve_node=...)` so a middleware-equipped agent can be
mounted there; see Chapter 17's mounting section for the pattern.

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
(`refund_failed`) that compensates by routing to `escalate` once the retry
policy stops (attempts exhausted, or an exception `retry_on` does not
cover). `ALLOWED_ROUTES` grows to include `"refund"`. The idempotency key
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
via `Command(goto=...)`; reject routes to `escalate`. Each routed decision
writes the `approval` audit record (decision, decider, timestamp), and an
edit's amount replaces the ticket's, which `refund` then charges. Because a resumed node
re-runs from the top (see the chapter's "gotcha" callout), `approval_gate`
does nothing but interrupt and route - no side effect lives here, the same
membrane discipline Chapter 10 established for `refund` itself.

Chapter 12, "Context Engineering", puts the graph's model call and its
documents on the budget. `triage` hands `classify` the view
`atlas/context.py`'s `trim_history` cuts to `BUDGET.history`, so however
long the thread grows, the classifier sees only the recent turns that fit
the history slice; `state["messages"]` itself keeps every turn.
(`triage_with_command` takes the same view.) `retrieve` passes the search
results through `select_docs` with `BUDGET.retrieved` before it writes
`retrieved`, so the best-scored documents that fit the slice are all that
`route_after_retrieve` and `answer` ever see. An article larger than the
whole slice leaves the list empty, which is Chapter 6's "nothing usable"
path: retry, then escalate - never a reply that found nothing. Because the
cap lives in `retrieve`, `retrieved` is capped whatever node is mounted at
"answer" through `build_graph` (Chapter 17's pattern); `atlas/resolve.py`'s
adapter hands that mounted agent the capped documents and the profile as
reference text appended to its system message, and the agent's own
`ContextBudget` bounds its history.

Chapter 13, "Short-Term vs Long-Term Memory", adds `recall` and `remember` -
the node-facing side of the cross-thread store (`atlas/memory.py` holds the
store's own shape: `profile_ns`, `relevant_memories`, the dev/prod backend
swap) - and wires both into Atlas. Both reach the store through
`runtime.store`, the same `Runtime` handle every node already receives.
`recall` runs first on every turn (START -> recall -> triage): it loads the
customer's profile entries most relevant to the new question into
`customer_profile` before anything reasons about the turn, and `answer`
passes that profile to `compose_answer`, which names the last issue. `remember` runs
after `answer` (answer -> remember -> END), the one path where Atlas
resolved the question itself: it records the issue this ticket raised under
the profile's `last_issue` key, so the customer's next thread starts with
it. The escalate and refund paths end without it: `escalate` overwrites
`ticket` (dropping `customer_id`), and a refund is already recorded by the
payment backend and the Chapter 11 approval record. The customer comes from
`ticket["customer_id"]`, which the support system's input carries (see
atlas/run.py's `inputs`); a run with no customer on its ticket (Chapter 9's
two-turn example sends none) skips both nodes' store calls.

Chapter 14, "Advanced Memory: Extraction, Compaction, and LangMem", adds no
node: reflection runs after the graph returns, off the hot path
(atlas/run.py's `run_and_reflect`), and writes extracted facts into the
same profile namespace, so `recall` above loads them on the next thread.

Chapter 17, "Subgraphs, Parallelism, and Map-Reduce", shows how a compiled
subgraph is mounted, using `atlas/research.py`'s `research_graph`.
`ResearchState` shares `messages` with `AtlasState`, but the key the
map-reduce reads, `sources`, does not exist here (Atlas has no native notion
of one), so a direct mount would hand the subgraph nothing to work on.
`research` below is the wrapping node the chapter shows for that case,
adapting in with `derive_sources` (from `ticket`, the same free-form
per-request dict `approval_gate`/`refund` read) and back out with
`summarize_findings`. Like `triage_with_command`, it is defined but not added
to the builder: no triage route leads to research, and an unreachable node
is dead weight. Research stays its own compiled graph - `atlas/run.py`'s
`run_research` runs it directly, compiled onto a checkpointer.

Chapter 17 also factors the builder-assembly wiring into `_make_builder` and
adds `build_graph(resolve_node=...)`, the seam `atlas/resolve.py` uses to
mount a middleware-equipped agent in the "answer" position.

Chapter 21, "Evaluation and Testing", gives `build_graph`'s `model=`
parameter its use: `atlas/replay.py`'s checkpoint-replay fixture substitutes
Chapter 1's `ScriptedModel` for the triage step's decision source without
touching `builder`/`graph` (still built exactly as before, via
`_make_builder(triage)`), which `tests/test_graph.py` and every other
chapter's tests continue to depend on unchanged.

Chapter 22, "Deployment and Scaling": the Agent Server supplies the
checkpointer and the store, and `langgraph dev` refuses a graph compiled with
its own, so `atlas/deploy/server.py` (the module `langgraph.json` serves)
compiles `_make_builder`'s wiring with neither. Every other caller,
`atlas/resolve.py`'s in-process `build_resolved_graph` included, keeps
`InMemorySaver`/`InMemoryStore`.

Chapter 23, "A durable audit log, deliberately separate from the trace",
mounts `make_approval_gate(served)` in the "approval_gate" position and
`make_authorized_refund(served)` in the "refund" position. The gate is
Chapter 11's `approval_gate`, unchanged, plus an approval record that carries
the amount and is appended to the audit namespace (`atlas/audit.py`'s
`record_approval`), where an erasure keeps it, rather than only into state,
which an erasure deletes. The refund authorizes itself immediately before
the charge, because a caller's `Command(goto="refund")` can reach it without
the gate: it charges only an approved amount, through Chapter 10's
`refund`. The served build (`served=True`, `atlas/deploy/server.py`) fails
closed: at both nodes the approver is the run's authenticated identity, never
the payload's `by`, and must hold an approver role."""

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Overwrite, RetryPolicy, TimeoutPolicy, interrupt

from atlas.audit import approval_on_record, charged_rows, record_approval
from atlas.auth import role_of
from atlas.context import BUDGET, select_docs, trim_history
from atlas.effects import RefundError, RefundRefused, charge_refund, idempotency_key
from atlas.helpers import compose_answer, search_kb
from atlas.memory import profile_ns, relevant_memories
from atlas.naive import ChatModel
from atlas.research import research_graph
from atlas.security import APPROVER_ROLES
from atlas.state import AtlasState
from atlas.tools import KnowledgeBaseUnavailable, text_of
from atlas.triage import classify

ALLOWED_ROUTES = ("answer", "retrieve", "escalate", "refund")
MAX_RETRIEVE_ATTEMPTS = 3

# Chapter 10's TimeoutPolicy example: a hard wall clock plus an idle timeout,
# attached to Atlas's async node candidate, Chapter 4's retrieve_async, below
# - see tests/test_graph.py for the async-only behavior this policy
# demonstrates.
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
    writer in its superstep.

    Chapter 12: the classifier sees the history slice, not the whole thread.
    `trim_history` returns a new list, so state keeps every turn."""
    view = trim_history(state["messages"], BUDGET.history)  # Chapter 12: bounded
    decision = classify(view)  # typed; route is already constrained
    route = decision.route if decision.route in ALLOWED_ROUTES else "escalate"
    # a new question starts clean: no attempts, no failure, no documents
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
    decision = classify(trim_history(state["messages"], BUDGET.history))
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
    `retrieve_attempts` so `route_after_retrieve` can cap the retry loop.

    Chapter 12 caps the results to the retrieved slice here, before routing:
    an article too large for the slice leaves nothing usable, and the bounded
    retry and escalation handle it like any other empty search."""
    try:
        hits = search_kb(state["messages"])
    except KnowledgeBaseUnavailable as exc:
        return {"error": str(exc)}  # record the failure - do not pretend it worked
    docs = select_docs(hits, BUDGET.retrieved)  # Chapter 12: cap before routing
    attempts = state.get("retrieve_attempts", 0) + 1
    return {"retrieved": docs, "retrieve_attempts": attempts}


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
    """`retrieve` already capped `retrieved` to the slice (Chapter 12), so
    the reply uses what is there. Chapter 13: the profile `recall` loaded
    reaches the reply too."""
    profile = state.get("customer_profile")          # Chapter 13: recall wrote it
    reply = compose_answer(state["messages"], state["retrieved"], profile=profile)
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
    `refund_done` key at all, and neither does a new run that has not reached
    the refund (only the refund node writes it) - `state["refund_done"]` would
    raise KeyError on both; `.get` with a default does not."""
    return state.get("refund_done", False)  # resumes old checkpoints safely


def approval_gate(state: AtlasState) -> Command[Literal["refund", "escalate"]]:
    """Chapter 11: the approval gate, sitting BEFORE the membrane `refund`
    crosses. `interrupt()` suspends the run to the checkpointer and surfaces
    the proposed refund; the human's decision comes back as `decision`, the
    return value of that same `interrupt()` call, once someone resumes the
    thread with `Command(resume=...)`.

    Approve and a re-validated edit route onward to `refund`; reject routes
    to `escalate`. This node does nothing else - on resume, LangGraph
    re-runs it from the top, so any side effect placed before `interrupt()`
    would fire again on every resume. See "A resumed node re-runs from the
    top". Every routed decision also writes `approval` (decision, `by`,
    timestamp) - computed after `interrupt()` returns, so only the resumed
    run records it, and the refund's checkpoint carries it."""
    ticket = state["ticket"]
    decision = interrupt(
        {
            "action": "issue_refund",
            "ticket_id": ticket["id"],
            "customer_id": ticket.get("customer_id"),
            "amount": ticket["amount"],
        }
    )
    record = {  # the audit record: who decided what, and when
        "decision": decision["type"],
        "by": decision.get("by"),
        "at": datetime.now(UTC).isoformat(),
        "shown": decision.get("shown"),  # what the approver saw, echoed back
    }
    if decision["type"] == "approve":
        return Command(update={"approval": record}, goto="refund")
    if decision["type"] == "edit":
        amount = decision["amount"]
        if not 0 < amount <= ticket["amount"]:  # re-validate the human's edit
            return Command(
                update={
                    "approval": record,
                    "error": f"edited amount {amount} out of policy",
                },
                goto="escalate",
            )
        return Command(
            update={"approval": record, "ticket": {**ticket, "amount": amount}},
            goto="refund",
        )
    if decision["type"] == "reject":
        return Command(
            update={
                "approval": record,
                "error": f"refund rejected: {decision.get('reason', '')}",
            },
            goto="escalate",
        )
    raise ValueError(f"unknown decision: {decision['type']}")


def served_approver(runtime: Runtime) -> tuple[str | None, str | None]:
    """(who, why refused) for a run the Agent Server is executing.

    The approver is the identity `@auth.authenticate` proved for THIS run
    (`runtime.server_info.user`), never the `by` a payload carries and never
    `state["approval"]`, which a caller can write. Fails closed: no server
    info, no user, or no approver role is a refusal."""
    user = runtime.server_info.user if runtime.server_info else None
    identity = getattr(user, "identity", None)
    if identity is None:
        return None, "refused: no authenticated approver"
    if role_of(user) not in APPROVER_ROLES:
        return identity, f"refused: {identity} may not approve refunds"
    return identity, None


def _audit(
    runtime: Runtime,
    config: RunnableConfig,
    customer_id: str | None,
    record: dict,
    event: str = "approval",
) -> str | None:
    """Append one row; returns its key (None when the graph has no store)."""
    if runtime.store is None:
        return None
    return record_approval(
        runtime.store,
        customer_id,
        config["configurable"]["thread_id"],
        runtime.execution_info.checkpoint_id,
        record,
        event=event,
    )


def shown_mismatch(shown: dict | None, ticket: dict, required: bool) -> str | None:
    """Why a decision does not bind to the ticket now in state, or None.

    The resume echoes what the approver was shown - `{"shown": {"ticket_id":
    ..., "customer_id": ..., "amount": ...}}`, the interrupt's own fields -
    and the gate, which re-runs on resume from whatever state holds NOW,
    compares the two. A ticket rewritten while the run was paused, even
    only its customer, no longer matches."""
    if shown is None:
        if required:
            return "refused: the decision does not echo what was shown"
        return None
    fields = ("ticket_id", "customer_id", "amount")
    if tuple(shown.get(f) for f in fields) != (
        ticket.get("id"),
        ticket.get("customer_id"),
        ticket.get("amount"),
    ):
        return "refused: the ticket changed after the approver saw it"
    return None


def make_approval_gate(served: bool) -> Callable[..., Command]:
    """Chapter 23: Chapter 11's gate, bound and audited. On the served build
    the approver is the authenticated identity, must hold an approver role,
    and must echo the ticket, customer and amount the interrupt showed."""

    def audited_approval_gate(
        state: AtlasState, config: RunnableConfig, runtime: Runtime
    ) -> Command[Literal["refund", "escalate"]]:
        if state.get("ticket") and "id" not in state["ticket"]:
            # an escalated ticket ({"status": "escalated"}) has nothing left
            # to approve: end cleanly rather than wedge the thread on KeyError
            return Command(update={"error": "no ticket to refund"}, goto="escalate")
        command = approval_gate(state)  # raises GraphInterrupt until a human decides
        ticket = state["ticket"]
        approved = command.update.get("ticket", ticket)
        approval = {
            **command.update["approval"],
            "ticket_id": ticket["id"],
            "customer_id": ticket.get("customer_id"),
            "amount": approved["amount"] if command.goto == "refund" else None,
        }  # <1>
        refusal = shown_mismatch(approval["shown"], ticket, required=served)  # <2>
        if served:  # <3>
            identity, role_refusal = served_approver(runtime)
            user = runtime.server_info.user if runtime.server_info else None
            approval = {**approval, "by": identity, "role": role_of(user)}
            refusal = refusal or role_refusal
        if refusal is not None and command.goto == "refund":
            approval = {**approval, "amount": None, "refused": refusal}
            command = Command(
                update={"approval": approval, "error": refusal}, goto="escalate"
            )
        key = _audit(runtime, config, approval["customer_id"], approval)  # <4>
        approval = {**approval, "audit_key": key}
        return Command(
            update={**command.update, "approval": approval}, goto=command.goto
        )

    return audited_approval_gate


audited_approval_gate = make_approval_gate(served=False)


def make_authorized_refund(served: bool) -> Callable[..., dict | Command]:
    """Chapter 23: the refund authorizes itself, immediately before the
    charge, and writes an audit row on every outcome. A run can reach this
    node without passing the gate - a caller's `Command(goto="refund")`, with
    or without an `update`, or a thread created with `supersteps` - so the
    gate's routing decision is not enough, and neither is the approval in
    state: it charges only against the gate's own audit row."""

    def authorized_refund(
        state: AtlasState, config: RunnableConfig, runtime: Runtime
    ) -> dict | Command[Literal["escalate"]]:
        approval = state.get("approval") or {}
        ticket = state.get("ticket") or {}
        thread_id = config["configurable"]["thread_id"]
        identity, refusal = (
            served_approver(runtime) if served else (approval.get("by"), None)
        )  # <5>
        record, key = approval, approval.get("audit_key")
        if runtime.store is not None:  # <6>
            record = approval_on_record(
                runtime.store, approval.get("customer_id"), key
            )
            refusal = refusal or row_refusal(record, ticket, thread_id)
            if refusal is None and served and (
                record.get("role") not in APPROVER_ROLES
                or record.get("by") != identity
            ):
                refusal = "refused: the approval on record is not this approver's"
            if refusal is None and charged_rows(
                runtime.store, record["customer_id"], approval_key=key
            ):
                refusal = "refused: this approval has already been charged"
        elif served:
            refusal = refusal or "refused: no audit store to verify the approval"
        else:  # in process with no store: Chapter 11's record, checked as is
            refusal = refusal or row_refusal(approval, ticket, None)
        record = record or {}
        amount = record.get("amount")
        customer_id = record.get("customer_id") or ticket.get("customer_id")
        row = {
            "by": identity,
            "at": datetime.now(UTC).isoformat(),
            "ticket_id": ticket.get("id"),
            "amount": amount,
            "approval_key": key,
        }
        earlier = (
            charged_rows(
                runtime.store, customer_id, thread=thread_id, ticket=ticket.get("id")
            )
            if refusal is None and runtime.store is not None
            else []
        )  # <7>
        if earlier and earlier[0]["amount"] != amount:
            refusal = f"refused: already refunded {earlier[0]['amount']:.2f}"
        if refusal is not None:
            _audit(runtime, config, customer_id,
                   {**row, "amount": None, "outcome": refusal}, event="refund")
            return Command(
                update={"messages": [AIMessage(f"Refund not issued: {refusal}")]},
                goto="escalate",
            )
        try:
            out = refund({**state, "ticket": {**ticket, "amount": amount}}, config)
        except RefundRefused as exc:  # <8>
            _audit(runtime, config, customer_id,
                   {**row, "outcome": f"provider refused: {exc}"}, event="refund")
            return refund_failed(state)
        except RefundError as exc:
            _audit(runtime, config, customer_id,
                   {**row, "outcome": f"failed: {exc}"}, event="refund")
            raise  # retried by the node's RetryPolicy, then refund_failed
        outcome = "replayed: already charged" if earlier else "charged"
        _audit(runtime, config, customer_id, {**row, "outcome": outcome},
               event="refund")  # <9>
        return {**out, "error": None}  # a stale error does not outlive a charge

    return authorized_refund


def row_refusal(record: dict | None, ticket: dict, thread_id: str | None) -> str | None:
    """Why an approval record does not authorize charging `ticket`, or None.

    `record` is the gate's audit row (or, in process with no store, the
    approval in state). It must be an approve or edit with a positive
    amount, not refused, for this thread, and for exactly this ticket: its
    id, its customer, and the amount state now holds (an edit already wrote
    the edited amount to the ticket)."""
    if not record:
        return "refused: no approval on record for this refund"
    if thread_id is not None and record.get("thread") != thread_id:
        return "refused: no approval on record for this refund"
    amount = record.get("amount")
    if (
        record.get("decision") not in ("approve", "edit")
        or record.get("refused")
        or amount is None
        or amount <= 0
    ):
        return "refused: no approved amount for this refund"
    if (record.get("ticket_id"), record.get("customer_id")) != (
        ticket.get("id"),
        ticket.get("customer_id"),
    ) or (thread_id is not None and amount != ticket.get("amount")):
        return "refused: the approval is for a different ticket"
    return None


# 1. The approval record binds the decision to a ticket: its id, its
#    customer, and the amount the gate sends on to `refund` (after an edit),
#    or None. `refund` charges that amount, for that ticket, and no other.
# 2. On resume the gate re-runs from the state as it is NOW. A ticket
#    rewritten while the run was paused (the threads state API is a write
#    path that is not a run) no longer matches what the approver echoed.
#    The served build requires the echo; in process it is checked when given.
# 3. On the served build (`atlas/deploy/server.py`), the approver is the
#    authenticated user of the run that resumed the gate, and a resume from
#    anyone else - a thread owner typing {"by": "ceo@corp"}, or an anonymous
#    caller - is refused and escalated. The in-process build keeps the
#    payload's `by` (Chapter 11): Atlas's own code resumes the run there,
#    and that graph is not something to expose to callers.
# 4. Reached only once `interrupt()` has returned a decision, so the write
#    happens on the resumed run, after the human acted, never before. Every
#    execution appends its own row (`record_approval`), refusals included,
#    in the namespace of the customer the approver confirmed. The row's key
#    goes into `state["approval"]`: it is how `refund` finds the row.
# 5. Served: the identity of THIS run, checked again here, so a caller who
#    routes straight to `refund` meets the same role check the gate applies.
# 6. `state["approval"]` is caller-writable (run input, a `goto` with an
#    `update`, a thread created with `supersteps`), so it only says where to
#    look. The authority is the gate's own audit row, which no caller can
#    write: it must exist for this thread, be an approve or edit, match this
#    ticket's id, customer and amount, and - served - be by an approver
#    role and by THIS run's identity. One charge per row: an approval that
#    already has a "charged" row authorizes nothing more. The refusal goes
#    in a message, not `error`: the gate may be refusing in the same step.
# 7. The provider dedupes on thread + ticket, so a second approval on this
#    thread for this ticket can never move money again. At the same amount
#    (a fork re-approving what was charged) the row says "replayed", with
#    the amount actually charged, and a reader summing "charged" rows counts
#    the money once. At a different amount it is refused and escalated:
#    the provider would replay the earlier charge, not the one approved.
# 8. The provider's cap (`RefundRefused`) can never succeed on retry, so the
#    node compensates on the spot with `refund_failed`, the same result the
#    error handler gives a `RefundError` once its retries run out.
# 9. Every outcome is on record, success included: a charge with no row
#    would be a money movement the audit log cannot account for.


def refund(state: AtlasState, config: RunnableConfig) -> dict:
    """Atlas's first crossing of the checkpoint membrane. The idempotency
    key is derived from durable state (`thread_id` + `ticket_id`), so a
    retry or a resume recomputes the SAME key and `charge_refund` dedupes at
    the backend instead of charging twice. The amount comes from state, so
    an approver's edit at the Chapter 11 gate is what gets charged."""
    ticket_id = state["ticket"]["id"]
    amount = state["ticket"]["amount"]  # read from state, not the backend
    thread_id = config["configurable"]["thread_id"]
    key = idempotency_key(thread_id, ticket_id)
    result = charge_refund(key, ticket_id, amount)  # safe to re-run: keyed
    return {"messages": [AIMessage(result)], "refund_done": True}


def refund_failed(state: AtlasState) -> Command:
    # Compensation routes to a human and records the failure. If this
    # reversed a prior action, that reversal would need its own
    # idempotency key - compensation is a side effect too.
    return Command(
        update={"error": "refund failed; needs manual review"},
        goto="escalate",
    )


def recall(state: AtlasState, runtime: Runtime) -> dict:
    """Load what the store knows about this customer before triage, so a
    returning customer is not a stranger."""
    ticket = state.get("ticket") or {}
    if "customer_id" not in ticket:
        return {}                     # no customer on the ticket: nothing to load
    question = text_of(state["messages"][-1])       # the turn that just arrived
    found = relevant_memories(runtime.store, ticket["customer_id"], question)
    return {"customer_profile": {item.key: item.value["value"] for item in found}}


def remember(state: AtlasState, runtime: Runtime) -> dict:
    """After the answer, record the issue this ticket raised - a durable
    fact about the customer that outlives the thread."""
    ticket = state.get("ticket") or {}
    if "customer_id" not in ticket:
        return {}
    human = [m for m in state["messages"] if m.type == "human"]
    question = text_of(human[-1])                    # the issue this ticket raised
    runtime.store.put(
        profile_ns(ticket["customer_id"]),
        "last_issue",
        {"value": question, "ticket": ticket.get("id")},
    )
    return {}


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
    parallel. Defined, not mounted: see the module docstring."""
    out = research_graph.invoke({"sources": derive_sources(state)})
    return {"messages": [summarize_findings(out["findings"])]}


def _make_builder(
    triage_node: Callable[[AtlasState], dict],
    resolve_node: Callable[[AtlasState], dict] = answer,
    served: bool = False,
) -> StateGraph:
    """Chapter 17, "Mounting the resolve agent": the wiring shared by the
    module-level `builder` below and every graph `build_graph` constructs -
    the same Chapter 6-14 topology, parameterized on the "triage" callable
    and on `resolve_node`, the callable in the answering position (the
    model-free `answer` by default). Chapter 21's replay fixture reuses it
    with a scripted triage. Chapter 23 adds `served`: True mounts the
    fail-closed approval gate and refund the Agent Server build uses, which
    take the approver from the run's authenticated identity; False (every
    in-process build) keeps the payload's `by`."""
    b = StateGraph(AtlasState)
    # Chapter 13: memory on both sides of the turn - recall before triage,
    # remember after answer. Store calls only; no retry_policy needed.
    b.add_node("recall", recall)
    b.add_node("remember", remember)
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
    b.add_node("answer", resolve_node)  # the seam: defaults to the model-free stub
    b.add_node("escalate", escalate)
    # Chapter 11: the approval gate - no retry_policy, no side effect. It
    # only interrupts and routes; retrying a suspended interrupt is not the
    # same kind of retry Chapter 10 earned for refund.
    b.add_node("approval_gate", make_approval_gate(served))
    b.add_node(
        "refund",
        make_authorized_refund(served),
        # Chapter 10: safe now that refund is idempotent (earns the retry
        # Chapter 4 held back on side-effecting nodes). error_handler runs once
        # the retry policy stops - retries exhausted, or an exception retry_on
        # does not cover - and compensates by routing to escalate.
        retry_policy=RetryPolicy(max_attempts=3, retry_on=(RefundError,)),
        error_handler=refund_failed,
    )
    b.add_edge(START, "recall")
    b.add_edge("recall", "triage")
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
    b.add_edge("answer", "remember")
    b.add_edge("remember", END)
    b.add_edge("escalate", END)
    b.add_edge("refund", END)
    # approval_gate has no static outgoing edge - it always returns a
    # Command with goto="refund" or goto="escalate", the same
    # dynamic-routing shape triage_with_command uses above. Its
    # Command[Literal["refund", "escalate"]] return type is the only place
    # the graph learns those destinations, so get_graph() draws them.
    return b


def build_graph(
    model: ChatModel | None = None,
    resolve_node: Callable[[AtlasState], dict] | None = None,
) -> Pregel:
    """Chapter 17 adds `resolve_node`: the callable mounted as "answer"
    (`atlas/resolve.py`'s `make_resolve_node` builds one); None keeps the
    model-free `answer`. Chapter 21, "Testing non-determinism: replaying a
    checkpoint", gives `model=` its use: the model is a parameter, the way
    `create_agent` already takes one, instead of the module-level `classify`
    every node closes over. `model=None` reconstructs the exact same graph
    as the module-level `graph` below (the real, `create_agent`-backed
    `classify`); passing Chapter 1's `atlas.breaks.ScriptedModel` swaps ONLY
    the triage step's decision source - `triage_node` below reads the
    scripted model's next response directly, the same one-call-per-turn
    contract `ScriptedModel` already provides - so a replay fixture can force
    a specific route deterministically, with zero real model calls, without
    touching `builder`/`graph` any other test or CI depends on."""
    if model is None:
        triage_node = triage
    else:

        def triage_node(state: AtlasState) -> dict:
            ai = model.invoke(trim_history(state["messages"], BUDGET.history))
            route = ai.content if ai.content in ALLOWED_ROUTES else "escalate"
            return {
                "route": route,
                "retrieve_attempts": 0,
                "error": None,
                "retrieved": Overwrite([]),
            }

    return _make_builder(
        triage_node, resolve_node=resolve_node or answer
    ).compile(checkpointer=InMemorySaver(), store=InMemoryStore())


builder = _make_builder(triage)

# Chapter 9: compiled onto a checkpointer, so every superstep is saved. The
# dev/test default - RAM-backed, gone on restart, but the right tool for
# tests: it exercises the real checkpointing path with no external service.
#
# Chapter 13 configures the store BESIDE the checkpointer, not instead of it:
# they do different jobs (in-thread state vs cross-thread facts). Passing it
# here is what makes `runtime.store` non-None inside `recall`/`remember`;
# `graph.store` is the same instance (atlas/run.py's reflection writes to it).
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
