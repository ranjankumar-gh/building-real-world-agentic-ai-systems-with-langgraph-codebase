"""Chapter 27, "Capstone" - atlas/sla_watch.py, the SLA Watch vertical.

See "Building SLA Watch by reuse, not rebuild". SLA Watch is a NEW small
vertical - proactive monitoring, not a refund-flow variant - built almost
entirely from infrastructure this book already shipped: Chapter 7's narrow
`@tool` discipline (`atlas/tools.py`'s `list_at_risk_tickets`/`send_checkin`),
Chapter 9's checkpointer (a pending approval can sit suspended for days),
Chapter 11's raw `interrupt()` primitive - the hand-written-node path, not
`HumanInTheLoopMiddleware`, whose `edited_action` schema is scoped to a
single tool call's arguments - and Chapter 13's `BaseStore`, reached through
`runtime.store`, the same `Runtime` handle `atlas/graph.py`'s
`remember`/`recall` nodes already use, not a module-level singleton (the
book's own text says "the exact instance Chapter 13 configured", which
means the handle every node receives, not a name importable from
`atlas.memory` - there is no such module-level `store` there).

`SLAWatchState` is the three-channel schema: `at_risk` (scan_tickets'
output), `drafts` (draft_checkins' output), `decisions` (the human's
per-draft decisions from approval_gate) - no reducer beyond Chapter 5's
default last-value channel, since each node returns its own key's full new
value.

`flagged_ns` is a new namespace on the SAME store Chapter 13 configured -
"has this ticket already been flagged" is a new fact, not a new
persistence layer, so a ticket checked on yesterday does not get a second
check-in drafted today just because it is still open.

`compose_checkin` is the one plain function the chapter's own prose calls
but never prints as code - a short, deterministic template (not a model
call), the same "trivial glue, not the pedagogical point" role
`atlas/helpers.py`'s stubs once held for `compose_answer`, except this one
is real: SLA Watch never needs a model to explain why a ticket is being
checked on.

`approval_gate` reuses Chapter 11's `interrupt()` mechanics but defines its
OWN decision shape - `{"type": "approve"|"edit"|"reject", "edited_message":
str}` per draft - since nothing here routes through
`HumanInTheLoopMiddleware`. One decision per draft, in order; `send_checkins`
re-validates an edited message (Chapter 11's discipline for an edited
refund amount, applied to an edited message instead) and never calls
`send_checkin` for a rejected draft - the guarantee `atlas/evals.py`'s
`checkins_sent_only_if_approved` checks.

`build_sla_watch_graph` wires the four nodes into a real, compiled
`StateGraph` - scan -> draft -> gate -> send - on an `InMemorySaver`
checkpointer plus an `InMemoryStore` (Chapter 9/13's dev/test defaults), so
`tests/test_sla_watch.py` can drive a REAL interrupt/resume cycle instead of
mocking `interrupt()` - the same "prove the suspension is real" discipline
Chapter 11's own end-to-end test used. The chapter's own text describes
this wiring ("one line each") without printing the assembly; this function
is that one line, made real."""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt

from atlas.tools import list_at_risk_tickets, send_checkin


class SLAWatchState(TypedDict):
    at_risk: list[dict]     # tickets past the threshold, from scan_tickets
    drafts: list[dict]      # {"ticket_id": str, "message": str}, from draft_checkins
    decisions: list[dict]   # the human's per-draft decision, from approval_gate


def flagged_ns() -> tuple:
    return ("sla_watch", "flagged")


def checkin_key(ticket_id: str) -> str:
    """A stable key for one logical check-in, the same contract as Chapter
    10's `idempotency_key` for refunds: identical across retries and resumes -
    never random, never time-based - so a replayed `send_checkins` collapses
    onto the same key instead of messaging the customer again.

    Scoped to the ticket rather than the thread, because `flagged_ns()` is
    itself global: one at-risk ticket is one logical check-in no matter which
    scan drafted it. A refund keys on the thread because the same ticket can
    legitimately be refunded more than once; an SLA check-in cannot."""
    return f"checkin:{ticket_id}"


def compose_checkin(ticket: dict) -> str:
    """Draft a short, neutral check-in for one at-risk ticket. Referenced by
    "Building SLA Watch by reuse, not rebuild" but not printed there - see
    the module docstring. Deterministic on purpose: SLA Watch never needs a
    model call to explain why a ticket is being checked on."""
    return (
        f"Hi, we're still working on ticket {ticket['ticket_id']} and wanted "
        "to check in - thank you for your patience. We'll follow up again soon."
    )


def scan_tickets(state: SLAWatchState, runtime: Runtime) -> dict:
    candidates = list_at_risk_tickets.invoke({"threshold_hours": 24})
    unflagged = [
        t for t in candidates
        if runtime.store.get(flagged_ns(), t["ticket_id"]) is None
    ]
    return {"at_risk": unflagged}   # <1>


def draft_checkins(state: SLAWatchState, runtime: Runtime) -> dict:
    """Draft a check-in per at-risk ticket, and claim the ticket as we go.

    The flag is written HERE, not after sending. An approval gate can sit for
    days (Chapter 11), and the scan runs hourly - flagging only on send would
    leave a parked ticket unflagged, so every subsequent scan would re-draft
    it and open another interrupt for the same customer. The flag is a claim
    on the ticket for the duration of the decision, not a record that a
    message went out."""
    drafts = [
        {"ticket_id": t["ticket_id"], "message": compose_checkin(t)}
        for t in state["at_risk"]
    ]
    for draft in drafts:
        runtime.store.put(
            flagged_ns(), draft["ticket_id"], {"status": "drafted"}
        )
    return {"drafts": drafts}


def approval_gate(state: SLAWatchState) -> dict:
    decisions = interrupt({"action": "sla_checkins", "drafts": state["drafts"]})
    return {"decisions": decisions}


def send_checkins(state: SLAWatchState, runtime: Runtime) -> dict:
    for draft, decision in zip(state["drafts"], state["decisions"]):
        if decision["type"] == "reject":
            # Release the claim `draft_checkins` took, so a rejected ticket
            # resurfaces on the next scan rather than being silently dropped.
            runtime.store.delete(flagged_ns(), draft["ticket_id"])
            continue
        message = decision.get("edited_message", draft["message"])   # <2>
        send_checkin.invoke(
            {
                "key": checkin_key(draft["ticket_id"]),   # <3>
                "ticket_id": draft["ticket_id"],
                "message": message,
            }
        )
        runtime.store.put(flagged_ns(), draft["ticket_id"], {"status": "sent"})
    return {}


# 1. Checking `runtime.store.get(flagged_ns(), ...)` before adding a ticket to
#    this run's batch is Chapter 13's memory horizon, applied to a new fact:
#    "has this ticket already been flagged" needs to outlive a single run, or
#    the same ticket gets a check-in drafted every hour until someone acts on
#    it.
# 2. `{"type": "approve"|"edit"|"reject", "edited_message": str}` is a
#    decision shape defined for this node alone - it doesn't need to match
#    `HumanInTheLoopMiddleware`'s `edited_action` schema, because nothing here
#    is routed through that middleware. Re-validating an edit against policy
#    before it reaches `send_checkin` is the same discipline Chapter 11
#    applied to an edited refund amount, applied here to an edited message
#    instead.


def build_sla_watch_graph(store: BaseStore | None = None):
    """Assemble scan -> draft -> gate -> send onto a real checkpointer/store
    pair - the dev/test defaults Chapters 9 and 13 already established."""
    builder = StateGraph(SLAWatchState)
    builder.add_node("scan_tickets", scan_tickets)
    builder.add_node("draft_checkins", draft_checkins)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("send_checkins", send_checkins)
    builder.add_edge(START, "scan_tickets")
    builder.add_edge("scan_tickets", "draft_checkins")
    builder.add_edge("draft_checkins", "approval_gate")
    builder.add_edge("approval_gate", "send_checkins")
    builder.add_edge("send_checkins", END)
    return builder.compile(
        checkpointer=InMemorySaver(), store=store or InMemoryStore()
    )
