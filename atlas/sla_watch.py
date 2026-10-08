"""Chapter 27, "Capstone" - atlas/sla_watch.py, the SLA Watch vertical.

See "Building SLA Watch by reuse, not rebuild". SLA Watch is a new small
vertical, proactive monitoring rather than a refund-flow variant, built from
infrastructure this book already shipped: Chapter 7's narrow `@tool`
discipline (`atlas/tools.py`'s `list_at_risk_tickets`/`send_checkin`),
Chapter 9's checkpointer (a pending approval can sit suspended for days),
Chapter 11's raw `interrupt()` primitive (the hand-written-node path, not
`HumanInTheLoopMiddleware`, whose `edited_action` schema is scoped to one
tool call's arguments), Chapter 13's `BaseStore` reached through
`runtime.store`, and Chapter 23's served approver and append-only audit rows.

`SLAWatchState` has three channels and no reducer beyond Chapter 5's default
last-value channel: `at_risk` (scan_tickets' output), `drafts`
(draft_checkins' output) and `decisions` (approval_gate's checked decision
per draft).

`flagged_ns` is a new namespace on the same store. "Has this ticket already
been flagged" is a new fact, not a new persistence layer. The flag is a
claim taken at draft time and released by anything that is not a send.

`compose_checkin` is a short, deterministic template, not a model call:
SLA Watch never needs a model to say why a ticket is being checked on, so it
has no model cost to budget.

The gate fails closed. The resume must carry one decision per draft, each
`{"type": "approve" | "edit" | "reject", "edited_message": str}`. An unknown
type, a missing decision, a resume of the wrong length, or an edited
message that fails `checkin_refusal` becomes a "refused" decision: nothing
is sent and the ticket's claim is released, so it resurfaces on the next
scan. On the served build the approver is the authenticated identity
(`atlas/graph.py`'s `served_approver`), so only `APPROVER_ROLES` may approve
a check-in, the same roles that approve a refund. Every checked decision,
refusals included, is appended to the customer's audit namespace with
`atlas/audit.py`'s `record_approval` (event "checkin").

A scan that finds nothing at risk ends there (Chapter 6's routing): no
empty approval request every hour.

`build_sla_watch_graph` compiles the in-process graph on `InMemorySaver`
plus `InMemoryStore`, so the tests drive a real interrupt/resume cycle.
`build_served_sla_watch` compiles the same topology with neither, for
`langgraph.json`: the Agent Server supplies both, and `langgraph dev`
refuses a graph that brings its own. Both are compiled with
`name="sla-watch"`, which names the root run of every scan in the trace
(Chapter 20); without it a run traces as `LangGraph`."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt

from atlas.audit import record_approval
from atlas.auth import role_of
from atlas.graph import served_approver
from atlas.tools import list_at_risk_tickets, send_checkin

MAX_CHECKIN_CHARS = 600


class SLAWatchState(TypedDict):
    at_risk: list[dict]     # tickets past the threshold, from scan_tickets
    drafts: list[dict]      # {"ticket_id", "customer_id", "message"}, one per ticket
    decisions: list[dict]   # the checked decision per draft, from approval_gate


def flagged_ns() -> tuple:
    return ("sla_watch", "flagged")


def checkin_key(ticket_id: str) -> str:
    """One logical check-in, keyed the way Chapter 10 keys a refund."""
    return f"checkin:{ticket_id}"


def compose_checkin(ticket: dict) -> str:
    """Draft a short, neutral check-in for one at-risk ticket. A template,
    not a model call: SLA Watch has nothing to explain that needs one."""
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


def anything_at_risk(state: SLAWatchState) -> Literal["draft_checkins", "__end__"]:
    return "draft_checkins" if state["at_risk"] else END


def draft_checkins(state: SLAWatchState, runtime: Runtime) -> dict:
    """Draft a check-in per at-risk ticket, and claim the ticket as we go.

    The flag is written here, not after sending. An approval gate can sit
    for days (Chapter 11) and the scan runs hourly, so flagging only on send
    would leave a parked ticket unflagged and the next scan would draft it
    again. The flag is a claim for the duration of the decision."""
    drafts = [
        {
            "ticket_id": t["ticket_id"],
            "customer_id": t.get("customer_id"),
            "message": compose_checkin(t),
        }
        for t in state["at_risk"]
    ]
    for draft in drafts:
        runtime.store.put(
            flagged_ns(), draft["ticket_id"], {"status": "drafted"}
        )
    return {"drafts": drafts}


def checkin_refusal(ticket_id: str, message: Any) -> str | None:
    """Why an edited check-in may not go out, or None. The draft is a fixed
    template; an edit is the only text a human wrote, so it is checked."""
    if not isinstance(message, str) or not message.strip():
        return "refused: the edited message is empty"
    if len(message) > MAX_CHECKIN_CHARS:
        return f"refused: the edited message is over {MAX_CHECKIN_CHARS} characters"
    if ticket_id not in message:
        return f"refused: the edited message does not name {ticket_id}"
    return None


def checked_decision(draft: dict, decision: Any) -> dict:
    """One draft's decision, validated: the message to send, or why not."""
    if not isinstance(decision, dict):
        return {"type": "refused", "reason": "refused: no decision for this draft"}
    by = decision.get("by")  # in process: Chapter 11's unverified `by`
    kind = decision.get("type")
    if kind == "approve":
        return {"type": "approve", "message": draft["message"], "by": by}
    if kind == "edit":
        message = decision.get("edited_message")
        refusal = checkin_refusal(draft["ticket_id"], message)
        if refusal is None:
            return {"type": "edit", "message": message, "by": by}
        return {"type": "refused", "reason": refusal, "by": by}
    if kind == "reject":
        return {"type": "reject", "by": by}
    refusal = f"refused: unknown decision {kind!r}"
    return {"type": "refused", "reason": refusal, "by": by}


def per_draft(resume: Any, n_drafts: int) -> list[Any]:
    """The resume as one decision per draft, or None for every draft when
    it is not a list of exactly that many: a short or shifted list could
    pair an approval with the wrong ticket."""
    if isinstance(resume, list) and len(resume) == n_drafts:
        return resume
    return [None] * n_drafts


def make_checkin_gate(served: bool) -> Callable[[SLAWatchState, Runtime], dict]:
    """Chapter 11's interrupt, with Chapter 23's approver and audit row."""

    def approval_gate(state: SLAWatchState, runtime: Runtime) -> dict:
        drafts = state["drafts"]
        resume = interrupt({"action": "sla_checkins", "drafts": drafts})
        by, refusal = served_approver(runtime) if served else (None, None)  # <2>
        if refusal and by:  # proved, but not an approver
            refusal = f"refused: {by} may not approve check-ins"
        decisions = []
        for draft, decision in zip(drafts, per_draft(resume, len(drafts))):
            checked = checked_decision(draft, decision)
            if served:
                checked = {**checked, "by": by}
            if refusal is not None:
                checked = {"type": "refused", "reason": refusal, "by": by}
            record_checkin(runtime, draft, checked)  # <3>
            decisions.append(checked)
        return {"decisions": decisions}

    return approval_gate


def record_checkin(runtime: Runtime, draft: dict, checked: dict) -> None:
    """Append one checked decision to the customer's audit namespace."""
    if runtime.store is None or runtime.execution_info is None:
        return
    user = runtime.server_info.user if runtime.server_info else None
    sends = checked["type"] in ("approve", "edit")
    record_approval(
        runtime.store,
        draft.get("customer_id"),
        runtime.execution_info.thread_id or "",
        runtime.execution_info.checkpoint_id,
        {
            "decision": checked["type"],
            "ticket_id": draft["ticket_id"],
            "customer_id": draft.get("customer_id"),
            "message": checked.get("message"),
            "key": checkin_key(draft["ticket_id"]) if sends else None,
            "refused": checked.get("reason"),
            "by": checked.get("by"),
            "role": role_of(user),
            "at": datetime.now(UTC).isoformat(),
        },
        event="checkin",
    )


approval_gate = make_checkin_gate(served=False)


def send_checkins(state: SLAWatchState, runtime: Runtime) -> dict:
    for draft, decision in zip(state["drafts"], state["decisions"]):
        if decision.get("type") not in ("approve", "edit"):
            runtime.store.delete(flagged_ns(), draft["ticket_id"])   # <4>
            continue
        send_checkin.invoke(
            {
                "key": checkin_key(draft["ticket_id"]),   # <5>
                "ticket_id": draft["ticket_id"],
                "message": decision["message"],
            }
        )
        runtime.store.put(flagged_ns(), draft["ticket_id"], {"status": "sent"})
    return {}


# 1. Checking `runtime.store.get(flagged_ns(), ...)` before adding a ticket
#    to this run's batch is Chapter 13's memory horizon, applied to a new
#    fact: "has this ticket already been flagged" must outlive a single run.
# 2. On the served build the approver is the identity `@auth.authenticate`
#    proved for this run, and it must hold an approver role; otherwise every
#    draft is refused. A check-in goes to a customer who did not ask for it,
#    so it is signed off by the same roles that sign off a refund.
# 3. Refusals are recorded too: an auditor asking why a ticket never got a
#    check-in finds the row that says so.
# 4. Anything that is not an approved or edited send - a reject, an unknown
#    type, a missing decision, a refused edit - releases the claim
#    `draft_checkins` took, so the ticket resurfaces on the next scan.
# 5. The key is stable across retries and resumes, so a replayed node does
#    not message the customer twice (Chapter 10's membrane rule).


def _builder(served: bool) -> StateGraph:
    builder = StateGraph(SLAWatchState)
    builder.add_node("scan_tickets", scan_tickets)
    builder.add_node("draft_checkins", draft_checkins)
    builder.add_node("approval_gate", make_checkin_gate(served))
    builder.add_node("send_checkins", send_checkins)
    builder.add_edge(START, "scan_tickets")
    builder.add_conditional_edges("scan_tickets", anything_at_risk)
    builder.add_edge("draft_checkins", "approval_gate")
    builder.add_edge("approval_gate", "send_checkins")
    builder.add_edge("send_checkins", END)
    return builder


def build_sla_watch_graph(store: BaseStore | None = None) -> Pregel:
    """The in-process graph, on Chapter 9 and 13's dev/test defaults."""
    return _builder(served=False).compile(
        checkpointer=InMemorySaver(),
        store=store or InMemoryStore(),
        name="sla-watch",
    )


def build_served_sla_watch() -> Pregel:
    """For the Agent Server: no checkpointer and no store of its own."""
    return _builder(served=True).compile(name="sla-watch")
