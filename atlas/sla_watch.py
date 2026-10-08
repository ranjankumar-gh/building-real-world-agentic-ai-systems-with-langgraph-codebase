"""Chapter 27, "Capstone" - atlas/sla_watch.py, the SLA Watch vertical.

See "Building SLA Watch by reuse, not rebuild". SLA Watch is a new small
vertical, proactive monitoring rather than a refund-flow variant, built from
infrastructure this book already shipped: Chapter 7's narrow `@tool`
discipline (`atlas/tools.py`'s `list_at_risk_tickets`/`send_checkin`),
Chapter 9's checkpointer (a pending approval can sit suspended for hours),
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
claim taken at draft time: it names the thread that holds it and when it
was taken, and it lapses after `CLAIM_TTL`, so a run that dies, is
cancelled, or is never resumed cannot keep a ticket out of every later
scan. Anything short of a send releases it, and a run releases only its
own claim.

`compose_checkin` is a short, deterministic template, not a model call:
SLA Watch never needs a model to say why a ticket is being checked on, so it
has no model cost to budget.

The gate fails closed. The resume must carry one decision per draft, each
`{"type": "approve" | "edit" | "reject", "ticket_id": str,
"edited_message": str}`, where `ticket_id` echoes the draft the reviewer was
shown. An unknown type, a missing decision, a resume of the wrong length, a
decision that echoes another ticket, or an edited message that fails
`checkin_refusal` becomes a "refused" decision. On the served build the
approver is the authenticated identity (`atlas/graph.py`'s
`served_approver`), so only `APPROVER_ROLES` may approve a check-in, the
same roles that approve a refund. Every checked decision, refusals
included, is appended to the customer's audit namespace with
`atlas/audit.py`'s `record_approval` (event "checkin"), and the decision in
state carries that row's key.

`send_checkins` does not trust the decisions in state. State is writable
from outside a run (the threads state API, or a write made `as_node` the
gate), so a decision there proves nothing about who approved it. The send
reads the gate's own row, as Chapter 23's refund does, and sends only when
the row is a check-in approval on this thread, for this ticket and
customer, for exactly this message (by its SHA-256), signed on the served
build by the approver resuming the run, and only when no "checkin_send"
row already records a send for the ticket. Each
attempted send is a "checkin_send" row: sent, replayed, or why not.

A scan that finds nothing at risk ends there (Chapter 6's routing): no
empty approval request every hour.

`build_sla_watch_graph` compiles the in-process graph on `InMemorySaver`
plus `InMemoryStore`, so the tests drive a real interrupt/resume cycle.
`build_served_sla_watch` compiles the same topology with neither, for
`langgraph.json`: the Agent Server supplies both, and `langgraph dev`
refuses a graph that brings its own. Both are compiled with
`name="sla-watch"`, which names the root run of every scan in the trace
(Chapter 20); without it a run traces as `LangGraph`. `atlas/auth.py` lets
only an approver start an sla-watch run or cron."""

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt

from atlas.audit import audit_ns, record_approval
from atlas.auth import role_of
from atlas.graph import served_approver
from atlas.security import APPROVER_ROLES
from atlas.tools import list_at_risk_tickets, send_checkin

MAX_CHECKIN_CHARS = 600
CLAIM_TTL = timedelta(hours=24)


class SLAWatchState(TypedDict):
    at_risk: list[dict]     # tickets past the threshold, from scan_tickets
    drafts: list[dict]      # {"ticket_id", "customer_id", "message"}, one per ticket
    decisions: list[dict]   # the checked decision per draft, from approval_gate


def flagged_ns() -> tuple:
    return ("sla_watch", "flagged")


def thread_of(runtime: Runtime) -> str:
    """The thread this run is on ("" when a test hands a bare Runtime)."""
    info = runtime.execution_info
    return (info.thread_id if info else None) or ""


def compose_checkin(ticket: dict) -> str:
    """Draft a short, neutral check-in for one at-risk ticket. A template,
    not a model call: SLA Watch has nothing to explain that needs one."""
    return (
        f"Hi, we're still working on ticket {ticket['ticket_id']} and wanted "
        "to check in - thank you for your patience. We'll follow up again soon."
    )


def claim_holds(claim: dict | None, now: datetime) -> bool:
    """Whether a flag still keeps its ticket out of the scan. A send holds
    for good; a draft's claim only until CLAIM_TTL has passed."""
    if not claim:
        return False
    if claim.get("status") == "sent":
        return True
    try:
        claimed_at = datetime.fromisoformat(claim["claimed_at"])
    except (KeyError, TypeError, ValueError):
        return False  # no owner time: it suppresses nothing
    return claim.get("status") == "drafted" and now - claimed_at < CLAIM_TTL


def scan_tickets(state: SLAWatchState, runtime: Runtime) -> dict:
    candidates = list_at_risk_tickets.invoke({"threshold_hours": 24})
    now = datetime.now(UTC)
    unflagged = []
    for ticket in candidates:
        flag = runtime.store.get(flagged_ns(), ticket["ticket_id"])
        if not claim_holds(flag.value if flag else None, now):   # <1>
            unflagged.append(ticket)
    return {"at_risk": unflagged}


def anything_at_risk(state: SLAWatchState) -> Literal["draft_checkins", "__end__"]:
    return "draft_checkins" if state["at_risk"] else END   # <2>


def draft_checkins(state: SLAWatchState, runtime: Runtime) -> dict:
    """Draft a check-in per at-risk ticket, and claim the ticket as we go.

    The flag is written here, not after sending. An approval gate can sit
    for hours (Chapter 11) and the scan runs hourly, so flagging only on
    send would leave a parked ticket unflagged and the next scan would draft
    it again. The flag is a claim for the duration of the decision."""
    drafts = [
        {
            "ticket_id": t["ticket_id"],
            "customer_id": t.get("customer_id"),
            "message": compose_checkin(t),
        }
        for t in state["at_risk"]
    ]
    claim = {
        "status": "drafted",
        "thread_id": thread_of(runtime),
        "claimed_at": datetime.now(UTC).isoformat(),
    }
    for draft in drafts:
        flag = runtime.store.get(flagged_ns(), draft["ticket_id"])
        if flag is None or flag.value.get("status") != "sent":   # <3>
            runtime.store.put(flagged_ns(), draft["ticket_id"], claim)
    return {"drafts": drafts}


def release_claim(runtime: Runtime, ticket_id: str) -> None:
    """Drop this run's own draft claim. Another thread's claim, or a sent
    flag, is not this run's to release."""
    flag = runtime.store.get(flagged_ns(), ticket_id)
    if flag and flag.value.get("status") == "drafted" and (
        flag.value.get("thread_id") == thread_of(runtime)
    ):
        runtime.store.delete(flagged_ns(), ticket_id)


# 1. Checking `runtime.store.get(flagged_ns(), ...)` before adding a ticket
#    to this run's batch is Chapter 13's memory horizon, applied to a new
#    fact: "has this ticket already been flagged" must outlive a single run.
#    A draft's claim lapses after CLAIM_TTL, so a dead or abandoned run
#    delays a ticket's check-in by a day at most.
# 2. With nothing at risk the run ends here, rather than pausing with an
#    empty approval request every hour.
# 3. The claim names its thread and its time, so `release_claim` touches
#    only its own and `claim_holds` can tell a live claim from a stale one.
#    A run that scanned before another run sent never claims over the
#    "sent" flag, so its reject cannot release the record of that send.


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
    if decision.get("ticket_id") != draft["ticket_id"]:   # <1>
        refusal = f"refused: the decision does not echo {draft['ticket_id']}"
        return {"type": "refused", "reason": refusal, "by": by}
    kind = decision.get("type")
    if kind == "approve":
        return {"type": "approve", "message": draft["message"], "by": by}
    if kind == "edit":
        message = decision.get("edited_message")
        refusal = checkin_refusal(draft["ticket_id"], message)   # <2>
        if refusal is None:
            return {"type": "edit", "message": message, "by": by}
        return {"type": "refused", "reason": refusal, "by": by}
    if kind == "reject":
        return {"type": "reject", "by": by}
    refusal = f"refused: unknown decision {kind!r}"   # <3>
    return {"type": "refused", "reason": refusal, "by": by}


def per_draft(resume: Any, n_drafts: int) -> list[Any]:
    """The resume as one decision per draft, or None for every draft when
    it is not a list of exactly that many: a short or shifted list could
    pair an approval with the wrong ticket."""
    if isinstance(resume, list) and len(resume) == n_drafts:
        return resume
    return [None] * n_drafts   # <4>


# 1. The decision names the ticket the reviewer was shown. On resume the
#    gate re-runs against the drafts in state NOW; if they changed while
#    the run was paused, the echo no longer matches and nothing binds.
# 2. An edit is the only text a human wrote, so it is re-validated.
# 3. Only the three known types do anything; a typo is refused.
# 4. A missing decision is refused rather than left to strand its ticket.


def message_digest(message: Any) -> str | None:
    """SHA-256 of the exact text a decision would send."""
    if not isinstance(message, str):
        return None
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def checkin_key(ticket_id: str, message: str) -> str:
    """One logical check-in: this ticket, this exact text (Chapter 10)."""
    return f"checkin:{ticket_id}:{message_digest(message)[:16]}"


def make_checkin_gate(served: bool) -> Callable[[SLAWatchState, Runtime], dict]:
    """Chapter 11's interrupt, with Chapter 23's approver and audit row."""

    def approval_gate(state: SLAWatchState, runtime: Runtime) -> dict:
        drafts = state["drafts"]
        resume = interrupt({"action": "sla_checkins", "drafts": drafts})
        by, refusal = served_approver(runtime) if served else (None, None)  # <1>
        if refusal and by:  # proved, but not an approver
            refusal = f"refused: {by} may not approve check-ins"
        decisions = []
        for draft, decision in zip(drafts, per_draft(resume, len(drafts))):
            checked = checked_decision(draft, decision)
            if served:
                checked = {**checked, "by": by}
            if refusal is not None:
                checked = {"type": "refused", "reason": refusal, "by": by}
            key = record_checkin(runtime, draft, checked)  # <2>
            decisions.append({**checked, "audit_key": key})
        return {"decisions": decisions}

    return approval_gate


def record_checkin(runtime: Runtime, draft: dict, checked: dict) -> str | None:
    """Append one checked decision to the customer's audit namespace and
    return the row's key (None when the graph has no store)."""
    if runtime.store is None:
        return None
    info = runtime.execution_info
    user = runtime.server_info.user if runtime.server_info else None
    message = checked.get("message")
    sends = checked["type"] in ("approve", "edit")
    return record_approval(
        runtime.store,
        draft.get("customer_id"),
        thread_of(runtime),
        info.checkpoint_id if info else "",
        {
            "decision": checked["type"],
            "ticket_id": draft["ticket_id"],
            "customer_id": draft.get("customer_id"),
            "message": message,
            "message_sha256": message_digest(message) if sends else None,
            "key": checkin_key(draft["ticket_id"], message) if sends else None,
            "refused": checked.get("reason"),
            "by": checked.get("by"),
            "role": role_of(user),
            "at": datetime.now(UTC).isoformat(),
        },
        event="checkin",
    )


approval_gate = make_checkin_gate(served=False)


def checkin_on_record(
    store: BaseStore, customer_id: str | None, key: str | None
) -> dict | None:
    """The check-in row the gate wrote under `key`, or None."""
    if not key:
        return None
    item = store.get(audit_ns(customer_id or "unknown"), key)
    if item is None or item.value.get("event") != "checkin":
        return None
    return item.value


def send_refusal(
    row: dict | None, draft: dict, decision: dict, thread_id: str
) -> str | None:
    """Why the gate's row does not authorize sending this draft, or None."""
    message = decision.get("message")
    if row is None or row.get("thread") != thread_id:
        return "refused: no check-in approval on record"
    if row.get("decision") not in ("approve", "edit") or row.get("refused"):
        return "refused: the check-in on record was not approved"
    if (row.get("ticket_id"), row.get("customer_id")) != (
        draft["ticket_id"],
        draft.get("customer_id"),
    ):
        return "refused: the approval is for a different ticket"
    if row.get("message_sha256") != message_digest(message):
        return "refused: the message is not the one approved"
    return None


def make_send_checkins(served: bool) -> Callable[[SLAWatchState, Runtime], dict]:
    """Send what the gate's own audit rows approve, and nothing else."""

    def send_checkins(state: SLAWatchState, runtime: Runtime) -> dict:
        thread_id = thread_of(runtime)
        identity, refusal = served_approver(runtime) if served else (None, None)
        if refusal and identity:
            refusal = f"refused: {identity} may not send check-ins"
        for draft, decision in zip(state["drafts"], state["decisions"]):
            if decision.get("type") not in ("approve", "edit"):
                release_claim(runtime, draft["ticket_id"])   # <1>
                continue
            key = decision.get("audit_key")
            row = checkin_on_record(runtime.store, draft.get("customer_id"), key)
            why = refusal or send_refusal(row, draft, decision, thread_id)  # <2>
            if why is None and served and (
                row.get("role") not in APPROVER_ROLES or row.get("by") != identity
            ):
                why = "refused: the approval on record is not this approver's"
            outcome = why or deliver(runtime, draft, row, key)   # <3>
            if why is not None:
                release_claim(runtime, draft["ticket_id"])
            record_approval(
                runtime.store,
                draft.get("customer_id"),
                thread_id,
                runtime.execution_info.checkpoint_id if runtime.execution_info else "",
                {
                    "ticket_id": draft["ticket_id"],
                    "customer_id": draft.get("customer_id"),
                    "approval_key": key,
                    "outcome": outcome,
                    "by": identity if served else (row or {}).get("by"),
                    "at": datetime.now(UTC).isoformat(),
                },
                event="checkin_send",
            )
        return {}

    return send_checkins


def deliver(runtime: Runtime, draft: dict, row: dict, approval_key: str) -> str:
    """Send the approved text once per ticket, and flag the ticket."""
    store, ticket_id = runtime.store, draft["ticket_id"]
    sent_before = store.search(
        audit_ns(draft.get("customer_id") or "unknown"),
        filter={"event": "checkin_send", "ticket_id": ticket_id, "outcome": "sent"},
        limit=1,
    )
    if sent_before and sent_before[0].value.get("approval_key") != approval_key:
        return f"refused: {ticket_id} already had a check-in"   # <4>
    if not sent_before:
        send_checkin.invoke(
            {"key": row["key"], "ticket_id": ticket_id, "message": row["message"]}
        )   # <5>
    store.put(
        flagged_ns(),
        ticket_id,
        {"status": "sent", "thread_id": thread_of(runtime)},
    )
    return "replayed: already sent" if sent_before else "sent"


send_checkins = make_send_checkins(served=False)


# 1. A reject, an unknown type, a missing decision, a refused edit or a
#    refused approver: the gate already wrote the row that says so, and
#    this run's claim is released so the ticket resurfaces.
# 2. The decision in state is only where to look. A state write made
#    `as_node="approval_gate"` can put any message there; it cannot put a
#    row in the audit namespace, which no caller reaches (atlas/auth.py).
# 3. Every attempted send leaves a "checkin_send" row, a refusal included,
#    so a forged decision is on record rather than silent.
# 4. A check-in already went out on this ticket under another approval:
#    a lapsed claim re-drafted it, or two runs scanned it before either
#    drafted. The "sent" flag can be overwritten; the append-only
#    "checkin_send" row cannot, so that row is what decides.
# 5. The text sent is the row's, and the key carries its hash, so a replay
#    after a crash collapses onto the send that already went out.


def _builder(served: bool) -> StateGraph:
    builder = StateGraph(SLAWatchState)
    builder.add_node("scan_tickets", scan_tickets)
    builder.add_node("draft_checkins", draft_checkins)
    builder.add_node("approval_gate", make_checkin_gate(served))
    builder.add_node("send_checkins", make_send_checkins(served))
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
