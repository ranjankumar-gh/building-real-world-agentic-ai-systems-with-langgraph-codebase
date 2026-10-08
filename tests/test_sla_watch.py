"""Chapter 27, "Capstone" - atlas/sla_watch.py.

See "Building SLA Watch by reuse, not rebuild". `scan_tickets`,
`draft_checkins` and `send_checkins` need no model or LangSmith connection,
so they are unit-tested directly with a hand-built `Runtime`, the convention
`tests/test_graph.py` and `tests/test_memory.py` use for Chapter 13's
store-backed nodes. The gate is exercised through the compiled graph with a
real `interrupt()`/`Command(resume=...)` cycle, never a monkeypatched
`interrupt`: a decision that is unknown, missing, echoes another ticket, or
is an out-of-policy edit sends nothing and releases the claim; an empty
scan ends without pausing; a claim names its thread and lapses after
CLAIM_TTL; on the served build the approver is the authenticated identity
of the run that resumes; every checked decision is an append-only audit
row; and the send re-verifies that row, so a decision written into state
from outside the gate sends nothing and is itself recorded."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import ExecutionInfo, Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from atlas.audit import audit_ns
from atlas.sla_watch import (
    CLAIM_TTL,
    build_served_sla_watch,
    build_sla_watch_graph,
    checked_decision,
    checkin_key,
    checkin_refusal,
    claim_holds,
    compose_checkin,
    draft_checkins,
    flagged_ns,
    message_digest,
    per_draft,
    record_checkin,
    scan_tickets,
    send_checkins,
)
from atlas.tools import _SLA_TICKETS

NO_DECISION = {"type": "refused", "reason": "refused: no decision for this draft"}
T2001 = {"ticket_id": "T-2001", "customer_id": "C-2", "hours_open": 30}
DRAFT = {"ticket_id": "T-2001", "customer_id": "C-2", "message": "original T-2001"}
TEMPLATE = compose_checkin({"ticket_id": "T-2001"})
APPROVE = {"type": "approve", "ticket_id": "T-2001"}
REJECT = {"type": "reject", "ticket_id": "T-2001"}


@pytest.fixture(autouse=True)
def _clean_send_ledger():
    """`send_checkin` is idempotent by key against a process-global backend,
    which is what makes it survive a node replay - and what would otherwise
    let one test's send suppress the next test's identical ticket id."""
    _SLA_TICKETS.reset()
    yield
    _SLA_TICKETS.reset()


def _config(thread_id: str, user: object | None = None) -> dict:
    conf: dict = {"thread_id": thread_id, "assistant_id": "sla-watch"}
    if user is not None:
        conf["langgraph_auth_user"] = user
    return {"configurable": conf}


def _runtime(store: InMemoryStore, thread_id: str = "th-1") -> Runtime:
    """A hand-built Runtime on a thread, as a node sees it inside a run."""
    info = ExecutionInfo(
        checkpoint_id="cp-1", checkpoint_ns="", task_id="task-1", thread_id=thread_id
    )
    return Runtime(store=store, execution_info=info)


def _claim(thread_id: str = "th-1", age: timedelta = timedelta(0)) -> dict:
    at = datetime.now(UTC) - age
    return {"status": "drafted", "thread_id": thread_id, "claimed_at": at.isoformat()}


def _sent() -> list[tuple[str, str]]:
    return [(m["ticket_id"], m["message"]) for m in _SLA_TICKETS.sent]


def _rows(store: InMemoryStore, customer_id: str = "C-2", event: str = "checkin"):
    rows = store.search(audit_ns(customer_id), filter={"event": event}, limit=100)
    return sorted((r.value for r in rows), key=lambda v: v["at"])


def _paused(store: InMemoryStore, thread_id: str) -> tuple[object, dict]:
    graph = build_sla_watch_graph(store=store)
    config = _config(thread_id)
    graph.invoke({}, config)
    return graph, config


def _gated(runtime: Runtime, checked: dict, draft: dict = DRAFT) -> dict:
    """A decision as the gate leaves it in state: checked, with its row key."""
    return {**checked, "audit_key": record_checkin(runtime, draft, checked)}


# --- flagged_ns / compose_checkin / checkin_key -----------------------------


def test_flagged_ns_is_its_own_namespace_on_the_shared_store():
    assert flagged_ns() == ("sla_watch", "flagged")


def test_compose_checkin_mentions_the_tickets_own_id():
    assert "T-2001" in compose_checkin({"ticket_id": "T-2001"})


def test_the_template_draft_passes_the_edit_policy():
    """The policy an edit must meet is one the shipped template meets."""
    assert checkin_refusal("T-2001", TEMPLATE) is None


def test_checkin_key_binds_the_ticket_and_the_exact_text():
    key = checkin_key("T-2001", "hello T-2001")

    assert key == f"checkin:T-2001:{message_digest('hello T-2001')[:16]}"
    assert key == checkin_key("T-2001", "hello T-2001")  # stable across replays
    assert key != checkin_key("T-2001", "hello T-2001!")
    assert key != checkin_key("T-2002", "hello T-2001")


# --- scan_tickets: the memory-horizon check, and claims that lapse ----------


def test_scan_tickets_surfaces_the_seeded_at_risk_ticket():
    result = scan_tickets({}, _runtime(InMemoryStore()))

    assert result == {"at_risk": [T2001]}


def test_scan_tickets_skips_a_ticket_with_a_live_claim():
    store = InMemoryStore()
    store.put(flagged_ns(), "T-2001", _claim("someone-elses-thread"))

    assert scan_tickets({}, _runtime(store)) == {"at_risk": []}


def test_scan_tickets_skips_a_ticket_already_sent():
    store = InMemoryStore()
    store.put(flagged_ns(), "T-2001", {"status": "sent", "thread_id": "th-0"})

    assert scan_tickets({}, _runtime(store)) == {"at_risk": []}


@pytest.mark.parametrize(
    "claim",
    [
        _claim("dead-run", age=CLAIM_TTL + timedelta(minutes=1)),
        {"status": "drafted"},  # no owner time: suppresses nothing
        {"status": "drafted", "thread_id": "t", "claimed_at": "not a time"},
    ],
)
def test_a_stale_or_ownerless_claim_is_released_by_the_next_scan(claim):
    """A run that died, was cancelled or was never resumed cannot keep the
    ticket out of every later scan."""
    store = InMemoryStore()
    store.put(flagged_ns(), "T-2001", claim)

    assert scan_tickets({}, _runtime(store)) == {"at_risk": [T2001]}


def test_claim_holds_until_the_ttl_and_a_send_holds_for_good():
    now = datetime.now(UTC)

    assert claim_holds(_claim(age=CLAIM_TTL - timedelta(minutes=1)), now)
    assert not claim_holds(_claim(age=CLAIM_TTL), now)
    assert claim_holds({"status": "sent"}, now)
    assert not claim_holds(None, now)


def test_a_never_resumed_run_does_not_suppress_the_ticket_past_the_ttl():
    """The two-identity scenario: one run claims T-2001 and is abandoned at
    the gate. Within the TTL the next scan skips it; after, it re-drafts."""
    store = InMemoryStore()
    _paused(store, "abandoned")
    graph = build_sla_watch_graph(store=store)

    assert "__interrupt__" not in graph.invoke({}, _config("next-hour"))

    claim = store.get(flagged_ns(), "T-2001").value
    aged = datetime.now(UTC) - CLAIM_TTL - timedelta(minutes=1)
    store.put(flagged_ns(), "T-2001", {**claim, "claimed_at": aged.isoformat()})
    later = graph.invoke({}, _config("next-day"))

    assert later["__interrupt__"][0].value["drafts"][0]["ticket_id"] == "T-2001"
    assert store.get(flagged_ns(), "T-2001").value["thread_id"] == "next-day"


# --- draft_checkins ----------------------------------------------------------


def test_draft_checkins_builds_one_draft_per_at_risk_ticket():
    result = draft_checkins({"at_risk": [T2001]}, _runtime(InMemoryStore()))

    [draft] = result["drafts"]
    assert draft["ticket_id"] == "T-2001"
    assert draft["customer_id"] == "C-2"
    assert "T-2001" in draft["message"]


def test_draft_checkins_claims_the_ticket_for_its_own_thread():
    """The flag is written at draft time, not send time: the gate can sit
    while the scan runs hourly. The claim names its owner and its time."""
    store = InMemoryStore()

    draft_checkins({"at_risk": [T2001]}, _runtime(store, "th-9"))

    claim = store.get(flagged_ns(), "T-2001").value
    assert (claim["status"], claim["thread_id"]) == ("drafted", "th-9")
    assert claim_holds(claim, datetime.now(UTC))


# --- checked_decision / per_draft / checkin_refusal -------------------------


@pytest.mark.parametrize(
    "decision, expected",
    [
        (APPROVE, {"type": "approve", "message": "original T-2001"}),
        (
            {"type": "edit", "ticket_id": "T-2001", "edited_message": "Still T-2001."},
            {"type": "edit", "message": "Still T-2001."},
        ),
        (REJECT, {"type": "reject"}),
        (
            {"type": "rejected", "ticket_id": "T-2001"},
            {"type": "refused", "reason": "refused: unknown decision 'rejected'"},
        ),
        (
            {"ticket_id": "T-2001"},
            {"type": "refused", "reason": "refused: unknown decision None"},
        ),
        (None, NO_DECISION),
        ("approve", NO_DECISION),
    ],
)
def test_checked_decision_sends_only_a_known_valid_decision(decision, expected):
    checked = checked_decision(DRAFT, decision)
    checked.pop("by", None)

    assert checked == expected


@pytest.mark.parametrize(
    "decision",
    [
        {"type": "approve"},  # no echo
        {"type": "approve", "ticket_id": "T-2002"},  # another ticket
        {"type": "edit", "ticket_id": "T-2002", "edited_message": "T-2001 hi"},
    ],
)
def test_a_decision_that_does_not_echo_the_shown_ticket_is_refused(decision):
    checked = checked_decision(DRAFT, decision)

    assert checked["type"] == "refused"
    assert checked["reason"] == "refused: the decision does not echo T-2001"


@pytest.mark.parametrize(
    "message, reason",
    [
        ("", "refused: the edited message is empty"),
        ("   ", "refused: the edited message is empty"),
        (None, "refused: the edited message is empty"),
        ("T-2001 " + "x" * 600, "refused: the edited message is over 600 characters"),
        ("Hi, checking in.", "refused: the edited message does not name T-2001"),
    ],
)
def test_an_edited_message_is_revalidated(message, reason):
    decision = {"type": "edit", "ticket_id": "T-2001", "edited_message": message}

    assert checked_decision(DRAFT, decision)["reason"] == reason


def test_per_draft_accepts_only_one_decision_per_draft():
    assert per_draft([APPROVE, APPROVE], 2) == [APPROVE, APPROVE]
    assert per_draft([APPROVE], 2) == [None, None]  # short: could be shifted
    assert per_draft([APPROVE] * 3, 2) == [None, None]
    assert per_draft(APPROVE, 1) == [None]  # a dict, not a list


# --- send_checkins: sends only what the gate's own row approves ------------


def test_send_checkins_sends_an_approved_draft_and_flags_it():
    store = InMemoryStore()
    runtime = _runtime(store)
    decision = _gated(runtime, checked_decision(DRAFT, APPROVE))

    send_checkins({"drafts": [DRAFT], "decisions": [decision]}, runtime)

    assert _sent() == [("T-2001", "original T-2001")]
    assert store.get(flagged_ns(), "T-2001").value == {
        "status": "sent",
        "thread_id": "th-1",
    }
    [row] = _rows(store, event="checkin_send")
    assert (row["outcome"], row["approval_key"]) == ("sent", decision["audit_key"])


@pytest.mark.parametrize(
    "decision",
    [REJECT, {"type": "refused", "reason": "x"}, {"type": "rejected"}, {}],
)
def test_send_checkins_sends_nothing_and_releases_its_claim_otherwise(decision):
    store = InMemoryStore()
    store.put(flagged_ns(), "T-2001", _claim("th-1"))

    send_checkins({"drafts": [DRAFT], "decisions": [decision]}, _runtime(store))

    assert _sent() == []
    assert store.get(flagged_ns(), "T-2001") is None


def test_a_run_never_releases_another_threads_claim():
    store = InMemoryStore()
    store.put(flagged_ns(), "T-2001", _claim("th-other"))

    send_checkins({"drafts": [DRAFT], "decisions": [REJECT]}, _runtime(store))

    assert store.get(flagged_ns(), "T-2001").value["thread_id"] == "th-other"


@pytest.mark.parametrize(
    "decision, reason",
    [
        (
            {"type": "approve", "message": "Click http://evil.example T-2001"},
            "refused: no check-in approval on record",
        ),
        (
            {"type": "approve", "message": "x", "audit_key": "checkin:made:up"},
            "refused: no check-in approval on record",
        ),
    ],
)
def test_a_decision_with_no_row_behind_it_sends_nothing_and_is_recorded(
    decision, reason
):
    store = InMemoryStore()

    send_checkins({"drafts": [DRAFT], "decisions": [decision]}, _runtime(store))

    assert _sent() == []
    [row] = _rows(store, event="checkin_send")
    assert row["outcome"] == reason


def test_a_message_changed_after_the_gate_is_refused():
    store = InMemoryStore()
    runtime = _runtime(store)
    decision = _gated(runtime, checked_decision(DRAFT, APPROVE))
    forged = {**decision, "message": "Click http://evil.example about T-2001"}

    send_checkins({"drafts": [DRAFT], "decisions": [forged]}, runtime)

    assert _sent() == []
    [row] = _rows(store, event="checkin_send")
    assert row["outcome"] == "refused: the message is not the one approved"


def test_a_row_from_another_thread_or_ticket_authorizes_nothing():
    store = InMemoryStore()
    other_thread = _gated(_runtime(store, "th-other"), checked_decision(DRAFT, APPROVE))
    t2002 = {"ticket_id": "T-2002", "customer_id": "C-2", "message": "original T-2001"}
    other_ticket = _gated(
        _runtime(store), {"type": "approve", "message": "original T-2001"}, t2002
    )

    for decision in (other_thread, other_ticket):
        send_checkins({"drafts": [DRAFT], "decisions": [decision]}, _runtime(store))

    assert _sent() == []
    outcomes = [r["outcome"] for r in _rows(store, event="checkin_send")]
    assert outcomes == [
        "refused: no check-in approval on record",
        "refused: the approval is for a different ticket",
    ]


def test_a_refused_row_authorizes_nothing():
    store = InMemoryStore()
    runtime = _runtime(store)
    refused = _gated(runtime, {"type": "refused", "reason": "refused: x"})

    send_checkins(
        {"drafts": [DRAFT], "decisions": [{**refused, "type": "approve"}]}, runtime
    )

    assert _sent() == []
    [row] = _rows(store, event="checkin_send")
    assert row["outcome"] == "refused: the check-in on record was not approved"


def test_a_replayed_send_checkins_does_not_message_the_customer_twice():
    """Chapter 10's membrane rule, applied to the capstone's own effect: a
    crash mid-loop replays the node from the top, and one approval row is
    one send."""
    store = InMemoryStore()
    runtime = _runtime(store)
    state = {
        "drafts": [DRAFT],
        "decisions": [_gated(runtime, checked_decision(DRAFT, APPROVE))],
    }

    send_checkins(state, runtime)
    send_checkins(state, runtime)  # the replay

    assert len(_SLA_TICKETS.sent) == 1
    outcomes = [r["outcome"] for r in _rows(store, event="checkin_send")]
    assert outcomes == ["sent", "replayed: already sent"]
    assert store.get(flagged_ns(), "T-2001").value["status"] == "sent"


def test_a_second_thread_cannot_check_in_on_a_ticket_already_sent():
    """A claim lapsed, a later scan re-drafted the ticket, and both get
    approved: one check-in per ticket all the same."""
    store = InMemoryStore()
    first, second = _runtime(store, "th-1"), _runtime(store, "th-2")
    edit = {"type": "edit", "ticket_id": "T-2001", "edited_message": "T-2001, again"}
    one = _gated(first, checked_decision(DRAFT, APPROVE))
    two = _gated(second, checked_decision(DRAFT, edit))

    send_checkins({"drafts": [DRAFT], "decisions": [one]}, first)
    send_checkins({"drafts": [DRAFT], "decisions": [two]}, second)

    assert _sent() == [("T-2001", "original T-2001")]
    outcomes = [r["outcome"] for r in _rows(store, event="checkin_send")]
    assert outcomes == ["sent", "refused: T-2001 already had a check-in"]


# --- the compiled graph: a real suspend/resume cycle ------------------------


def test_the_graph_suspends_at_approval_gate_with_the_real_drafts():
    result = build_sla_watch_graph().invoke({}, _config("sla-watch-suspend"))

    payload = result["__interrupt__"][0].value
    assert payload["action"] == "sla_checkins"
    assert payload["drafts"][0]["ticket_id"] == "T-2001"


def test_approving_on_resume_sends_the_checkin_and_flags_the_ticket():
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-approve")

    result = graph.invoke(Command(resume=[APPROVE]), config)

    assert result["decisions"][0]["type"] == "approve"
    assert _sent() == [("T-2001", TEMPLATE)]
    assert store.get(flagged_ns(), "T-2001").value["status"] == "sent"


def test_an_edit_on_resume_sends_the_edited_message():
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-edit")

    edit = {"type": "edit", "ticket_id": "T-2001", "edited_message": "Still T-2001."}
    graph.invoke(Command(resume=[edit]), config)

    assert _sent() == [("T-2001", "Still T-2001.")]


def test_rejecting_releases_the_claim_so_the_ticket_resurfaces_next_scan():
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-reject")

    graph.invoke(Command(resume=[REJECT]), config)

    assert _sent() == []
    assert store.get(flagged_ns(), "T-2001") is None
    again = graph.invoke({}, _config("sla-watch-reject-rescan"))
    assert again["__interrupt__"][0].value["drafts"][0]["ticket_id"] == "T-2001"


@pytest.mark.parametrize(
    "resume",
    [
        [{"type": "rejected", "ticket_id": "T-2001"}],  # a typo: used to send
        [{"type": "edit", "ticket_id": "T-2001", "edited_message": ""}],
        [{"type": "edit", "ticket_id": "T-2001", "edited_message": "Hi there."}],
        [{"type": "approve", "ticket_id": "T-2002"}],  # echoes another ticket
        [{"type": "approve"}],  # echoes nothing
        [],  # a missing decision: used to strand the ticket
        [APPROVE, APPROVE],  # one too many
        APPROVE,  # a dict, not a list: used to raise TypeError
    ],
)
def test_a_malformed_decision_sends_nothing_and_releases_the_ticket(resume):
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-malformed")

    result = graph.invoke(Command(resume=resume), config)

    assert result["decisions"][0]["type"] == "refused"
    assert _sent() == []
    assert store.get(flagged_ns(), "T-2001") is None
    assert graph.get_state(config).next == ()


def test_drafts_rewritten_while_paused_no_longer_match_the_echo():
    """The gate re-runs on resume against the drafts in state now. A draft
    swapped while the run was paused (here, an approver's own state write)
    no longer matches the ticket the reviewer echoed."""
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-swapped")
    swapped = {"ticket_id": "T-2002", "customer_id": "C-3", "message": "T-2002 hi"}
    graph.update_state(config, {"drafts": [swapped]})

    result = graph.invoke(Command(resume=[APPROVE]), config)

    reason = result["decisions"][0]["reason"]
    assert reason == "refused: the decision does not echo T-2002"
    assert _sent() == []


def test_an_empty_scan_ends_without_asking_anyone_to_approve_nothing():
    store = InMemoryStore()
    store.put(flagged_ns(), "T-2001", {"status": "sent", "thread_id": "th-0"})
    graph = build_sla_watch_graph(store=store)
    config = _config("sla-watch-empty")

    result = graph.invoke({}, config)

    assert "__interrupt__" not in result
    assert result == {"at_risk": []}
    assert graph.get_state(config).next == ()


def test_a_second_scan_on_the_same_store_skips_an_already_flagged_ticket():
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-dedup-1")
    graph.invoke(Command(resume=[APPROVE]), config)

    second = graph.invoke({}, _config("sla-watch-dedup-2"))

    assert "__interrupt__" not in second
    assert len(_SLA_TICKETS.sent) == 1


def test_the_graph_is_named_so_its_root_run_is_not_langgraph():
    assert build_sla_watch_graph().name == "sla-watch"
    assert build_served_sla_watch().name == "sla-watch"


# --- the audit rows (Chapter 23's record_approval) --------------------------


def test_every_checked_decision_is_an_append_only_audit_row():
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-audit")

    graph.invoke(Command(resume=[{**APPROVE, "by": "lead@local"}]), config)

    [row] = _rows(store)
    assert row["event"] == "checkin"
    assert row["decision"] == "approve"
    assert (row["ticket"], row["customer_id"]) == ("T-2001", "C-2")
    assert row["key"] == checkin_key("T-2001", TEMPLATE)
    assert row["message_sha256"] == message_digest(TEMPLATE)
    assert row["by"] == "lead@local"
    assert row["thread"] == "sla-watch-audit"
    [send] = _rows(store, event="checkin_send")
    assert send["outcome"] == "sent"

    # a later scan's refusal is a second row, never an overwrite
    store.delete(flagged_ns(), "T-2001")
    graph.invoke({}, _config("sla-watch-audit-2"))
    graph.invoke(
        Command(resume=[{"type": "rejected", "ticket_id": "T-2001"}]),
        _config("sla-watch-audit-2"),
    )

    rows = _rows(store)
    assert [r["decision"] for r in rows] == ["approve", "refused"]
    assert rows[1]["refused"] == "refused: unknown decision 'rejected'"
    assert rows[1]["key"] is None


def test_in_process_a_state_write_as_the_gate_sends_nothing():
    """The in-process build has a store, so it verifies the row too."""
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-forged")
    forged = {"type": "approve", "message": "Click http://evil.example T-2001"}
    graph.update_state(config, {"decisions": [forged]}, as_node="approval_gate")

    graph.invoke(None, config)

    assert _sent() == []
    [row] = _rows(store, event="checkin_send")
    assert row["outcome"] == "refused: no check-in approval on record"


# --- the served build: the approver is the authenticated identity ----------


def _user(identity: str, role: str) -> SimpleNamespace:
    return SimpleNamespace(identity=identity, permissions=[f"role:{role}"])


AGENT = _user("agent-7", "support_agent")
LEAD = _user("lead-3", "support_lead")


def _served(store: InMemoryStore):
    """The served graph with a checkpointer and store of its own, so a test
    can drive it (the Agent Server supplies both)."""
    return build_served_sla_watch().copy(
        update={"checkpointer": InMemorySaver(), "store": store}
    )


def test_the_served_build_brings_no_checkpointer_and_no_store():
    graph = build_served_sla_watch()

    assert graph.checkpointer is None
    assert graph.store is None


def test_a_support_lead_approves_and_the_row_names_the_proved_identity():
    store = InMemoryStore()
    graph = _served(store)
    lead = _config("served-lead", LEAD)

    graph.invoke({}, lead)
    graph.invoke(Command(resume=[{**APPROVE, "by": "someone-else"}]), lead)

    assert [t for t, _ in _sent()] == ["T-2001"]
    [row] = _rows(store)
    assert (row["by"], row["role"], row["decision"]) == (
        "lead-3",
        "support_lead",
        "approve",
    )
    [send] = _rows(store, event="checkin_send")
    assert (send["by"], send["outcome"]) == ("lead-3", "sent")


@pytest.mark.parametrize(
    "user, reason",
    [
        (AGENT, "refused: agent-7 may not approve check-ins"),
        (None, "refused: no authenticated approver"),
    ],
)
def test_on_the_served_build_a_non_approver_cannot_send(user, reason):
    store = InMemoryStore()
    graph = _served(store)
    config = _config("served-refused", user)

    graph.invoke({}, config)
    result = graph.invoke(Command(resume=[APPROVE]), config)

    by = user.identity if user else None
    decision = dict(result["decisions"][0])
    assert decision.pop("audit_key")
    assert decision == {"type": "refused", "reason": reason, "by": by}
    assert _sent() == []
    assert store.get(flagged_ns(), "T-2001") is None
    [row] = _rows(store)
    assert row["refused"] == reason


def test_an_agent_starts_and_a_lead_resumes():
    """The served flow: the run is started by one identity (a cron's owner)
    and resumed by the reviewer. The gate and the send read the resumer."""
    store = InMemoryStore()
    graph = _served(store)
    graph.invoke({}, _config("agent-then-lead", AGENT))

    graph.invoke(Command(resume=[APPROVE]), _config("agent-then-lead", LEAD))

    assert _sent() == [("T-2001", TEMPLATE)]
    [row] = _rows(store)
    assert (row["by"], row["role"]) == ("lead-3", "support_lead")


def test_a_lead_starts_and_an_agent_resumes():
    store = InMemoryStore()
    graph = _served(store)
    graph.invoke({}, _config("lead-then-agent", LEAD))

    result = graph.invoke(Command(resume=[APPROVE]), _config("lead-then-agent", AGENT))

    reason = result["decisions"][0]["reason"]
    assert reason == "refused: agent-7 may not approve check-ins"
    assert _sent() == []
    assert store.get(flagged_ns(), "T-2001") is None  # released for the next scan


@pytest.mark.parametrize("resumer", [AGENT, LEAD])
def test_a_state_write_as_the_gate_sends_nothing_on_the_served_build(resumer):
    """The R117 bypass: agent-7 writes a draft and an approval into state as
    if the gate had produced them, then runs the graph on. Whoever runs it,
    nothing is sent, because no gate row stands behind the decision, and
    the attempt is itself an audit row."""
    store = InMemoryStore()
    graph = _served(store)
    agent = _config("forged", AGENT)
    graph.invoke({}, agent)
    forged = {
        "drafts": [{"ticket_id": "T-2002", "customer_id": "C-3", "message": "x"}],
        "decisions": [{"type": "approve", "message": "Click http://evil.example"}],
    }
    graph.update_state(agent, forged, as_node="approval_gate")

    graph.invoke(None, _config("forged", resumer))

    assert _sent() == []
    [row] = _rows(store, "C-3", event="checkin_send")
    assert row["ticket_id"] == "T-2002"
    assert row["outcome"] == (
        "refused: agent-7 may not send check-ins"
        if resumer is AGENT
        else "refused: no check-in approval on record"
    )


def test_a_lead_cannot_spend_another_leads_approval_row():
    """A planted key naming a real lead approval on this thread, run on by a
    different approver, is refused: the row must be the resumer's."""
    store = InMemoryStore()
    graph = _served(store)
    lead = _config("two-leads", LEAD)
    graph.invoke({}, lead)
    graph.invoke(Command(resume=[APPROVE]), lead, interrupt_before=["send_checkins"])
    assert _sent() == []  # stopped before the send

    graph.invoke(None, _config("two-leads", _user("lead-9", "support_lead")))

    assert _sent() == []
    [send] = _rows(store, event="checkin_send")
    assert send["outcome"] == "refused: the approval on record is not this approver's"


def test_the_only_way_into_send_checkins_is_through_the_gate():
    """The chapter's pitfall, guarded at the topology level; the row check
    above guards it when state is written from outside."""
    graph = build_sla_watch_graph().get_graph()
    into_send = {e.source for e in graph.edges if e.target == "send_checkins"}

    assert into_send == {"approval_gate"}
