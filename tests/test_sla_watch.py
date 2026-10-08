"""Chapter 27, "Capstone" - atlas/sla_watch.py.

See "Building SLA Watch by reuse, not rebuild". `scan_tickets`,
`draft_checkins` and `send_checkins` need no model or LangSmith connection,
so they are unit-tested directly with a hand-built `Runtime`, the convention
`tests/test_graph.py` and `tests/test_memory.py` use for Chapter 13's
store-backed nodes. The gate is exercised through the compiled graph with a
real `interrupt()`/`Command(resume=...)` cycle, never a monkeypatched
`interrupt`: a decision that is unknown, missing, or an out-of-policy edit
sends nothing and releases the claim; an empty scan ends without pausing;
on the served build the approver is the authenticated identity; every
checked decision is an append-only audit row."""

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from atlas.audit import audit_ns
from atlas.sla_watch import (
    build_served_sla_watch,
    build_sla_watch_graph,
    checked_decision,
    checkin_key,
    checkin_refusal,
    compose_checkin,
    draft_checkins,
    flagged_ns,
    per_draft,
    scan_tickets,
    send_checkins,
)
from atlas.tools import _SLA_TICKETS

NO_DECISION = {"type": "refused", "reason": "refused: no decision for this draft"}
T2001 = {"ticket_id": "T-2001", "customer_id": "C-2", "hours_open": 30}
DRAFT = {"ticket_id": "T-2001", "customer_id": "C-2", "message": "original T-2001"}


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


def _sent() -> list[tuple[str, str]]:
    return [(m["ticket_id"], m["message"]) for m in _SLA_TICKETS.sent]


def _rows(store: InMemoryStore, customer_id: str = "C-2") -> list[dict]:
    rows = store.search(audit_ns(customer_id), limit=100)
    return sorted((r.value for r in rows), key=lambda v: v["at"])


def _paused(store: InMemoryStore, thread_id: str) -> tuple[object, dict]:
    graph = build_sla_watch_graph(store=store)
    config = _config(thread_id)
    graph.invoke({}, config)
    return graph, config


# --- flagged_ns / compose_checkin -------------------------------------------


def test_flagged_ns_is_its_own_namespace_on_the_shared_store():
    assert flagged_ns() == ("sla_watch", "flagged")


def test_compose_checkin_mentions_the_tickets_own_id():
    assert "T-2001" in compose_checkin({"ticket_id": "T-2001"})


def test_the_template_draft_passes_the_edit_policy():
    """The policy an edit must meet is one the shipped template meets."""
    assert checkin_refusal("T-2001", compose_checkin({"ticket_id": "T-2001"})) is None


# --- scan_tickets: the memory-horizon check ---------------------------------


def test_scan_tickets_surfaces_the_seeded_at_risk_ticket():
    result = scan_tickets({}, Runtime(store=InMemoryStore()))

    assert result == {"at_risk": [T2001]}


def test_scan_tickets_skips_a_ticket_already_flagged_in_the_store():
    store = InMemoryStore()
    store.put(flagged_ns(), "T-2001", {"status": "drafted"})

    assert scan_tickets({}, Runtime(store=store)) == {"at_risk": []}


# --- draft_checkins ----------------------------------------------------------


def test_draft_checkins_builds_one_draft_per_at_risk_ticket():
    result = draft_checkins({"at_risk": [T2001]}, Runtime(store=InMemoryStore()))

    [draft] = result["drafts"]
    assert draft["ticket_id"] == "T-2001"
    assert draft["customer_id"] == "C-2"
    assert "T-2001" in draft["message"]


def test_draft_checkins_claims_the_ticket_so_a_parked_approval_is_not_redrafted():
    """The flag is written at draft time, not send time: the gate can sit for
    days while the scan runs hourly."""
    store = InMemoryStore()

    draft_checkins({"at_risk": [T2001]}, Runtime(store=store))

    assert store.get(flagged_ns(), "T-2001").value == {"status": "drafted"}


# --- checked_decision / per_draft / checkin_refusal -------------------------


@pytest.mark.parametrize(
    "decision, expected",
    [
        ({"type": "approve"}, {"type": "approve", "message": "original T-2001"}),
        (
            {"type": "edit", "edited_message": "Still on T-2001, sorry."},
            {"type": "edit", "message": "Still on T-2001, sorry."},
        ),
        ({"type": "reject"}, {"type": "reject"}),
        (
            {"type": "rejected"},
            {"type": "refused", "reason": "refused: unknown decision 'rejected'"},
        ),
        ({}, {"type": "refused", "reason": "refused: unknown decision None"}),
        (None, NO_DECISION),
        ("approve", NO_DECISION),
    ],
)
def test_checked_decision_sends_only_a_known_valid_decision(decision, expected):
    checked = checked_decision(DRAFT, decision)
    checked.pop("by", None)

    assert checked == expected


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
    decision = {"type": "edit", "edited_message": message}

    assert checked_decision(DRAFT, decision)["reason"] == reason


def test_per_draft_accepts_only_one_decision_per_draft():
    approve = {"type": "approve"}

    assert per_draft([approve, approve], 2) == [approve, approve]
    assert per_draft([approve], 2) == [None, None]  # short: could be shifted
    assert per_draft([approve] * 3, 2) == [None, None]
    assert per_draft(approve, 1) == [None]  # a dict, not a list


# --- send_checkins: sends only approved or edited drafts --------------------


def test_send_checkins_sends_an_approved_draft_and_flags_it():
    store = InMemoryStore()
    state = {
        "drafts": [DRAFT],
        "decisions": [{"type": "approve", "message": "original T-2001"}],
    }

    send_checkins(state, Runtime(store=store))

    assert _sent() == [("T-2001", "original T-2001")]
    assert store.get(flagged_ns(), "T-2001").value == {"status": "sent"}


@pytest.mark.parametrize(
    "decision",
    [{"type": "reject"}, {"type": "refused", "reason": "x"}, {"type": "rejected"}, {}],
)
def test_send_checkins_sends_nothing_and_releases_the_claim_otherwise(decision):
    store = InMemoryStore()
    store.put(flagged_ns(), "T-2001", {"status": "drafted"})

    send_checkins({"drafts": [DRAFT], "decisions": [decision]}, Runtime(store=store))

    assert _sent() == []
    assert store.get(flagged_ns(), "T-2001") is None


def test_a_replayed_send_checkins_does_not_message_the_customer_twice():
    """Chapter 10's membrane rule, applied to the capstone's own effect: a
    crash mid-loop replays the node from the top, and the stable key makes
    the second pass a no-op."""
    state = {
        "drafts": [DRAFT],
        "decisions": [{"type": "approve", "message": "original T-2001"}],
    }
    runtime = Runtime(store=InMemoryStore())

    send_checkins(state, runtime)
    send_checkins(state, runtime)  # the replay

    assert len(_SLA_TICKETS.sent) == 1


# --- the compiled graph: a real suspend/resume cycle ------------------------


def test_the_graph_suspends_at_approval_gate_with_the_real_drafts():
    result = build_sla_watch_graph().invoke({}, _config("sla-watch-suspend"))

    payload = result["__interrupt__"][0].value
    assert payload["action"] == "sla_checkins"
    assert payload["drafts"][0]["ticket_id"] == "T-2001"


def test_approving_on_resume_sends_the_checkin_and_flags_the_ticket():
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-approve")

    result = graph.invoke(Command(resume=[{"type": "approve"}]), config)

    assert result["decisions"][0]["type"] == "approve"
    assert _sent() == [("T-2001", compose_checkin({"ticket_id": "T-2001"}))]
    assert store.get(flagged_ns(), "T-2001").value == {"status": "sent"}


def test_an_edit_on_resume_sends_the_edited_message():
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-edit")

    edit = {"type": "edit", "edited_message": "Still working on T-2001."}
    graph.invoke(Command(resume=[edit]), config)

    assert _sent() == [("T-2001", "Still working on T-2001.")]


def test_rejecting_releases_the_claim_so_the_ticket_resurfaces_next_scan():
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-reject")

    graph.invoke(Command(resume=[{"type": "reject"}]), config)

    assert _sent() == []
    assert store.get(flagged_ns(), "T-2001") is None
    again = graph.invoke({}, _config("sla-watch-reject-rescan"))
    assert again["__interrupt__"][0].value["drafts"][0]["ticket_id"] == "T-2001"


@pytest.mark.parametrize(
    "resume",
    [
        [{"type": "rejected"}],  # a typo: used to send
        [{"type": "edit", "edited_message": ""}],  # used to send an empty message
        [{"type": "edit", "edited_message": "Hi there."}],  # names no ticket
        [],  # a missing decision: used to strand the ticket
        [{"type": "approve"}, {"type": "approve"}],  # one too many
        {"type": "approve"},  # a dict, not a list: used to raise TypeError
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


def test_an_empty_scan_ends_without_asking_anyone_to_approve_nothing():
    store = InMemoryStore()
    store.put(flagged_ns(), "T-2001", {"status": "sent"})
    graph = build_sla_watch_graph(store=store)
    config = _config("sla-watch-empty")

    result = graph.invoke({}, config)

    assert "__interrupt__" not in result
    assert result == {"at_risk": []}
    assert graph.get_state(config).next == ()


def test_a_second_scan_on_the_same_store_skips_an_already_flagged_ticket():
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-dedup-1")
    graph.invoke(Command(resume=[{"type": "approve"}]), config)

    second = graph.invoke({}, _config("sla-watch-dedup-2"))

    assert "__interrupt__" not in second
    assert len(_SLA_TICKETS.sent) == 1


def test_the_graph_is_named_so_its_root_run_is_not_langgraph():
    assert build_sla_watch_graph().name == "sla-watch"
    assert build_served_sla_watch().name == "sla-watch"


# --- the audit rows (Chapter 23's record_approval, event "checkin") --------


def test_every_checked_decision_is_an_append_only_audit_row():
    store = InMemoryStore()
    graph, config = _paused(store, "sla-watch-audit")

    graph.invoke(Command(resume=[{"type": "approve", "by": "lead@local"}]), config)

    [row] = _rows(store)
    assert row["event"] == "checkin"
    assert row["decision"] == "approve"
    assert (row["ticket"], row["customer_id"]) == ("T-2001", "C-2")
    assert row["key"] == checkin_key("T-2001")
    assert row["by"] == "lead@local"
    assert row["thread"] == "sla-watch-audit"

    # a later scan's refusal is a second row, never an overwrite
    store.delete(flagged_ns(), "T-2001")
    graph.invoke({}, _config("sla-watch-audit-2"))
    graph.invoke(Command(resume=[{"type": "rejected"}]), _config("sla-watch-audit-2"))

    rows = _rows(store)
    assert [r["decision"] for r in rows] == ["approve", "refused"]
    assert rows[1]["refused"] == "refused: unknown decision 'rejected'"
    assert rows[1]["key"] is None


# --- the served build: the approver is the authenticated identity ----------


def _user(identity: str, role: str) -> SimpleNamespace:
    return SimpleNamespace(identity=identity, permissions=[f"role:{role}"])


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
    lead = _config("served-lead", _user("lead-3", "support_lead"))

    graph.invoke({}, lead)
    graph.invoke(Command(resume=[{"type": "approve", "by": "someone-else"}]), lead)

    assert [t for t, _ in _sent()] == ["T-2001"]
    [row] = _rows(store)
    assert (row["by"], row["role"], row["decision"]) == (
        "lead-3",
        "support_lead",
        "approve",
    )


@pytest.mark.parametrize(
    "user, reason",
    [
        (
            _user("agent-7", "support_agent"),
            "refused: agent-7 may not approve check-ins",
        ),
        (None, "refused: no authenticated approver"),
    ],
)
def test_on_the_served_build_a_non_approver_cannot_send(user, reason):
    store = InMemoryStore()
    graph = _served(store)
    config = _config("served-refused", user)

    graph.invoke({}, config)
    result = graph.invoke(Command(resume=[{"type": "approve"}]), config)

    by = user.identity if user else None
    assert result["decisions"][0] == {"type": "refused", "reason": reason, "by": by}
    assert _sent() == []
    assert store.get(flagged_ns(), "T-2001") is None
    [row] = _rows(store)
    assert row["refused"] == reason


def test_the_only_way_into_send_checkins_is_through_the_gate():
    """The chapter's pitfall, guarded at the topology level."""
    graph = build_sla_watch_graph().get_graph()
    into_send = {e.source for e in graph.edges if e.target == "send_checkins"}

    assert into_send == {"approval_gate"}
