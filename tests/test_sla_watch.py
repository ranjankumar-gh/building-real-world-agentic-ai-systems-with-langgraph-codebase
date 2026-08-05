"""Chapter 27, "Capstone" - atlas/sla_watch.py.

See "Building SLA Watch by reuse, not rebuild". `scan_tickets`/
`draft_checkins`/`send_checkins` need no live model or LangSmith
connection - they are unit-tested directly below, the same
hand-built-`Runtime` convention `tests/test_graph.py` and
`tests/test_memory.py` already use for Chapter 13's store-backed nodes.
`build_sla_watch_graph` is exercised end to end with a REAL
`interrupt()`/`Command(resume=...)` cycle - no monkeypatched `interrupt` -
the same "prove the suspension is real" discipline Chapter 11's own
end-to-end test used."""

import pytest
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from atlas.sla_watch import (
    approval_gate,
    build_sla_watch_graph,
    compose_checkin,
    draft_checkins,
    flagged_ns,
    scan_tickets,
    send_checkins,
)
from atlas.tools import _SLA_TICKETS


@pytest.fixture(autouse=True)
def _clean_send_ledger():
    """`send_checkin` is idempotent by key against a process-global backend,
    which is what makes it survive a node replay - and what would otherwise
    let one test's send suppress the next test's identical ticket id."""
    _SLA_TICKETS.reset()
    yield
    _SLA_TICKETS.reset()


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


# --- flagged_ns / compose_checkin -------------------------------------------


def test_flagged_ns_is_its_own_namespace_on_the_shared_store():
    assert flagged_ns() == ("sla_watch", "flagged")


def test_compose_checkin_mentions_the_tickets_own_id():
    message = compose_checkin({"ticket_id": "T-2001"})

    assert "T-2001" in message


# --- scan_tickets: the memory-horizon check ---------------------------------


def test_scan_tickets_surfaces_the_seeded_at_risk_ticket():
    store = InMemoryStore()
    runtime = Runtime(store=store)

    result = scan_tickets({}, runtime)

    assert result == {"at_risk": [{"ticket_id": "T-2001", "hours_open": 30}]}


def test_scan_tickets_skips_a_ticket_already_flagged_in_the_store():
    store = InMemoryStore()
    store.put(flagged_ns(), "T-2001", {"sent_at": "..."})
    runtime = Runtime(store=store)

    result = scan_tickets({}, runtime)

    assert result == {"at_risk": []}


# --- draft_checkins ----------------------------------------------------------


def test_draft_checkins_builds_one_draft_per_at_risk_ticket():
    state = {"at_risk": [{"ticket_id": "T-2001", "hours_open": 30}]}

    result = draft_checkins(state, Runtime(store=InMemoryStore()))

    assert len(result["drafts"]) == 1
    assert result["drafts"][0]["ticket_id"] == "T-2001"
    assert "T-2001" in result["drafts"][0]["message"]


def test_draft_checkins_drafts_nothing_when_nothing_is_at_risk():
    runtime = Runtime(store=InMemoryStore())

    assert draft_checkins({"at_risk": []}, runtime) == {"drafts": []}


def test_draft_checkins_claims_the_ticket_so_a_parked_approval_is_not_redrafted():
    """The flag is written at DRAFT time, not send time. An approval gate can
    sit for days (Chapter 11) while the scan runs hourly - flagging only on
    send would leave a parked ticket unflagged, so every subsequent scan would
    re-draft it and open another interrupt for the same customer."""
    store = InMemoryStore()
    state = {"at_risk": [{"ticket_id": "T-2001", "hours_open": 30}]}

    draft_checkins(state, Runtime(store=store))

    assert store.get(flagged_ns(), "T-2001") is not None


# --- approval_gate: interrupts with the drafts, returns the decisions ------


def test_approval_gate_surfaces_the_drafts_and_returns_the_human_decisions(
    monkeypatch,
):
    import atlas.sla_watch as sla_watch_module

    seen_payloads = []
    monkeypatch.setattr(
        sla_watch_module,
        "interrupt",
        lambda payload: seen_payloads.append(payload) or [{"type": "approve"}],
    )
    drafts = [{"ticket_id": "T-2001", "message": "hi"}]

    result = approval_gate({"drafts": drafts})

    assert seen_payloads == [{"action": "sla_checkins", "drafts": drafts}]
    assert result == {"decisions": [{"type": "approve"}]}


# --- send_checkins: never sends for a rejected draft, re-validates edits --


def test_send_checkins_sends_an_approved_draft_and_flags_it():
    store = InMemoryStore()
    runtime = Runtime(store=store)
    state = {
        "drafts": [{"ticket_id": "T-2001", "message": "original message"}],
        "decisions": [{"type": "approve"}],
    }
    before = len(_SLA_TICKETS.sent)

    send_checkins(state, runtime)

    assert len(_SLA_TICKETS.sent) == before + 1
    assert _SLA_TICKETS.sent[-1]["ticket_id"] == "T-2001"
    assert _SLA_TICKETS.sent[-1]["message"] == "original message"
    assert store.get(flagged_ns(), "T-2001") is not None


def test_send_checkins_never_sends_for_a_rejected_draft():
    store = InMemoryStore()
    runtime = Runtime(store=store)
    state = {
        "drafts": [{"ticket_id": "T-2002", "message": "original message"}],
        "decisions": [{"type": "reject"}],
    }
    before = len(_SLA_TICKETS.sent)

    send_checkins(state, runtime)

    assert len(_SLA_TICKETS.sent) == before  # nothing sent
    assert store.get(flagged_ns(), "T-2002") is None  # nothing flagged either


def test_send_checkins_sends_the_edited_message_when_one_was_provided():
    store = InMemoryStore()
    runtime = Runtime(store=store)
    state = {
        "drafts": [{"ticket_id": "T-2003", "message": "original message"}],
        "decisions": [{"type": "edit", "edited_message": "a shorter check-in"}],
    }

    send_checkins(state, runtime)

    assert _SLA_TICKETS.sent[-1]["message"] == "a shorter check-in"


# --- build_sla_watch_graph: the real, compiled suspend/resume cycle --------


def test_the_graph_suspends_at_approval_gate_with_the_real_drafts():
    graph = build_sla_watch_graph()

    result = graph.invoke({}, _config("sla-watch-suspend"))

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["action"] == "sla_checkins"
    assert payload["drafts"][0]["ticket_id"] == "T-2001"


def test_approving_on_resume_sends_the_checkin_and_flags_the_ticket():
    store = InMemoryStore()
    graph = build_sla_watch_graph(store=store)
    config = _config("sla-watch-approve")

    graph.invoke({}, config)
    result = graph.invoke(Command(resume=[{"type": "approve"}]), config)

    assert result["decisions"] == [{"type": "approve"}]
    assert store.get(flagged_ns(), "T-2001") is not None


def test_rejecting_on_resume_never_flags_the_ticket_so_it_resurfaces_next_scan():
    """The callout's own claim, exercised for real: skipping approval_gate
    is the temptation, but here the gate ran, rejected, and the ticket is
    NOT flagged - so a fresh scan (a new thread on the SAME store) surfaces
    it again instead of silently dropping it."""
    store = InMemoryStore()
    graph = build_sla_watch_graph(store=store)
    config = _config("sla-watch-reject")

    graph.invoke({}, config)
    graph.invoke(Command(resume=[{"type": "reject"}]), config)

    assert store.get(flagged_ns(), "T-2001") is None

    second_run = graph.invoke({}, _config("sla-watch-reject-rescan"))
    assert second_run["__interrupt__"][0].value["drafts"][0]["ticket_id"] == "T-2001"


def test_a_second_scan_on_the_same_store_skips_an_already_flagged_ticket():
    """The flagged ticket never re-enters the batch on the second scan -
    `approval_gate` still interrupts unconditionally (it has no "nothing to
    review" guard), but this time the surfaced payload's own `drafts` list
    is empty, and resuming with an empty decisions list completes cleanly."""
    store = InMemoryStore()
    graph = build_sla_watch_graph(store=store)
    first = _config("sla-watch-dedup-1")

    graph.invoke({}, first)
    graph.invoke(Command(resume=[{"type": "approve"}]), first)

    second = _config("sla-watch-dedup-2")
    suspended = graph.invoke({}, second)

    assert suspended["__interrupt__"][0].value == {"action": "sla_checkins", "drafts": []}

    result = graph.invoke(Command(resume=[]), second)

    assert result["at_risk"] == []
    assert result["drafts"] == []
    assert result["decisions"] == []


def test_a_replayed_send_checkins_does_not_message_the_customer_twice():
    """The Chapter 10 membrane rule, applied to the capstone's own effect.
    `send_checkins` performs N irreversible customer-facing sends in a loop;
    a crash mid-loop replays the node from the top on resume. The stable
    `checkin_key` is what makes the second pass a no-op instead of a second
    message to an already-frustrated customer."""
    store = InMemoryStore()
    runtime = Runtime(store=store)
    state = {
        "drafts": [{"ticket_id": "T-2001", "message": "original message"}],
        "decisions": [{"type": "approve"}],
    }
    before = len(_SLA_TICKETS.sent)

    send_checkins(state, runtime)
    send_checkins(state, runtime)  # the replay

    assert len(_SLA_TICKETS.sent) == before + 1
