"""Chapter 23, "Security, Privacy, Cost, and Governance" - atlas/audit.py.

See "A durable audit log, deliberately separate from the trace".
`AuditGate.wrap_tool_call` needs no live model call or LangSmith connection
to test - it only writes a durable record to a `BaseStore` after the real
handler runs, so an `InMemoryStore` plus a hand-built `ToolCallRequest`
(the `test_middleware.py`/`test_security.py` convention) is enough to
exercise it directly."""

import asyncio
from types import SimpleNamespace

import pytest

from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from atlas import graph as graph_module
from atlas.audit import AuditGate, audit_ns, record_approval
from atlas.graph import _make_builder, graph
from atlas.security import AtlasContext


def _request(name: str, args: dict, role: str, customer_id: str) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "call-1"},
        tool=None,
        state=None,
        runtime=Runtime(context=AtlasContext(role=role, customer_id=customer_id)),
    )


def test_audit_gate_records_a_successful_tool_call_after_it_runs():
    store = InMemoryStore()
    gate = AuditGate(store)
    request = _request(
        "set_ticket_status",
        {"ticket_id": "T-1001", "status": "pending"},
        role="support_agent",
        customer_id="C-1",
    )

    def handler(_req):
        return ToolMessage("Ticket T-1001 set to pending.", tool_call_id="call-1")

    result = gate.wrap_tool_call(request, handler)

    assert result.content == "Ticket T-1001 set to pending."
    record = store.get(("audit", "C-1"), "call-1")
    assert record.value["tool"] == "set_ticket_status"
    assert record.value["args"] == {"ticket_id": "T-1001", "status": "pending"}
    assert record.value["role"] == "support_agent"
    assert record.value["result_status"] == "success"
    assert "recorded_at" in record.value


def test_audit_gate_records_an_error_result_status_too():
    store = InMemoryStore()
    gate = AuditGate(store)
    request = _request(
        "set_ticket_status",
        {"ticket_id": "T-1001", "status": "resolved"},
        role="support_agent",
        customer_id="C-1",
    )

    def handler(_req):
        return ToolMessage(
            "Blocked: marking a ticket resolved requires human approval.",
            tool_call_id="call-1",
            status="error",
        )

    gate.wrap_tool_call(request, handler)

    record = store.get(("audit", "C-1"), "call-1")
    assert record.value["result_status"] == "error"


def test_audit_gate_keeps_customers_in_separate_namespaces():
    store = InMemoryStore()
    gate = AuditGate(store)

    gate.wrap_tool_call(
        _request("search_kb", {"query": "refund"}, role="support_agent", customer_id="C-1"),
        lambda _req: ToolMessage("30 days", tool_call_id="call-1"),
    )

    assert store.get(("audit", "C-1"), "call-1") is not None
    assert store.get(("audit", "C-2"), "call-1") is None


def test_audit_gate_writes_a_durable_record_never_bypassing_the_handler():
    """The audit write happens AFTER handler(request) - never a substitute
    for actually running the tool."""
    store = InMemoryStore()
    gate = AuditGate(store)
    called = []

    def handler(_req):
        called.append(True)
        return ToolMessage("ran", tool_call_id="call-1")

    gate.wrap_tool_call(
        _request("lookup_ticket", {"ticket_id": "T-1001"}, role="support_agent", customer_id="C-1"),
        handler,
    )

    assert called == [True]


# --- The graph's store, the async twin, and a Command result ---------------

def test_audit_gate_writes_to_the_graphs_store_not_its_own():
    """The store the graph was compiled with wins: an auditor reading
    `graph.store` sees the rows. The constructor's store is only a fallback."""
    own, graphs = InMemoryStore(), InMemoryStore()
    request = _request("search_kb", {"query": "refund"}, "support_agent", "C-1")
    request = request.override(
        runtime=Runtime(context=request.runtime.context, store=graphs)
    )

    AuditGate(own).wrap_tool_call(
        request, lambda _req: ToolMessage("30 days", tool_call_id="call-1")
    )

    assert graphs.get(audit_ns("C-1"), "call-1") is not None
    assert own.get(audit_ns("C-1"), "call-1") is None


def test_audit_gate_async_twin_records_the_same_row():
    store = InMemoryStore()

    async def handler(_req):
        return ToolMessage("ran", tool_call_id="call-1", status="error")

    asyncio.run(
        AuditGate(store).awrap_tool_call(
            _request("lookup_ticket", {"ticket_id": "T-1"}, "support_agent", "C-1"),
            handler,
        )
    )

    assert store.get(audit_ns("C-1"), "call-1").value["result_status"] == "error"


def test_audit_gate_records_a_command_result_without_failing():
    store = InMemoryStore()

    AuditGate(store).wrap_tool_call(
        _request("lookup_ticket", {"ticket_id": "T-1"}, "support_agent", "C-1"),
        lambda _req: Command(update={}),
    )

    assert store.get(audit_ns("C-1"), "call-1").value["result_status"] == "command"


def test_record_approval_is_append_only():
    """Every write gets its own key - no read first, nothing overwritten.
    The same decision recorded twice is two rows: a re-executed gate logs
    its own."""
    store = InMemoryStore()
    first = {"decision": "approve", "by": "lead-3", "at": "t0", "amount": 49.0}

    k1 = record_approval(store, "C-1", "t-1", "cp-1", {**first, "ticket_id": "T-9"})
    k2 = record_approval(store, "C-1", "t-1", "cp-1", {**first, "ticket_id": "T-9"})

    assert k1 != k2
    assert k1.startswith("approval:t-1:cp-1:")
    assert k2.startswith("approval:t-1:cp-1:")
    for key in (k1, k2):
        assert store.get(audit_ns("C-1"), key).value == {
            **first,
            "ticket_id": "T-9",
            "event": "approval",
            "ticket": "T-9",
            "thread": "t-1",
        }


def _approval_rows(customer_id: str) -> list:
    rows = graph.store.search(audit_ns(customer_id), limit=100)
    return [r for r in rows if r.value.get("event") == "approval"]


def _refund_thread(monkeypatch, thread_id: str, customer_id: str, **conf) -> dict:
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: SimpleNamespace(route="refund")
    )
    config = {"configurable": {"thread_id": thread_id, **conf}}
    graph.invoke(
        {
            "messages": [{"role": "user", "content": "refund please"}],
            "ticket": {"id": "T-1001", "amount": 49.0, "customer_id": customer_id},
        },
        config,
    )
    return config


def test_the_approval_gate_writes_its_decision_to_the_audit_namespace(monkeypatch):
    """End to end through the compiled graph, in process: suspend at the
    gate, resume with an approval, and find the decision - with the payload's
    `by` and the amount charged - in the graph's own store, after the human
    acted, never before."""
    config = _refund_thread(monkeypatch, "audit-approval-1", "C-77")
    assert _approval_rows("C-77") == []

    graph.invoke(Command(resume={"type": "approve", "by": "lead@support"}), config)

    [row] = _approval_rows("C-77")
    assert row.key.startswith("approval:audit-approval-1:")
    assert row.value["decision"] == "approve"
    assert row.value["by"] == "lead@support"
    assert row.value["amount"] == 49.0
    assert row.value["ticket"] == "T-1001"
    assert row.value["thread"] == "audit-approval-1"
    assert "at" in row.value


def test_an_edited_approval_records_the_amount_actually_charged(monkeypatch):
    config = _refund_thread(monkeypatch, "audit-approval-edit", "C-78")

    graph.invoke(
        Command(resume={"type": "edit", "amount": 20.0, "by": "lead@support"}), config
    )

    [row] = _approval_rows("C-78")
    assert row.value["decision"] == "edit"
    assert row.value["amount"] == 20.0


def test_a_fork_from_the_paused_checkpoint_appends_never_overwrites(monkeypatch):
    """Time travel (Chapter 9): fork the thread from the paused checkpoint
    and decide differently there. The first row is untouched; the fork,
    running from a new checkpoint, adds its own."""
    config = _refund_thread(monkeypatch, "audit-approval-fork", "C-79")
    paused = graph.get_state(config).config

    graph.invoke(Command(resume={"type": "approve", "by": "lead@support"}), config)
    fork = graph.update_state(paused, {"error": None}, as_node="triage")
    graph.invoke(None, fork)  # pauses again at the gate, on the fork
    graph.invoke(Command(resume={"type": "reject", "by": "lead@support"}), fork)

    rows = sorted(_approval_rows("C-79"), key=lambda r: r.key)
    assert [r.value["decision"] for r in rows] == ["approve", "reject"]
    assert rows[0].value["amount"] == 49.0 and rows[1].value["amount"] is None


# --- R94/R101/R104: the served build fails closed, at the gate and the charge


class _User:
    def __init__(self, identity: str, role: str) -> None:
        self.identity = identity
        self.permissions = [f"role:{role}"]


AGENT = _User("agent-7", "support_agent")
LEAD = _User("lead-3", "support_lead")
TICKET = {"id": "T-1001", "amount": 10.0, "customer_id": "C-90"}
OTHER = {"id": "T-2002", "amount": 300.0, "customer_id": "C-91"}
SHOWN = {"ticket_id": "T-1001", "customer_id": "C-90", "amount": 10.0}  # as shown


def approve(**extra: object) -> Command:
    return Command(resume={"type": "approve", "shown": SHOWN, **extra})


@pytest.fixture
def charged(monkeypatch) -> list:
    """Every (ticket, amount) actually charged. The fake keeps the
    provider's cumulative cap per ticket (T-1001 was paid 49, T-2002 was
    paid 300) and its dedupe on the idempotency key."""
    from atlas.effects import RefundRefused

    paid_for = {"T-1001": 49.0, "T-2002": 300.0}
    done: list = []

    ledger: dict[str, str] = {}

    def fake_charge(key: str, ticket_id: str, amount: float) -> str:
        if key in ledger:  # the provider's dedupe, as in atlas/effects.py
            return ledger[key]
        refunded = sum(a for t, a in done if t == ticket_id)  # cumulative cap
        if refunded + amount > paid_for[ticket_id]:
            raise RefundRefused(f"{amount:.2f} exceeds the original")
        done.append((ticket_id, amount))
        ledger[key] = f"Refund of ${amount:.2f} issued for {ticket_id}."
        return ledger[key]

    monkeypatch.setattr(graph_module, "charge_refund", fake_charge)
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: SimpleNamespace(route="refund")
    )
    return done


def _served():
    """The served build's gate and refund wiring, with a checkpointer and a
    store of its own so a test can drive it (the server supplies both)."""
    return _make_builder(graph_module.triage, served=True).compile(
        checkpointer=InMemorySaver(), store=InMemoryStore()
    )


def _cfg(thread_id: str, user: object | None) -> dict:
    conf: dict = {"thread_id": thread_id, "assistant_id": "resolve"}
    if user is not None:
        conf["langgraph_auth_user"] = user
    return {"configurable": conf}


def _start(g, config: dict, ticket: dict = TICKET) -> None:
    g.invoke(
        {"messages": [{"role": "user", "content": "refund please"}], "ticket": ticket},
        config,
    )


def _rows(g, customer_id: str = "C-90", event: str | None = None) -> list[dict]:
    rows = g.store.search(audit_ns(customer_id), limit=100)
    return sorted(
        (
            r.value
            for r in rows
            if r.value.get("event") in ("approval", "refund")
            and (event is None or r.value["event"] == event)
        ),
        key=lambda v: v["at"],
    )


def test_a_forged_by_from_a_non_approver_is_refused_and_nothing_is_charged(
    charged,
):
    g = _served()
    config = _cfg("srv-forged", AGENT)
    _start(g, config)

    out = g.invoke(approve(by="ceo@corp"), config)

    assert charged == []
    assert out.get("refund_done") is not True
    assert out["ticket"] == {"status": "escalated"}
    [row] = _rows(g)
    assert row["by"] == "agent-7" and row["amount"] is None
    assert "may not approve" in row["refused"]


def test_goto_refund_by_a_non_approver_is_refused_at_the_charge(charged):
    """Probe A: skip the gate with Command(goto="refund") after the pause."""
    g = _served()
    config = _cfg("srv-goto", AGENT)
    _start(g, config)

    out = g.invoke(Command(goto="refund"), config)

    assert charged == []
    assert out.get("refund_done") is not True
    [row] = _rows(g, event="refund")
    assert row["by"] == "agent-7" and "may not approve" in row["outcome"]


def test_resume_plus_goto_by_a_non_approver_is_refused_twice(charged):
    """Probe B: the gate refuses the resume, the goto still lands on refund,
    and refund refuses again: refusal rows only, no charge."""
    g = _served()
    config = _cfg("srv-resume-goto", AGENT)
    _start(g, config)

    out = g.invoke(
        Command(resume={"type": "approve", "shown": SHOWN}, goto="refund"), config
    )

    assert charged == []
    assert out.get("refund_done") is not True
    rows = _rows(g)
    assert len(rows) == 2 and all(r["by"] == "agent-7" for r in rows)


def test_goto_with_a_forged_amount_by_a_non_approver_charges_nothing(charged):
    """Probe C: goto + update(ticket.amount=9999) from a thread owner."""
    g = _served()
    config = _cfg("srv-goto-9999", AGENT)
    _start(g, config)

    out = g.invoke(
        Command(update={"ticket": {**TICKET, "amount": 9999.0}}, goto="refund"),
        config,
    )

    assert charged == []
    assert out.get("refund_done") is not True
    [row] = _rows(g, event="refund")
    assert row["amount"] is None and "may not approve" in row["outcome"]


def test_an_over_cap_charge_routes_to_refund_failed(charged):
    """M2 (R101): a lead approves a ticket claiming more than was paid; the
    provider refuses, and the node compensates with refund_failed and logs
    the refusal. (A forged approval for 9999 no longer reaches the provider:
    there is no audit row behind it.)"""
    g = _served()
    config = _cfg("srv-lead-over-cap", LEAD)
    over = {**TICKET, "amount": 100.0}
    _start(g, config, over)

    out = g.invoke(approve(shown={**SHOWN, "amount": 100.0}), config)

    assert charged == []
    assert out.get("refund_done") is not True
    assert out["error"] == "refund failed; needs manual review"
    [row] = _rows(g, event="refund")
    assert row["by"] == "lead-3" and row["outcome"].startswith("provider refused")


def test_goto_refund_by_a_lead_without_an_approval_charges_nothing(charged):
    g = _served()
    config = _cfg("srv-lead-goto", LEAD)
    _start(g, config)

    g.invoke(Command(goto="refund"), config)

    assert charged == []
    [row] = _rows(g, event="refund")
    assert row["outcome"] == "refused: no approval on record for this refund"


def test_a_fresh_thread_goto_ends_cleanly_and_does_not_wedge_the_thread(charged):
    """Probe G / M1: a brand-new thread routed straight to refund. Refund
    refuses and escalates; the gate on the START path finds the escalated
    ticket and ends cleanly. New input afterwards runs normally, and a
    lead's echoed approve charges exactly what was shown."""
    g = _served()
    agent_cfg = _cfg("srv-fresh", AGENT)

    g.invoke(
        Command(
            update={
                "messages": [{"role": "user", "content": "refund"}],
                "ticket": {**TICKET, "amount": 500.0},
            },
            goto="refund",
        ),
        agent_cfg,
    )

    assert charged == []
    assert g.get_state(agent_cfg).next == ()
    [refused] = _rows(g, event="refund")
    assert refused["by"] == "agent-7" and "may not approve" in refused["outcome"]

    lead_cfg = _cfg("srv-fresh", LEAD)
    _start(g, lead_cfg)
    assert g.get_state(lead_cfg).next == ("approval_gate",)
    out = g.invoke(approve(), lead_cfg)

    assert charged == [("T-1001", 10.0)]
    assert out["refund_done"] is True


def test_the_served_build_with_no_identity_fails_closed(charged):
    """I2/probe K: no user and no assistant_id in configurable. The served
    build does not fall back to the payload's `by`."""
    g = _served()
    config = {"configurable": {"thread_id": "srv-no-identity"}}
    _start(g, config)

    out = g.invoke(approve(by="ceo"), config)

    assert charged == []
    assert out.get("refund_done") is not True
    [row] = _rows(g)
    assert row["by"] is None
    assert row["refused"] == "refused: no authenticated approver"


def test_a_lead_edit_charges_the_approved_amount_and_logs_the_charge(charged):
    """R104/I1: a success writes its own row, with who charged what."""
    g = _served()
    config = _cfg("srv-lead", LEAD)
    _start(g, config)

    out = g.invoke(
        Command(resume={"type": "edit", "amount": 8.0, "by": "ceo", "shown": SHOWN}),
        config,
    )
    assert out["approval"]["audit_key"].startswith("approval:srv-lead:")

    assert charged == [("T-1001", 8.0)]
    assert out["refund_done"] is True
    [approval] = _rows(g, event="approval")
    assert approval["by"] == "lead-3" and approval["amount"] == 8.0
    assert approval["ticket_id"] == "T-1001" and approval["customer_id"] == "C-90"
    assert "refused" not in approval
    [charge] = _rows(g, event="refund")
    assert charge == {
        "by": "lead-3",
        "at": charge["at"],
        "ticket_id": "T-1001",
        "amount": 8.0,
        "approval_key": out["approval"]["audit_key"],
        "outcome": "charged",
        "event": "refund",
        "ticket": "T-1001",
        "thread": "srv-lead",
    }


def test_a_resume_without_the_echo_is_refused_on_the_served_build(charged):
    g = _served()
    config = _cfg("srv-no-echo", LEAD)
    _start(g, config)

    out = g.invoke(Command(resume={"type": "approve"}), config)

    assert charged == []
    assert out.get("refund_done") is not True
    [row] = _rows(g)
    assert row["refused"] == "refused: the decision does not echo what was shown"


@pytest.mark.parametrize(
    ("label", "rewrite"),
    [
        ("same ticket, new amount", {**TICKET, "amount": 49.0}),
        ("another customer's ticket", OTHER),
    ],
)
def test_a_ticket_rewritten_while_paused_is_refused_when_the_lead_approves(
    charged, label, rewrite
):
    """R104 C1: while the refund is paused, the thread owner rewrites the
    ticket through the state API (no run, so no create_run 403). The lead
    approves what they were shown - T-1001, $10 - and the gate, re-running
    on the rewritten state, refuses: nothing is charged, and the refusal is
    filed under the customer the approver confirmed."""
    g = _served()
    config = _cfg(f"srv-swap-{label}", AGENT)
    _start(g, config)
    g.update_state(config, {"ticket": rewrite})
    assert g.get_state(config).next == ("approval_gate",)

    out = g.invoke(approve(), _cfg(f"srv-swap-{label}", LEAD))

    assert charged == []
    assert out.get("refund_done") is not True
    [row] = _rows(g, customer_id=rewrite["customer_id"])
    assert row["by"] == "lead-3"
    assert row["refused"] == "refused: the ticket changed after the approver saw it"


def test_a_stale_approval_with_a_swapped_ticket_charges_nothing(charged):
    """R104 I1: after a lead approves T-1001, a goto to refund with the
    ticket swapped to T-2002 finds an approval for a different ticket."""
    g = _served()
    config = _cfg("srv-stale", LEAD)
    _start(g, config)
    g.invoke(approve(), config)
    assert charged == [("T-1001", 10.0)]

    g.invoke(Command(update={"ticket": OTHER}, goto="refund"), config)

    assert charged == [("T-1001", 10.0)]
    refund_rows = _rows(g, event="refund")
    assert [r["outcome"] for r in refund_rows] == [
        "charged",
        "refused: the approval is for a different ticket",
    ]


@pytest.mark.parametrize(
    ("user", "expected"), [(AGENT, []), (LEAD, [("T-1001", 10.0)])]
)
def test_server_resolve_wiring_refuses_a_non_lead_and_charges_a_lead(
    charged, user, expected
):
    """Through atlas.deploy.server.resolve itself (its real gate and refund),
    given the checkpointer and store the server would supply."""
    from atlas.deploy import server

    g = server.resolve.copy(
        update={"checkpointer": InMemorySaver(), "store": InMemoryStore()}
    )
    config = _cfg(f"srv-real-{user.identity}", user)

    async def run() -> dict:
        await g.ainvoke(
            {"messages": [{"role": "user", "content": "refund"}], "ticket": TICKET},
            config,
        )
        return await g.ainvoke(approve(), config)

    out = asyncio.run(run())

    assert charged == expected
    assert (out.get("refund_done") is True) == bool(expected)


def test_in_process_goto_refund_without_an_approval_charges_nothing(charged):
    """The in-process build keeps the payload's `by`, but refund still needs
    an approve/edit decision with an amount before it charges."""
    config = {"configurable": {"thread_id": "inproc-goto"}}
    graph.invoke(
        {"messages": [{"role": "user", "content": "refund"}], "ticket": TICKET},
        config,
    )

    graph.invoke(Command(goto="refund"), config)

    assert charged == []
    assert graph.get_state(config).values.get("refund_done") is not True


# --- R105: refund charges only against the gate's own audit row --------------


def _planted(key: str | None = "approval:forged") -> dict:
    """An approval a caller writes into state: approve, T-2002/C-91, $300."""
    return {
        "decision": "approve",
        "by": "lead-3",
        "at": "t",
        "ticket_id": "T-2002",
        "customer_id": "C-91",
        "amount": 300.0,
        "audit_key": key,
    }


def _refund_rows(g, customer_id: str) -> list[tuple]:
    return [
        (r["by"], r["ticket_id"], r["amount"], r["outcome"])
        for r in _rows(g, customer_id, event="refund")
    ]


def _at(checkpoint: dict, config: dict) -> dict:
    """A checkpoint's config, run as the caller `config` names."""
    return {"configurable": {**checkpoint["configurable"], **config["configurable"]}}


NO_ROW = "refused: no approval on record for this refund"
CHANGED = "refused: the ticket changed after the approver saw it"
NOT_YOURS = "refused: the approval on record is not this approver's"
FORGED_T1001 = {"ticket_id": "T-1001", "customer_id": "C-90", "amount": 10.0}


@pytest.mark.parametrize("lead_run", ["echo", "invoke(None)"])
def test_an_approval_planted_with_supersteps_charges_nothing(charged, lead_run):
    """R104 C1, the supersteps shape: the agent applies
    `Command(goto="refund", update={ticket, approval})` as a state update
    with no run - what `threads.create(supersteps=...)` asks the server to
    do - and the lead's next run reaches `refund` without the gate. There is
    no audit row behind the approval, so nothing is charged."""
    from langgraph.types import StateUpdate

    g = _served()
    agent_cfg = _cfg(f"srv-supersteps-{lead_run}", AGENT)
    _start(g, agent_cfg)
    planted = Command(goto="refund", update={"ticket": OTHER, "approval": _planted()})
    g.bulk_update_state(agent_cfg, [[StateUpdate(planted, None)]])
    assert "refund" in g.get_state(agent_cfg).next

    lead_cfg = _cfg(f"srv-supersteps-{lead_run}", LEAD)
    out = g.invoke(approve() if lead_run == "echo" else None, lead_cfg)

    assert charged == []
    assert out.get("refund_done") is not True
    assert _refund_rows(g, "C-91") == [("lead-3", "T-2002", None, NO_ROW)]
    gate_rows = [
        (r["by"], r["ticket_id"], r["amount"], r["refused"])
        for r in _rows(g, "C-91", event="approval")
    ]  # the echo also resumes the paused gate, which sees the planted ticket
    assert gate_rows == (
        [("lead-3", "T-2002", None, CHANGED)]
        if lead_run == "echo"
        else []
    )
    assert _rows(g, "C-90") == []


def test_a_planted_key_naming_another_threads_real_row_charges_nothing(charged):
    """The planted approval points at a genuine approval row - the lead's,
    for T-2002, on another thread whose charge the provider refused. The
    row exists, but it is not this thread's."""
    g = _served()
    real_cfg = _cfg("srv-real-row", LEAD)
    _start(g, real_cfg, {**OTHER, "amount": 400.0})  # over T-2002's cap
    shown = {"ticket_id": "T-2002", "customer_id": "C-91", "amount": 400.0}
    real = g.invoke(approve(shown=shown), real_cfg)["approval"]
    assert charged == []

    trap_cfg = _cfg("srv-trap", AGENT)
    _start(g, trap_cfg)
    planted = {**real, "amount": 300.0}
    g.update_state(
        trap_cfg, Command(goto="refund", update={"ticket": OTHER, "approval": planted})
    )
    g.invoke(None, _cfg("srv-trap", LEAD))

    assert charged == []
    assert _refund_rows(g, "C-91") == [
        ("lead-3", "T-2002", 400.0, "provider refused: 400.00 exceeds the original"),
        ("lead-3", "T-2002", None, NO_ROW),
    ]


def test_an_approval_planted_through_run_input_charges_nothing(charged):
    """R104 C1, the run-input shape: the agent's input carries an approval
    (the auth hook refuses this over HTTP; here it reaches the graph). The
    gate interrupts honestly on the planted ticket; the lead skips it with a
    manual `goto` - the override that trusted state before R105."""
    g = _served()
    agent_cfg = _cfg("srv-input-plant", AGENT)
    g.invoke(
        {
            "messages": [{"role": "user", "content": "refund please"}],
            "ticket": OTHER,
            "approval": _planted(),
        },
        agent_cfg,
    )

    out = g.invoke(Command(goto="refund"), _cfg("srv-input-plant", LEAD))

    assert charged == []
    assert out.get("refund_done") is not True
    assert _refund_rows(g, "C-91") == [("lead-3", "T-2002", None, NO_ROW)]
    assert _rows(g, "C-91", event="approval") == []


def test_a_lead_forging_an_approval_with_goto_charges_nothing(charged):
    """R101 I1 (charged $30, by design, before R105): an approver's forged
    approval has no row behind it either."""
    g = _served()
    config = _cfg("srv-lead-forge", LEAD)
    _start(g, config)
    forged = {**_planted(), **FORGED_T1001}

    g.invoke(Command(update={"approval": forged}, goto="refund"), config)

    assert charged == []
    assert _refund_rows(g, "C-90") == [("lead-3", "T-1001", None, NO_ROW)]


def test_a_lead_approves_an_agents_refund_and_it_is_charged_once(charged):
    """The served flow R105 makes reachable: the agent opens the refund,
    the lead resumes the agent's thread with the echo."""
    g = _served()
    _start(g, _cfg("srv-handoff", AGENT))

    out = g.invoke(approve(), _cfg("srv-handoff", LEAD))

    assert charged == [("T-1001", 10.0)]
    assert out["refund_done"] is True and out["error"] is None
    [approval] = _rows(g, event="approval")
    assert (approval["by"], approval["role"], approval["amount"]) == (
        "lead-3",
        "support_lead",
        10.0,
    )
    assert _refund_rows(g, "C-90") == [("lead-3", "T-1001", 10.0, "charged")]


def test_another_approver_cannot_charge_against_a_leads_approval(charged):
    """(d): the row's `by` must be this run's identity. lead-3's run stops
    after the gate, before `refund` (a crash, a drain); lead-9 picking the
    thread up cannot spend lead-3's approval."""
    g = _served()
    lead_cfg = _cfg("srv-other-lead", LEAD)
    _start(g, lead_cfg)
    g.invoke(approve(), lead_cfg, interrupt_before=["refund"])
    assert g.get_state(lead_cfg).next == ("refund",)

    out = g.invoke(None, _cfg("srv-other-lead", _User("lead-9", "support_lead")))

    assert charged == []
    assert out.get("refund_done") is not True
    assert _refund_rows(g, "C-90") == [
        ("lead-9", "T-1001", None, NOT_YOURS)
    ]


@pytest.mark.parametrize("how", ["state API", "new input"])
def test_a_customer_only_rewrite_is_refused_by_the_echo(charged, how):
    """R104 M1: same ticket id and amount, another customer. The lead echoes
    what they were shown (C-90); the gate re-runs on C-91 and refuses."""
    g = _served()
    agent_cfg = _cfg(f"srv-cust-{how}", AGENT)
    _start(g, agent_cfg)
    swapped = {**TICKET, "customer_id": "C-91"}
    if how == "state API":
        g.update_state(agent_cfg, {"ticket": swapped})
    else:
        _start(g, agent_cfg, swapped)
        shown = g.get_state(agent_cfg).tasks[0].interrupts[0].value
        assert shown["customer_id"] == "C-91"  # the new interrupt names it

    out = g.invoke(approve(), _cfg(f"srv-cust-{how}", LEAD))

    assert charged == []
    assert out.get("refund_done") is not True
    [row] = _rows(g, "C-91", event="approval")
    assert row["refused"] == "refused: the ticket changed after the approver saw it"
    assert _rows(g, "C-90") == []


def test_a_replayed_refund_charges_once_and_logs_one_charged_row(charged):
    """R104 M2. Re-running `refund` from the checkpoint before it reuses the
    same approval key: refused. A fork that passes the gate again gets a new
    approval, and the provider dedupes: the row says "replayed"."""
    g = _served()
    config = _cfg("srv-replay", LEAD)
    _start(g, config)
    paused = g.get_state(config).config
    g.invoke(approve(), config)
    before_refund = next(
        s.config for s in g.get_state_history(config) if s.next == ("refund",)
    )

    g.invoke(None, _at(before_refund, config))
    fork = _at(g.update_state(paused, {"error": None}, as_node="triage"), config)
    g.invoke(None, fork)
    g.invoke(approve(), fork)

    assert charged == [("T-1001", 10.0)]
    assert [r[3] for r in _refund_rows(g, "C-90")] == [
        "charged",
        "refused: this approval has already been charged",
        "replayed: already charged",
    ]


def test_a_charge_clears_a_stale_error(charged):
    """R104 M3: a provider refusal leaves `error` set; a later echoed approve
    at an amount the provider accepts charges and clears it."""
    g = _served()
    config = _cfg("srv-stale-error", LEAD)
    _start(g, config, {**TICKET, "amount": 60.0})
    out = g.invoke(approve(shown={**SHOWN, "amount": 60.0}), config)
    assert out["error"] == "refund failed; needs manual review"

    _start(g, config)
    out = g.invoke(approve(), config)

    assert charged == [("T-1001", 10.0)]
    assert out["refund_done"] is True and out["error"] is None


@pytest.mark.parametrize(
    "probe",
    [
        "E: update_state(Command(goto)) + invoke(None)",
        "C2: goto + forged approval naming the ticket",
        "K2: no identity, goto + forged approval",
        "L: langgraph_auth_user as a plain dict, echoed approve",
    ],
)
def test_earlier_review_probes_still_charge_nothing(charged, probe):
    """R94/R101/R104 probes not already pinned above, re-run on R105."""
    tag = probe[:2].rstrip(":")
    user = {"E": AGENT, "C2": AGENT, "K2": None, "L": {"identity": "lead-3"}}[tag]
    config = _cfg(f"srv-probe-{tag}", user)
    if user is None:
        config = {"configurable": {"thread_id": "srv-probe-K2"}}
    g = _served()
    _start(g, config)
    forged = {**_planted(), **FORGED_T1001}
    if tag == "E":
        g.update_state(config, Command(goto="refund"))
        out = g.invoke(None, config)
    elif tag == "L":
        out = g.invoke(approve(), config)
    else:
        out = g.invoke(Command(update={"approval": forged}, goto="refund"), config)

    assert charged == []
    assert out.get("refund_done") is not True
    assert [r for r in _rows(g) if r.get("outcome") == "charged"] == []
    assert all(r.get("refused") or r.get("outcome") for r in _rows(g))


# --- R106: the cap is cumulative per ticket; a re-approval cannot re-price --


def test_two_threads_cannot_refund_more_than_was_paid_in_total(monkeypatch):
    """R105 M1, with the real seeded provider: T-1001 was paid 49. A lead
    genuinely approves $30 on each of two threads. The idempotency key is
    per thread, so the provider sees two refunds; the cumulative cap
    refuses the second, and the node compensates with refund_failed."""
    from atlas import effects  # conftest gives each test the seeded provider

    monkeypatch.setattr(graph_module, "charge_refund", effects.charge_refund)
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: SimpleNamespace(route="refund")
    )
    g = _served()
    ticket = {**TICKET, "amount": 30.0}
    outs = []
    for thread in ("srv-cap-d1", "srv-cap-d2"):
        config = _cfg(thread, LEAD)
        _start(g, config, ticket)
        outs.append(g.invoke(approve(shown={**SHOWN, "amount": 30.0}), config))

    assert outs[0]["refund_done"] is True
    assert outs[1].get("refund_done") is not True
    assert outs[1]["error"] == "refund failed; needs manual review"
    assert effects._REFUNDS["T-1001"]["refunded"] == 30.0
    assert [r[3] for r in _refund_rows(g, "C-90")] == [
        "charged",
        "provider refused: 30.00 plus 30.00 already refunded exceeds the "
        "original 49.00",
    ]


def test_a_reapproval_at_a_new_amount_after_a_charge_is_refused(charged):
    """R105 M2: on one thread, $10 is charged; a new turn asks for $20 and
    the lead approves it. The provider would replay the $10 charge (same
    thread, same ticket), so refund refuses and escalates instead of
    reporting a refund that did not happen."""
    g = _served()
    config = _cfg("srv-reprice", LEAD)
    _start(g, config)
    g.invoke(approve(), config)
    assert charged == [("T-1001", 10.0)]

    _start(g, config, {**TICKET, "amount": 20.0})
    out = g.invoke(approve(shown={**SHOWN, "amount": 20.0}), config)

    assert charged == [("T-1001", 10.0)]
    assert out["messages"][-1].content == (
        "Refund not issued: refused: already refunded 10.00"
    )
    assert _refund_rows(g, "C-90") == [
        ("lead-3", "T-1001", 10.0, "charged"),
        ("lead-3", "T-1001", None, "refused: already refunded 10.00"),
    ]


def test_a_replay_at_the_same_amount_records_the_amount_charged(charged):
    """The "replayed" row names the amount the provider actually charged."""
    g = _served()
    config = _cfg("srv-replay-amount", LEAD)
    _start(g, config)
    g.invoke(approve(), config)
    _start(g, config)
    g.invoke(approve(), config)

    assert charged == [("T-1001", 10.0)]
    assert _refund_rows(g, "C-90") == [
        ("lead-3", "T-1001", 10.0, "charged"),
        ("lead-3", "T-1001", 10.0, "replayed: already charged"),
    ]
