"""Chapter 23, "Security, Privacy, Cost, and Governance" - atlas/audit.py.

See "A durable audit log, deliberately separate from the trace".
`AuditGate.wrap_tool_call` needs no live model call or LangSmith connection
to test - it only writes a durable record to a `BaseStore` after the real
handler runs, so an `InMemoryStore` plus a hand-built `ToolCallRequest`
(the `test_middleware.py`/`test_security.py` convention) is enough to
exercise it directly."""

import asyncio
from types import SimpleNamespace

from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from atlas import graph as graph_module
from atlas.audit import AuditGate, audit_ns, record_approval
from atlas.graph import graph
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
    """A replay of the same decision adds nothing; a different decision at
    the same checkpoint (a fork) appends; no row is ever overwritten."""
    store = InMemoryStore()
    ticket = {"id": "T-9", "customer_id": "C-1"}
    first = {"decision": "approve", "by": "lead-3", "at": "t0", "amount": 49.0}

    k1 = record_approval(store, ticket, "t-1", "cp-1", first)
    k1_again = record_approval(store, ticket, "t-1", "cp-1", {**first, "at": "t1"})
    fork = {"decision": "reject", "by": "lead-3", "at": "t2", "amount": None}
    k2 = record_approval(store, ticket, "t-1", "cp-1", fork)

    assert k1 == k1_again == "approval:t-1:cp-1"
    assert k2 == "approval:t-1:cp-1:2"
    assert store.get(audit_ns("C-1"), k1).value == {
        **first,
        "event": "approval",
        "ticket": "T-9",
        "thread": "t-1",
    }
    assert store.get(audit_ns("C-1"), k2).value["decision"] == "reject"


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


# --- R94: on the served path the approver is the authenticated identity ----


class _User:
    def __init__(self, identity: str, role: str) -> None:
        self.identity = identity
        self.permissions = [f"role:{role}"]


def _charges(monkeypatch) -> list:
    charged: list = []
    monkeypatch.setattr(
        graph_module,
        "charge_refund",
        lambda key, ticket_id, amount: charged.append(amount) or "Refund issued.",
    )
    return charged


def test_a_forged_by_from_a_non_approver_is_refused_and_nothing_is_charged(
    monkeypatch,
):
    """A thread owner without the approver role resumes with
    {"by": "ceo@corp"}. The gate reads the identity the server proved, not
    the payload, refuses, escalates, and records who actually tried."""
    charged = _charges(monkeypatch)
    agent = _User("agent-7", "support_agent")
    config = _refund_thread(
        monkeypatch, "served-forged", "C-80", langgraph_auth_user=agent
    )

    out = graph.invoke(
        Command(resume={"type": "approve", "by": "ceo@corp"}), config
    )

    assert charged == []
    assert out.get("refund_done") is not True
    assert out["ticket"] == {"status": "escalated"}
    [row] = _approval_rows("C-80")
    assert row.value["by"] == "agent-7"
    assert row.value["amount"] is None
    assert "may not approve" in row.value["refused"]


def test_an_anonymous_resume_on_the_served_path_is_refused(monkeypatch):
    charged = _charges(monkeypatch)
    config = _refund_thread(
        monkeypatch, "served-anon", "C-81", assistant_id="resolve"
    )

    graph.invoke(Command(resume={"type": "approve", "by": "lead-3"}), config)

    assert charged == []
    [row] = _approval_rows("C-81")
    assert row.value["by"] is None
    assert row.value["refused"] == "approval refused: no authenticated approver"


def test_an_authenticated_approver_is_recorded_whatever_the_payload_says(
    monkeypatch,
):
    charged = _charges(monkeypatch)
    lead = _User("lead-3", "support_lead")
    config = _refund_thread(
        monkeypatch, "served-lead", "C-82", langgraph_auth_user=lead
    )

    out = graph.invoke(
        Command(resume={"type": "approve", "by": "ceo@corp"}), config
    )

    assert charged == [49.0]
    assert out["refund_done"] is True
    [row] = _approval_rows("C-82")
    assert row.value["by"] == "lead-3"
    assert "refused" not in row.value
