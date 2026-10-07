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


def test_record_approval_keys_by_thread_and_ticket_so_a_replay_overwrites():
    store = InMemoryStore()
    ticket = {"id": "T-9", "customer_id": "C-1"}
    first = {"decision": "approve", "by": "lead-3", "at": "t0"}
    record_approval(store, ticket, "t-1", first)
    record_approval(store, ticket, "t-1", {**first, "at": "t1"})

    rows = store.search(audit_ns("C-1"))
    assert [r.key for r in rows] == ["approval:t-1:T-9"]
    assert rows[0].value == {
        "decision": "approve",
        "by": "lead-3",
        "at": "t1",
        "event": "approval",
        "ticket": "T-9",
        "thread": "t-1",
    }


def test_the_approval_gate_writes_its_decision_to_the_audit_namespace(monkeypatch):
    """End to end through the compiled graph: suspend at the gate, resume
    with an approval, and find the decision in the audit namespace of the
    graph's own store - after the human acted, never before."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: SimpleNamespace(route="refund")
    )
    config = {"configurable": {"thread_id": "audit-approval-1"}}
    graph.invoke(
        {
            "messages": [{"role": "user", "content": "refund please"}],
            "ticket": {"id": "T-1001", "amount": 49.0, "customer_id": "C-77"},
        },
        config,
    )
    key = "approval:audit-approval-1:T-1001"
    assert graph.store.get(audit_ns("C-77"), key) is None

    graph.invoke(Command(resume={"type": "approve", "by": "lead@support"}), config)

    row = graph.store.get(audit_ns("C-77"), key)
    assert row.value["decision"] == "approve"
    assert row.value["by"] == "lead@support"
    assert row.value["ticket"] == "T-1001"
    assert row.value["thread"] == "audit-approval-1"
    assert "at" in row.value
