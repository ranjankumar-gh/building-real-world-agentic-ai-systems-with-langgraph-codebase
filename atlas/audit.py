"""Chapter 23, "Security, Privacy, Cost, and Governance" - a durable audit
log, deliberately separate from the trace.

See "A durable audit log, deliberately separate from the trace". Chapter
20 (`atlas/tracing.py`) was explicit that a trace is not a log - a
debugging tool, walkable as a tree, built to explain one run. It is also,
by the same design, sampleable (Chapter 21) and redactable (Chapter 20's
own PII fix). Neither property is a defect; a trace was never meant to be
a compliance record. An auditor asking "who approved this refund, and
when" needs something a trace's own retention policy was never built to
guarantee - a complete, durable record that lives in Atlas's own store,
not in LangSmith's.

Two writers share one namespace, `audit_ns(customer_id)`:

- `AuditGate` records every tool call the resolve agent makes, refused or
  not: tool, args, role, result status, time.
- `record_approval` records every Chapter 11 approval decision: decision,
  approver, time, ticket, thread, and the amount sent on to `refund`
  (after an edit), append-only, one row per checkpoint the decision was
  made at. The refund is a graph node, not a tool,
  so `AuditGate` never sees it; and Chapter 11's `approval` record lives in
  graph state, which is a checkpoint, and checkpoints are the first thing
  an erasure deletes (`atlas/erasure.py`). So `atlas/graph.py`'s
  `audited_approval_gate` copies the decision here as well.

Both write to the store the graph was compiled with - Chapter 13's
`BaseStore`, the same underlying Postgres in production - through
`atlas/security.py`'s `graph_store`. That path never goes through
LangSmith's ingestion at all. A sampling rate tuned for cost (Chapter 21's
online monitor) or a redaction rule tuned for compliance (Chapter 20) can
change without anyone realizing the audit trail changed with it, because
they were never the same record."""

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.store.base import BaseStore
from langgraph.types import Command

from atlas.security import graph_store


def audit_ns(customer_id: str) -> tuple[str, ...]:
    """One audit namespace per customer, outside every thread."""
    return ("audit", customer_id)


class AuditGate(AgentMiddleware):
    """Records every tool call - Ch8's log.info, made durable and complete,
    in Atlas's OWN store rather than a trace that can be sampled/redacted."""

    def __init__(self, store: BaseStore | None = None) -> None:
        self.store = store  # only for an agent run with no store of its own

    def _entry(
        self, request: ToolCallRequest, response: ToolMessage | Command
    ) -> tuple[tuple[str, ...], str, dict[str, Any]]:
        context = request.runtime.context
        return (
            audit_ns(context.customer_id),
            request.tool_call["id"],
            {
                "tool": request.tool_call["name"],
                "args": request.tool_call["args"],
                "role": context.role,
                "result_status": getattr(response, "status", "command"),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        response = handler(request)
        store = graph_store(request.runtime, self.store)
        store.put(*self._entry(request, response))
        return response

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        response = await handler(request)
        store = graph_store(request.runtime, self.store)
        await store.aput(*self._entry(request, response))
        return response


_SAME = ("decision", "by", "amount", "refused")


def record_approval(
    store: BaseStore,
    ticket: dict[str, Any],
    thread_id: str,
    checkpoint_id: str,
    record: dict[str, Any],
) -> str:
    """Append a Chapter 11 approval decision to the audit namespace.

    Append-only. The key names the checkpoint the decision was made at; a
    replay of the same decision finds its row and adds nothing, and a
    different decision at the same checkpoint (a fork resumed from it) gets
    the next free suffix. No row is ever overwritten. Returns the key."""
    ns = audit_ns(ticket.get("customer_id") or "unknown")
    base = f"approval:{thread_id}:{checkpoint_id}"
    row = {**record, "event": "approval", "ticket": ticket["id"], "thread": thread_id}
    n = 1
    while True:
        key = base if n == 1 else f"{base}:{n}"
        existing = store.get(ns, key)
        if existing is None:
            store.put(ns, key, row)
            return key
        if all(existing.value.get(f) == row.get(f) for f in _SAME):
            return key  # this decision is already on record
        n += 1
