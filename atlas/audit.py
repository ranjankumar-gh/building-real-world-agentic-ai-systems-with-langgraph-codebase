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
- `record_approval` records every Chapter 11 approval decision (event
  "approval": decision, approver, time, the ticket id, customer and amount
  the approver confirmed, what they were shown, or the refusal) and every
  outcome of the refund node, which checks again before it charges (event
  "refund": charged, replayed, refused, provider refused, failed).
  Append-only: every write has its own key. The approval row is also the
  refund's authority: the gate puts its key in `state["approval"]`, and
  `refund` charges only against that row, once. The refund is a graph node,
  not a tool,
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
from uuid import uuid4

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


def record_approval(
    store: BaseStore,
    customer_id: str | None,
    thread_id: str,
    checkpoint_id: str,
    record: dict[str, Any],
    event: str = "approval",
) -> str:
    """Append one approval or refund event to a customer's audit namespace.

    Append-only by construction: every write gets a fresh key - the thread,
    the checkpoint the event happened at, and a random suffix - so no write
    can land on another's row and nothing is read first. A gate or refund
    node that runs again (a retry, a replay, a fork from an earlier
    checkpoint) logs its own row; a reader sorts by `at` and reads them all.
    `customer_id` is the customer the approver confirmed (the approval
    record's), not whatever the ticket in state says now. Returns the key."""
    key = f"{event}:{thread_id}:{checkpoint_id}:{uuid4().hex}"
    store.put(
        audit_ns(customer_id or "unknown"),
        key,
        {
            **record,
            "event": event,
            "ticket": record.get("ticket_id"),
            "thread": thread_id,
        },
    )
    return key


def approval_on_record(
    store: BaseStore, customer_id: str | None, key: str | None
) -> dict[str, Any] | None:
    """The approval row the gate wrote under `key`, or None.

    `refund` authorizes from this row, not from `state["approval"]`: state
    is caller-writable, and the audit namespace is not (no caller reaches
    the store except through `atlas/auth.py`'s store handler, which refuses
    every "audit" namespace). A key that names no row, or a row that is not
    an approval, authorizes nothing."""
    if not key:
        return None
    item = store.get(audit_ns(customer_id or "unknown"), key)
    if item is None or item.value.get("event") != "approval":
        return None
    return item.value


def charged_rows(
    store: BaseStore, customer_id: str | None, **match: Any
) -> list[dict[str, Any]]:
    """The "charged" refund rows in a customer's namespace matching `match`
    (exact field values, e.g. approval_key=..., or thread=..., ticket=...)."""
    found = store.search(
        audit_ns(customer_id or "unknown"),
        filter={"event": "refund", "outcome": "charged", **match},
        limit=10,
    )
    return [item.value for item in found]
