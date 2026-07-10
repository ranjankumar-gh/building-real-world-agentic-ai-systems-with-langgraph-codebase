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

`AuditGate` writes to the same durable `BaseStore` Chapter 13
(`atlas/memory.py`) already configured for long-term memory - a different
namespace (`("audit", customer_id)`), the same underlying Postgres, and
critically, a path that never goes through LangSmith's ingestion at all. A
sampling rate tuned for cost (Chapter 21's online monitor) or a redaction
rule tuned for compliance (Chapter 20) can change without anyone realizing
the audit trail changed with it, because they were never the same
record."""

from datetime import datetime, timezone

from langchain.agents.middleware import AgentMiddleware
from langgraph.store.base import BaseStore


class AuditGate(AgentMiddleware):
    """Records every tool call - Ch8's log.info, made durable and complete,
    in Atlas's OWN store rather than a trace that can be sampled/redacted."""

    def __init__(self, store: BaseStore) -> None:
        self.store = store

    def wrap_tool_call(self, request, handler):
        response = handler(request)
        customer_id = request.runtime.context.customer_id
        self.store.put(
            ("audit", customer_id),
            request.tool_call["id"],
            {
                "tool": request.tool_call["name"],
                "args": request.tool_call["args"],
                "role": request.runtime.context.role,
                "result_status": response.status,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return response
