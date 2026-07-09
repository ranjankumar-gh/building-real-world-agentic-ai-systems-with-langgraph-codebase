"""Chapter 8, "The Middleware System" - Atlas's middleware stack.

See "Building Atlas's middleware stack". Four cross-cutting concerns that
used to threaten to leak into `atlas/tools.py` or `atlas/agent.py` move here
instead, as an ordered stack of layers on the agent-loop seam:

- `pii` - a `PIIMiddleware` instance that redacts email addresses on the
  way in *and* on the way out. See "PII redaction". `PIIMiddleware`
  identifies itself to `create_agent` by `pii_type` alone, so two separate
  instances for the same type (one `apply_to_input`, one `apply_to_output`)
  collide and `create_agent` raises `AssertionError: Please remove
  duplicate middleware instances.` - one instance with both flags set
  covers both directions.
- `summarizer` - a `SummarizationMiddleware` that trims history once it
  crosses a token threshold, keeping the most recent messages verbatim. See
  "History summarization". A stopgap until Chapter 12's context budget.
- `AuthorityGate` - a custom `AgentMiddleware` with `wrap_tool_call` that
  logs every tool call and hard-blocks `set_ticket_status(..., "resolved")`
  pending human approval. See "Gating the authority surface (custom
  middleware)". This enforces the authority surface from Chapter 7.
- `approval` - a `HumanInTheLoopMiddleware` wired to pause before
  `set_ticket_status`. See "Human approval, as a placeholder" - wiring it is
  one line, but it cannot actually suspend a run without the checkpointer
  Chapter 9 introduces, so it is inert here.

`atlas/agent.py`'s `resolve_agent` composes these four in
`[pii, summarizer, AuthorityGate(), approval]` - list order is nesting
order (first = outermost); see "Composing the stack".
"""

import logging
from collections.abc import Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallRequest,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langgraph.types import Command

log = logging.getLogger("atlas")

# --- PII redaction: mask email addresses in, and again out. ----------------
# One PIIMiddleware instance covers both directions - two separate instances
# for the same pii_type collide on create_agent's duplicate-middleware check
# (both resolve to the name "PIIMiddleware[email]"). See the module docstring.
pii = PIIMiddleware(
    "email", strategy="redact", apply_to_input=True, apply_to_output=True
)

# --- History summarization: bound context growth, keep recent turns. -------
summarizer = SummarizationMiddleware(
    model=init_chat_model("claude-sonnet-4-6"),
    trigger=("tokens", 4000),  # summarize once history crosses 4k tokens
    keep=("messages", 20),  # always keep the 20 most recent messages verbatim
)

# Writes that change a ticket's terminal state need a human (full HITL: Ch 11).
_NEEDS_APPROVAL = {"set_ticket_status"}


class AuthorityGate(AgentMiddleware):
    """Log every tool call; block terminal ticket writes pending approval."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        name = request.tool_call["name"]
        log.info("tool_call name=%s args=%s", name, request.tool_call["args"])  # <1>
        if name in _NEEDS_APPROVAL and request.tool_call["args"].get(
            "status"
        ) == "resolved":
            return ToolMessage(  # <2>
                "Blocked: marking a ticket resolved requires human approval.",
                tool_call_id=request.tool_call["id"],
                status="error",
            )
        return handler(request)  # <3>


# --- Human approval, as a placeholder: inert until Ch 9's checkpointer. ----
approval = HumanInTheLoopMiddleware(
    interrupt_on={"set_ticket_status": True},  # pause before this tool runs
)
