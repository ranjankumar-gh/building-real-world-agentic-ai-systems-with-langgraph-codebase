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
  logs every tool call that reaches the tools node and refuses
  `set_ticket_status(..., "resolved")` unless a human approved that exact
  call. See "Gating the authority surface (custom middleware)". This
  enforces the authority surface from Chapter 7.
- `approval` - a `RecordingApproval`, a `HumanInTheLoopMiddleware` that
  pauses in `after_model` on any `set_ticket_status` call, before the tools
  node runs, and then writes the ids of the calls the human approved (or
  edited) to the `approved_calls` state key. `AuthorityGate` reads that
  key: `HumanInTheLoopMiddleware` itself leaves an approved call exactly as
  it leaves an auto-approved one, so without the record the gate could not
  tell them apart. See "Human approval, as a placeholder". Without a
  checkpointer the pause stops the run but can never resume it; Chapter 9
  adds the checkpointer and Chapter 11 operates the pause.

`atlas/agent.py`'s `resolve_agent` composes these four in
`[pii, summarizer, AuthorityGate(), approval]` - list order is nesting
order (first = outermost); see "Composing the stack".

Chapter 20, "Observability and Debugging with LangSmith", closes a gap
between `pii`'s wire-level redaction and what LangSmith's tracing client
records - see "The PII redaction ordering bug, made concrete". `pii`
redacting the stream says nothing about what a trace stores; the two are
independent sinks fed by independent mechanisms. `EMAIL_PATTERN` is the one
shared definition of what counts as an email: its `.pattern` string form is
`pii`'s `detector=` (matching `PIIMiddleware`'s actual contract - a custom
`detector` is a callable returning `list[PIIMatch]` or a plain regex pattern
string, verified against the installed `langchain==1.3.0` build; a callable
returning a *redacted string* - the earlier draft's mistake - raises
`AttributeError: 'str' object has no attribute 'get'` the moment content is
scanned), and `redact_email` (built from the same compiled pattern) is what
`atlas/tracing.py`'s `Client(hide_outputs=...)` uses for the trace side.
"""

import logging
import re
from collections.abc import Callable
from typing import Any, NotRequired

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallRequest,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

log = logging.getLogger("atlas")

# --- PII redaction: redact email addresses in, and again out. ----------------
# One PIIMiddleware instance covers both directions - two separate instances
# for the same pii_type collide on create_agent's duplicate-middleware check
# (both resolve to the name "PIIMiddleware[email]"). See the module docstring.

# Chapter 20: the ONE shared definition of "what is an email" - PIIMiddleware
# below uses its .pattern string as a detector; atlas/tracing.py's
# redact_trace_outputs uses the compiled pattern directly via redact_email.
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

pii = PIIMiddleware(
    "email",
    strategy="redact",
    apply_to_input=True,
    apply_to_output=True,
    detector=EMAIL_PATTERN.pattern,  # <1>
)


def redact_email(text: str) -> str:
    """Atlas's own string-level redaction, built from the same EMAIL_PATTERN
    passed to `pii` above - so the wire and the trace (atlas/tracing.py's
    `redact_trace_outputs`) can never disagree about what "redacted" means.
    """
    return EMAIL_PATTERN.sub("[EMAIL]", text)


# 1. A regex pattern string, not a callable - PIIMiddleware's `detector=`
#    contract expects either `None` (its own built-in detector), a plain
#    regex pattern string, or a callable returning `list[PIIMatch]`. A
#    callable that returns a *redacted string* (what `redact_email` returns)
#    satisfies none of those and breaks the moment content is scanned - the
#    pattern-string form is what "one shared definition" has to mean here.

# --- History summarization: bound context growth, keep recent turns. -------
summarizer = SummarizationMiddleware(
    model=init_chat_model("claude-sonnet-4-6"),
    trigger=("tokens", 4000),  # summarize once history crosses 4k tokens
    keep=("messages", 20),  # always keep the 20 most recent messages verbatim
)

# Writes that change a ticket's terminal state need a human (full HITL: Ch 11).
_NEEDS_APPROVAL = {"set_ticket_status"}


class AuthorityGate(AgentMiddleware):
    """Log every tool call; refuse a terminal ticket write no human approved."""

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
            approved = request.state.get("approved_calls", [])  # <2>
            if request.tool_call["id"] not in approved:
                return ToolMessage(  # <3>
                    "Blocked: marking a ticket resolved requires human approval.",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
        return handler(request)  # <4>


# --- Human approval: pause, then record what the human let through. --------
# Without a checkpointer the pause stops the run but can never resume it
# (Ch 9 adds the checkpointer; Ch 11 operates the pause).


class ApprovalState(AgentState):
    approved_calls: NotRequired[list[str]]  # tool-call ids a human let through


class RecordingApproval(HumanInTheLoopMiddleware):
    """Pause for a human, then record which paused calls went ahead."""

    state_schema = ApprovalState

    def after_model(
        self, state: ApprovalState, runtime: Runtime
    ) -> dict[str, Any] | None:
        update = super().after_model(state, runtime)  # <1>
        if update is None:
            return None
        ai_msg, *answered = update["messages"]  # <2>
        answered_ids = {m.tool_call_id for m in answered}
        approved = [
            call["id"]
            for call in ai_msg.tool_calls
            if call["name"] in self.interrupt_on and call["id"] not in answered_ids
        ]
        return {**update, "approved_calls": approved}  # <3>


approval = RecordingApproval(
    interrupt_on={"set_ticket_status": True},  # pause before this tool runs
)
