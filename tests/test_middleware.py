"""Chapter 8, "The Middleware System" - atlas/middleware.py.

See "Building Atlas's middleware stack". These tests check construction and
configuration of the three built-ins (no live model call needed to build
them, matching the no-live-call convention from `tests/test_hello.py`), and
exercise `AuthorityGate.wrap_tool_call` directly against a hand-built
`ToolCallRequest` - the one piece of this chapter's stack that is plain
Python logic rather than a wired-up built-in.

`test_stack_composes_without_duplicate_middleware_errors` guards the bug the
chapter's first draft had: `create_agent` identifies each `PIIMiddleware` by
`pii_type` alone, so two separate instances for the same type (one
`apply_to_input`, one `apply_to_output`) collide and `create_agent` raises
`AssertionError: Please remove duplicate middleware instances.` - fixed by
folding both flags onto the single `pii` instance below."""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from atlas.middleware import AuthorityGate, approval, pii, summarizer


def _request(name: str, args: dict) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "call-1"},
        tool=None,
        state=None,
        runtime=None,
    )


def test_pii_redacts_email_on_both_the_way_in_and_the_way_out():
    assert isinstance(pii, PIIMiddleware)
    assert pii.pii_type == "email"
    assert pii.apply_to_input is True
    assert pii.apply_to_output is True


def test_stack_composes_without_duplicate_middleware_errors():
    """Two separate PIIMiddleware("email", ...) instances (one in, one out)
    both resolve to the name "PIIMiddleware[email]" and create_agent rejects
    the stack as duplicates. The single combined `pii` instance must not."""

    @tool
    def _dummy_tool(x: str) -> str:
        """A throwaway tool just to give create_agent something to wrap."""
        return x

    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=[_dummy_tool],
        system_prompt="test",
        middleware=[pii, summarizer, AuthorityGate(), approval],
    )

    assert hasattr(agent, "invoke")


def test_summarizer_triggers_on_tokens_and_keeps_recent_messages():
    assert isinstance(summarizer, SummarizationMiddleware)
    assert summarizer.trigger == ("tokens", 4000)
    assert summarizer.keep == ("messages", 20)


def test_approval_pauses_before_set_ticket_status():
    assert isinstance(approval, HumanInTheLoopMiddleware)
    assert "set_ticket_status" in approval.interrupt_on


def test_authority_gate_blocks_resolving_a_ticket_without_running_the_tool():
    gate = AuthorityGate()
    request = _request("set_ticket_status", {"ticket_id": "T-1001", "status": "resolved"})
    called = []

    def handler(_request):
        called.append(True)
        return "should not run"

    result = gate.wrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "approval" in result.content
    assert called == []  # the real tool never ran


def test_authority_gate_passes_through_non_resolving_calls():
    gate = AuthorityGate()
    request = _request("set_ticket_status", {"ticket_id": "T-1001", "status": "pending"})

    def handler(req):
        return f"ran {req.tool_call['name']}"

    assert gate.wrap_tool_call(request, handler) == "ran set_ticket_status"


def test_authority_gate_passes_through_other_tools_unconditionally():
    gate = AuthorityGate()
    request = _request("search_kb", {"query": "refund window"})

    def handler(req):
        return "delegated"

    assert gate.wrap_tool_call(request, handler) == "delegated"
