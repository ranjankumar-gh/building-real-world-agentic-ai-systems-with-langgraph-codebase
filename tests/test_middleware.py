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
folding both flags onto the single `pii` instance below.

Chapter 20, "Observability and Debugging with LangSmith", adds `EMAIL_PATTERN`
and `redact_email`, and wires `EMAIL_PATTERN.pattern` into `pii`'s own
`detector=`. See "The PII redaction ordering bug, made concrete": the
chapter's first draft passed `redact_email` itself (a `Callable[[str],
str]`) as `detector=`, which does not satisfy `PIIMiddleware`'s actual
contract (`Callable[[str], list[PIIMatch]] | str | None`) - confirmed
against the installed `langchain==1.3.0` build, `_process_content` raises
`AttributeError: 'str' object has no attribute 'get'` the moment content is
scanned. `test_pii_detector_is_a_regex_pattern_string_not_a_broken_callable`
guards against that regression."""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallRequest,
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool

from atlas.middleware import EMAIL_PATTERN, AuthorityGate, approval, pii, redact_email, summarizer


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


# --- Chapter 20: trace/wire PII redaction share one EMAIL_PATTERN ----------


def test_redact_email_masks_every_address_in_the_text():
    assert redact_email("reach jane@example.com or john@example.com") == (
        "reach [EMAIL] or [EMAIL]"
    )


def test_redact_email_leaves_text_without_an_email_untouched():
    assert redact_email("no email here") == "no email here"


def test_pii_detector_is_a_regex_pattern_string_not_a_broken_callable():
    """Guards the chapter's real bug: PIIMiddleware's detector= contract is
    `Callable[[str], list[PIIMatch]] | str | None`, not `Callable[[str],
    str]`. Passing redact_email itself (which returns a redacted STRING)
    breaks the moment content is scanned - the fix is passing
    EMAIL_PATTERN.pattern (a plain regex string) instead."""
    assert pii.detector is not redact_email
    assert isinstance(EMAIL_PATTERN.pattern, str)


def test_pii_scans_content_without_raising_with_the_fixed_detector():
    """End-to-end proof the fix works: scanning real content through pii's
    configured detector must not raise - the exact failure mode the
    broken `detector=redact_email` draft hit."""
    redacted_text, matches = pii._process_content("contact me at a@b.com")

    assert redacted_text == "contact me at [REDACTED_EMAIL]"
    assert matches[0]["value"] == "a@b.com"
