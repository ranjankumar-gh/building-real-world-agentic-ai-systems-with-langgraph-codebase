"""Chapter 8, "The Middleware System" - atlas/middleware.py.

See "Building Atlas's middleware stack". These tests check construction and
configuration of the three built-ins (no live model call needed to build
them, matching the no-live-call convention from `tests/test_hello.py`), and
exercise `AuthorityGate.wrap_tool_call` directly against a hand-built
`ToolCallRequest` - the one piece of this chapter's stack that is plain
Python logic rather than a wired-up built-in. The approve / reject /
never-approved tests then run the gate end to end behind `approval` (a
`RecordingApproval`) with a scripted fake model and an in-memory
checkpointer, and the PII-order test backs Exercise 1 with two probes.

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
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    ToolCallRequest,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from atlas.middleware import (
    EMAIL_PATTERN,
    AuthorityGate,
    RecordingApproval,
    approval,
    pii,
    redact_email,
    summarizer,
)


def _request(name: str, args: dict, state: dict | None = None) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "call-1"},
        tool=None,
        state={} if state is None else state,
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
    assert isinstance(approval, RecordingApproval)
    assert "set_ticket_status" in approval.interrupt_on


# --- Chapter 8, "Human approval, as a placeholder": the gate defers to the
# --- approval pause. End to end through create_agent, with a scripted fake
# --- model (no API key) and an in-memory checkpointer so the pause resumes.

_ran: list[dict] = []


@tool
def set_ticket_status(ticket_id: str, status: str) -> str:
    """Stand-in for Chapter 7's write tool; records every real run."""
    _ran.append({"ticket_id": ticket_id, "status": status})
    return f"{ticket_id} -> {status}"


class _ScriptedModel(GenericFakeChatModel):
    """GenericFakeChatModel lacks bind_tools, which create_agent calls."""

    def bind_tools(self, tools, **kwargs):
        return self


def _resolve_run(middleware: list, decision: dict | None) -> dict:
    """One `set_ticket_status(T-1001, "resolved")` proposal, then "done".
    Resumes with `decision` if one is given; returns the final state."""
    _ran.clear()
    call = {
        "name": "set_ticket_status",
        "id": "call-1",
        "args": {"ticket_id": "T-1001", "status": "resolved"},
    }
    model = _ScriptedModel(
        messages=iter([AIMessage("", tool_calls=[call]), AIMessage("done")])
    )
    agent = create_agent(
        model=model,
        tools=[set_ticket_status],
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}
    out = agent.invoke({"messages": [("user", "resolve T-1001")]}, config)
    if decision is not None:
        assert "__interrupt__" in out  # the pause fired before the tools node
        out = agent.invoke(Command(resume={"decisions": [decision]}), config)
    return out


def _tool_results(out: dict) -> list[ToolMessage]:
    return [m for m in out["messages"] if isinstance(m, ToolMessage)]


def test_an_approved_resolve_runs_through_the_gate():
    """(a) In the chapter's composed stack, a human approves the paused call
    and the tool runs - the gate does not refuse what a human approved."""
    out = _resolve_run(
        [pii, summarizer, AuthorityGate(), approval], {"type": "approve"}
    )

    assert _ran == [{"ticket_id": "T-1001", "status": "resolved"}]
    assert out["approved_calls"] == ["call-1"]
    assert _tool_results(out)[0].status == "success"


def test_an_edited_resolve_counts_as_approved():
    edited = {
        "type": "edit",
        "edited_action": {
            "name": "set_ticket_status",
            "args": {"ticket_id": "T-1001", "status": "resolved"},
        },
    }
    out = _resolve_run([AuthorityGate(), approval], edited)

    assert _ran == [{"ticket_id": "T-1001", "status": "resolved"}]
    assert out["approved_calls"] == ["call-1"]


def test_a_rejected_resolve_never_runs():
    """(b) A rejection answers the call with a ToolMessage in after_model,
    so the tools node never runs it and nothing is recorded as approved."""
    out = _resolve_run(
        [pii, summarizer, AuthorityGate(), approval],
        {"type": "reject", "message": "not yet"},
    )

    assert _ran == []
    assert out["approved_calls"] == []
    assert [m.content for m in _tool_results(out)] == ["not yet"]


def test_a_resolve_with_no_approval_layer_is_blocked():
    """(c) A write that reaches the gate without the approval step - here
    the approval middleware is not on the list at all - is refused."""
    out = _resolve_run([AuthorityGate()], None)

    assert _ran == []
    assert "approval" in _tool_results(out)[0].content


def test_a_resolve_the_approval_layer_does_not_pause_on_is_blocked():
    """(c) The approval layer is present but not configured for this tool,
    so no pause fires, no record is written, and the gate refuses."""
    unrelated = RecordingApproval(interrupt_on={"search_kb": True})
    out = _resolve_run([AuthorityGate(), unrelated], None)

    assert _ran == []
    assert _tool_results(out)[0].status == "error"


def _aresolve_run(middleware: list, decision: dict | None) -> dict:
    """`_resolve_run` on the async path: `ainvoke` makes the tools node call
    `awrap_tool_call`, which raises NotImplementedError on a middleware that
    defines only the sync hook."""
    import asyncio

    async def go() -> dict:
        _ran.clear()
        call = {
            "name": "set_ticket_status",
            "id": "call-1",
            "args": {"ticket_id": "T-1001", "status": "resolved"},
        }
        model = _ScriptedModel(
            messages=iter([AIMessage("", tool_calls=[call]), AIMessage("done")])
        )
        agent = create_agent(
            model=model,
            tools=[set_ticket_status],
            middleware=middleware,
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "t"}}
        out = await agent.ainvoke(
            {"messages": [("user", "resolve T-1001")]}, config
        )
        if decision is not None:
            assert "__interrupt__" in out
            out = await agent.ainvoke(
                Command(resume={"decisions": [decision]}), config
            )
        return out

    return asyncio.run(go())


def test_an_approved_resolve_runs_through_the_gate_under_ainvoke():
    out = _aresolve_run(
        [pii, summarizer, AuthorityGate(), approval], {"type": "approve"}
    )

    assert _ran == [{"ticket_id": "T-1001", "status": "resolved"}]
    assert out["approved_calls"] == ["call-1"]
    assert _tool_results(out)[0].status == "success"


def test_an_unapproved_resolve_is_blocked_under_ainvoke():
    out = _aresolve_run([AuthorityGate()], None)

    assert _ran == []
    assert "approval" in _tool_results(out)[0].content


# --- Chapter 8, Exercise 1: order decides which layers read the raw email,
# --- not what the model receives.


class _RawProbe(AgentMiddleware):
    """Records the latest HumanMessage as this layer's before_model sees it."""

    def __init__(self, seen: list[str]) -> None:
        super().__init__()
        self.seen = seen

    def before_model(self, state, runtime):
        last = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1]
        self.seen.append(last.content)
        return None


class _ModelInputProbe(AgentMiddleware):
    """Innermost wrap_model_call: records what the model would receive and
    answers without calling it, so no model and no API key are needed."""

    def __init__(self, seen: list[str]) -> None:
        super().__init__()
        self.seen = seen

    def wrap_model_call(self, request, handler):
        self.seen.append(request.messages[-1].content)
        return AIMessage("ok")


def _probe_order(shipped: bool) -> tuple[list[str], list[str]]:
    raw: list[str] = []
    model_in: list[str] = []
    probe = _RawProbe(raw)
    if shipped:
        stack = [pii, probe, summarizer, AuthorityGate(), approval]
    else:
        stack = [probe, summarizer, AuthorityGate(), approval, pii]
    agent = create_agent(
        model="claude-sonnet-4-6",
        tools=[set_ticket_status],
        middleware=[*stack, _ModelInputProbe(model_in)],
    )
    agent.invoke({"messages": [("user", "Refund jane.doe@example.com please")]})
    return raw, model_in


def test_pii_order_changes_what_the_summarizer_reads_not_what_the_model_gets():
    shipped_raw, shipped_model = _probe_order(shipped=True)
    moved_raw, moved_model = _probe_order(shipped=False)

    assert shipped_raw == ["Refund [REDACTED_EMAIL] please"]
    assert moved_raw == ["Refund jane.doe@example.com please"]
    assert shipped_model == moved_model == ["Refund [REDACTED_EMAIL] please"]


def test_authority_gate_blocks_an_unapproved_resolve_without_running_the_tool():
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


def test_authority_gate_lets_through_a_resolve_the_human_approved():
    """The record `approval` writes on an approve or edit decision is what
    the gate defers to: the same call id in `approved_calls` runs."""
    gate = AuthorityGate()
    request = _request(
        "set_ticket_status",
        {"ticket_id": "T-1001", "status": "resolved"},
        state={"approved_calls": ["call-1"]},
    )

    assert gate.wrap_tool_call(request, lambda req: "ran") == "ran"


def test_authority_gate_does_not_accept_another_calls_approval():
    gate = AuthorityGate()
    request = _request(
        "set_ticket_status",
        {"ticket_id": "T-1001", "status": "resolved"},
        state={"approved_calls": ["call-0"]},
    )

    result = gate.wrap_tool_call(request, lambda req: "ran")

    assert isinstance(result, ToolMessage)
    assert result.status == "error"


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

# --- Chapter 8, "A cache that ignores the full request serves wrong
# --- answers": the node-level cache LangGraph already ships, and the two
# --- ways it silently does nothing.


def _counting_cache_graph(cache, ttl=None):
    """A one-node graph whose node records every real execution, so a cache
    hit is observable as a run that did not happen."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph
    from langgraph.types import CachePolicy

    class CacheState(TypedDict):
        n: int

    runs: list[int] = []

    def work(state: CacheState) -> dict:
        runs.append(state["n"])
        return {"n": state["n"] + 1}

    builder = StateGraph(CacheState)
    builder.add_node(
        "work",
        work,
        cache_policy=CachePolicy(key_func=lambda s: str(s["n"]), ttl=ttl),
    )
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    return builder.compile(cache=cache), runs


def test_node_cache_skips_a_repeated_identical_step():
    """The feature exists and works: the same input twice runs the node
    once."""
    from langgraph.cache.memory import InMemoryCache

    graph, runs = _counting_cache_graph(InMemoryCache())

    graph.invoke({"n": 1})
    graph.invoke({"n": 1})

    assert runs == [1]


def test_a_cache_policy_without_a_cache_is_silently_ignored():
    """Trap one, and the reason to assert on cache behaviour rather than
    trust the configuration. `cache_policy=` on a node does nothing unless
    `compile(cache=...)` supplies a backend. There is no error and no
    warning: the node just runs every time, and the only symptom is a bill
    that never went down."""
    graph, runs = _counting_cache_graph(None)

    graph.invoke({"n": 1})
    graph.invoke({"n": 1})

    assert runs == [1, 1]


def test_cache_policy_ttl_is_seconds_not_minutes():
    """Trap two. `CachePolicy.ttl` is in SECONDS, while `BaseStore.put`'s
    `ttl` argument is in MINUTES (see tests/test_research.py). Two
    time-to-live settings in one framework, two different units. Reading
    one as the other is off by sixty in whichever direction hurts."""
    import time

    from langgraph.cache.memory import InMemoryCache

    graph, runs = _counting_cache_graph(InMemoryCache(), ttl=1)

    graph.invoke({"n": 1})
    graph.invoke({"n": 1})
    assert runs == [1]  # inside the window

    time.sleep(1.6)
    graph.invoke({"n": 1})

    # Expired after 1.6 seconds, which it would not be if ttl=1 meant a minute.
    assert runs == [1, 1]
