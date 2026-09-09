"""Chapter 7, "Tools, Models, MCP, and create_agent" - atlas/agent.py.
Chapter 8, "The Middleware System", adds coverage for the middleware= wiring.
Chapter 12, "Context Engineering", adds coverage for the context_budget
middleware folded into the same stack.
Chapter 20, "Observability and Debugging with LangSmith", adds coverage for
`name="resolve-agent"` and the `run_resolve` attribution wrapper.

See "Binding the model" and "Reaching external tools with MCP". Building
`create_agent` (and `init_chat_model`) does not require a live API key -
only invoking it does - so these tests check construction, tool wiring, and
the MCP empty-tools guard, matching the no-live-call convention from
`tests/test_hello.py` (Chapter 2). No test here spins up a real MCP
client/server pair (no pytest-asyncio dependency either): `client.get_tools()`
is monkeypatched, and the async builder is driven with `asyncio.run`, the
same pattern `tests/test_graph.py` uses for `retrieve_async`.

`run_resolve`'s own `resolve_agent.invoke(...)` call is monkeypatched the
same way `tests/test_research.py` fakes `web_research`'s scoped agent - no
live model call, entering `trace()` needs no live LangSmith connection
either (see "Naming the fleet"), so the test proves the attribution
wrapper's own logic (tags, metadata, delegation) without either service."""

import asyncio
import functools

import pytest
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.tools import tool
from langgraph.runtime import Runtime

from atlas import agent as agent_module
from atlas.agent import (
    RESOLVE_MIDDLEWARE,
    RESOLVE_PROMPT,
    RESOLVE_TOOLS,
    build_resolve_agent,
    build_resolve_agent_from_model_id,
    context_budget,
    resolve_agent,
    run_resolve,
)
from atlas.context import ContextBudget
from atlas.middleware import AuthorityGate, approval, pii, summarizer
from atlas.security import AtlasContext
from atlas.tools import lookup_ticket, search_kb, set_ticket_status


@tool
def _fake_mcp_tool(component: str) -> str:
    """Stand-in for a tool loaded over MCP - a real BaseTool, so create_agent
    accepts it, without spinning up a real MCP client/server pair."""
    return "operational"


def test_resolve_tools_are_the_three_real_tools():
    assert RESOLVE_TOOLS == [search_kb, lookup_ticket, set_ticket_status]


def test_resolve_agent_compiles_to_an_invokable_graph_without_calling_the_model():
    assert hasattr(resolve_agent, "invoke")


def test_resolve_agent_wires_chapter_8s_middleware_stack():
    """Chapter 8's middleware= argument compiles into extra graph nodes for
    each hook-based middleware (compiling, not invoking, needs no live model
    call). PIIMiddleware and SummarizationMiddleware attach before_model
    hooks; PIIMiddleware also attaches an after_model hook; and
    HumanInTheLoopMiddleware attaches after_model. AuthorityGate's
    wrap_tool_call wraps the "tools" node in place rather than adding a
    node, so it is not visible here - see tests/test_middleware.py."""
    node_names = set(resolve_agent.nodes.keys())

    assert "PIIMiddleware[email].before_model" in node_names
    assert "PIIMiddleware[email].after_model" in node_names
    assert "SummarizationMiddleware.before_model" in node_names
    assert "HumanInTheLoopMiddleware.after_model" in node_names


def test_resolve_agent_carries_the_chapter_12_context_budget():
    """`context_budget`'s wrap_model_call adds no extra graph node (unlike
    the before_model/after_model hooks Chapter 8 checks above) - it wraps
    the existing model-call step in place - so this checks the middleware
    instance and its configured slices directly, the same way
    `tests/test_middleware.py` checks `AuthorityGate.wrap_tool_call`."""
    assert isinstance(context_budget, ContextBudget)
    assert context_budget.budget.history == 4000
    assert context_budget.budget.retrieved == 2000


def test_resolve_agent_still_compiles_with_the_context_budget_added():
    assert hasattr(resolve_agent, "invoke")


def test_build_resolve_agent_from_model_id_accepts_a_bare_string_id():
    """The string-shorthand binding: create_agent resolves the id through
    init_chat_model, with no separate init_chat_model call needed."""
    agent = build_resolve_agent_from_model_id("claude-sonnet-4-6")

    assert hasattr(agent, "invoke")


def test_resolve_prompt_tells_the_model_never_to_claim_unconfirmed_success():
    assert "Never" in RESOLVE_PROMPT or "never" in RESOLVE_PROMPT


def test_build_resolve_agent_folds_in_mcp_tools_alongside_the_local_ones(monkeypatch):
    class _FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def get_tools(self):
            return [_fake_mcp_tool]

    monkeypatch.setattr(agent_module, "MultiServerMCPClient", _FakeClient)

    agent = asyncio.run(build_resolve_agent())

    assert hasattr(agent, "invoke")


def test_build_resolve_agent_refuses_to_start_when_mcp_returns_no_tools(monkeypatch):
    """The silent-empty-get_tools failure mode from "Production
    considerations": a server that fails to connect can make get_tools()
    return an empty list without raising - Atlas must refuse to start
    rather than run with a silently shrunken authority surface."""

    class _FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def get_tools(self):
            return []

    monkeypatch.setattr(agent_module, "MultiServerMCPClient", _FakeClient)

    with pytest.raises(RuntimeError, match="no tools"):
        asyncio.run(build_resolve_agent())


# --- Chapter 20: attribution across resolve_agent's turn -------------------


def test_resolve_agent_is_named_for_the_trace_tree():
    """`name=` turns a generic AgentExecutor span into "resolve-agent" -
    the cheapest fix in the chapter."""
    assert resolve_agent.name == "resolve-agent"


def test_run_resolve_wraps_the_invoke_in_a_trace_context_and_returns_its_result(
    monkeypatch,
):
    """The trace() context adds tags/metadata once at the entry point;
    resolve_agent.invoke is faked so no live model call happens, and
    LANGSMITH_TRACING is left off so trace() stays a local no-op context
    manager - proving run_resolve's own delegation logic, not LangSmith's."""
    captured = {}

    class _FakeAgent:
        def invoke(self, inputs, config):
            captured["inputs"] = inputs
            captured["config"] = config
            return {"messages": [{"role": "assistant", "content": "done"}]}

    monkeypatch.setattr(agent_module, "resolve_agent", _FakeAgent())

    inputs = {"messages": [{"role": "user", "content": "hello"}]}
    config = {
        "configurable": {
            "route": "resolve",
            "thread_id": "t-1",
            "customer_id": "cust-1",
        }
    }

    result = run_resolve(inputs, config)

    assert result == {"messages": [{"role": "assistant", "content": "done"}]}
    assert captured["inputs"] == inputs
    assert captured["config"] == config


def test_run_resolve_requires_route_thread_id_and_customer_id_in_configurable(
    monkeypatch,
):
    """A caller that forgets to set one of these gets a loud KeyError, not a
    trace silently missing an attribution dimension."""

    class _FakeAgent:
        def invoke(self, inputs, config):
            return {}

    monkeypatch.setattr(agent_module, "resolve_agent", _FakeAgent())

    with pytest.raises(KeyError):
        run_resolve({"messages": []}, {"configurable": {}})


# --- Chapter 23: the security/cost/audit stack, in argued order ------------


def test_resolve_stack_carries_the_security_middleware_in_argued_order():
    """Chapter 23: order is a security control, not a style choice.

    `resolve_agent` is a compiled `CompiledStateGraph` with no readable
    `middleware` attribute on langgraph==1.2.6, so this asserts against
    `RESOLVE_MIDDLEWARE`, the list actually passed to `create_agent` -
    the two are the same list by construction (see atlas/agent.py).

    AuditGate must wrap outside BOTH authority gates - not sit innermost
    of the four - so a call either one refuses still reaches AuditGate's
    handler(request) call and gets logged with result_status="error",
    the same refusal tests/test_audit.py's own
    test_audit_gate_records_an_error_result_status_too already expects
    an AuditGate built in isolation to capture. RoleAuthorityGate must
    likewise sit outside AuthorityGate, so an unauthorized role is
    blocked before AuthorityGate's own approval check ever runs.

    InjectionGuard must be the INNERMOST wrap_tool_call middleware -
    wrapping only the real tool invocation - not outermost: it tags
    whatever its handler(request) call returns as untrusted content
    unconditionally, and cannot distinguish a governance refusal from
    real tool output, since both arrive as a ToolMessage. Outermost, a
    refusal from either authority gate would reach it and get wrapped in
    <untrusted-content> tags as if Atlas's own policy decision were
    untrusted tool output. See
    test_a_role_refusal_reaches_injection_guard_untagged below for the
    behavioral proof.

    ContextBudget must wrap outside TenantBudgetGuard (a different hook,
    wrap_model_call, so this is a separate ordering claim from the
    wrap_tool_call ones above) so the cumulative cap counts tokens from
    the already-trimmed per-call request, not the raw pre-trim history.
    context_budget is an instance, not a class with one name in `names`,
    so this compares positions by identity via RESOLVE_MIDDLEWARE.index
    rather than by name."""
    names = [type(m).__name__ for m in RESOLVE_MIDDLEWARE]

    assert "InjectionGuard" in names
    assert "RoleAuthorityGate" in names
    assert "TenantBudgetGuard" in names
    assert "AuditGate" in names
    assert names.index("AuditGate") < names.index("RoleAuthorityGate")
    assert names.index("AuditGate") < names.index("AuthorityGate")
    assert names.index("RoleAuthorityGate") < names.index("AuthorityGate")
    assert names.index("InjectionGuard") > names.index("RoleAuthorityGate")
    assert names.index("InjectionGuard") > names.index("AuthorityGate")
    assert names.index("InjectionGuard") > names.index("AuditGate")
    assert RESOLVE_MIDDLEWARE.index(context_budget) < names.index(
        "TenantBudgetGuard"
    )


def test_a_role_refusal_reaches_injection_guard_untagged():
    """The behavioral proof the ordering test above only asserts by index:
    drive an unauthorized tool call through the actual RESOLVE_MIDDLEWARE
    wrap_tool_call chain (AuditGate -> RoleAuthorityGate -> AuthorityGate
    -> InjectionGuard, nested exactly as create_agent nests list order,
    first = outermost) and confirm RoleAuthorityGate's refusal reaches
    the caller WITHOUT ever passing through InjectionGuard - so it is
    never wrapped in <untrusted-content> tags, which would tell the
    model to treat Atlas's own policy refusal as reference material,
    never a command, exactly backwards for something Atlas itself said.

    Only middleware that actually override wrap_tool_call participate -
    AgentMiddleware's own default raises rather than passing a call
    through untouched, so pii/context_budget/summarizer/TenantBudgetGuard
    (wrap_model_call or before_model hooks) and approval (after_model,
    confirmed by reading langchain.agents.middleware.human_in_the_loop -
    it resolves a rejection before the tools node runs, so it never nests
    with wrap_tool_call middleware at all) are excluded rather than
    invoked as no-ops."""
    hooking = [
        mw
        for mw in RESOLVE_MIDDLEWARE
        if type(mw).wrap_tool_call is not AgentMiddleware.wrap_tool_call
    ]

    def _tool_must_never_run(_request: ToolCallRequest):
        raise AssertionError("an unauthorized role must never reach the tool")

    chain = _tool_must_never_run
    for middleware in reversed(hooking):
        chain = functools.partial(middleware.wrap_tool_call, handler=chain)

    request = ToolCallRequest(
        tool_call={
            "name": "set_ticket_status",
            "args": {"ticket_id": "T-1001", "status": "resolved"},
            "id": "call-1",
        },
        tool=None,
        state=None,
        runtime=Runtime(
            context=AtlasContext(role="support_readonly", customer_id="C-1")
        ),
    )

    result = chain(request)

    assert result.status == "error"
    assert "not authorized" in result.content
    assert "<untrusted-content" not in result.content
