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
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

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
from atlas.containment import revoke
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
    `approval` (a RecordingApproval, Chapter 8's HumanInTheLoopMiddleware
    subclass) attaches after_model. AuthorityGate's
    wrap_tool_call wraps the "tools" node in place rather than adding a
    node, so it is not visible here - see tests/test_middleware.py."""
    node_names = set(resolve_agent.nodes.keys())

    assert "PIIMiddleware[email].before_model" in node_names
    assert "PIIMiddleware[email].after_model" in node_names
    assert "SummarizationMiddleware.before_model" in node_names
    assert "RecordingApproval.after_model" in node_names


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
    """`name=` turns the default "LangGraph" span into "resolve-agent" -
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
            "thread_id": "t-1",
            "customer_id": "cust-1",
        }
    }

    result = run_resolve(inputs, config)

    assert result == {"messages": [{"role": "assistant", "content": "done"}]}
    assert captured["inputs"] == inputs
    assert captured["config"] == config


def test_run_resolve_requires_thread_id_and_customer_id_in_configurable(
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
    rather than by name.

    RevocationGate must sit at index 0. That makes it the outermost of
    the three wrap_model_call entries, so a revoked subject never reaches
    context_budget's trim or TenantBudgetGuard's spend write. It does not
    nest with the wrap_tool_call gates - it has no wrap_tool_call method -
    and AuditGate writes no record for a revoked subject because raising
    inside the "model" node means the "tools" node is never reached, which
    is node order rather than list position. Index 0 buys nothing against
    the before_model hooks (pii, summarizer); those run before any wrapped
    model call wherever the list puts them."""
    names = [type(m).__name__ for m in RESOLVE_MIDDLEWARE]

    assert names[0] == "RevocationGate"
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
            context=AtlasContext(role="support_readonly", customer_id="C-1"),
            store=InMemoryStore(),  # the graph's store, where AuditGate writes
        ),
    )

    result = chain(request)

    assert result.status == "error"
    assert "not authorized" in result.content
    assert "<untrusted-content" not in result.content


# --- Chapter 23: every gate has its async twin ------------------------------


class _ToolCallingFake(GenericFakeChatModel):
    """A scripted model create_agent can bind tools to (no API key)."""

    def bind_tools(self, tools, **kwargs):
        return self


def _scripted(*calls: dict) -> _ToolCallingFake:
    replies = [AIMessage("", tool_calls=[call]) for call in calls]
    return _ToolCallingFake(messages=iter([*replies, AIMessage("done")]))


def _stack_agent(model: _ToolCallingFake, store: InMemoryStore):
    return create_agent(
        model=model,
        tools=RESOLVE_TOOLS,
        system_prompt=RESOLVE_PROMPT,
        context_schema=AtlasContext,
        middleware=RESOLVE_MIDDLEWARE,
        store=store,
    )


_SEARCH = {"name": "search_kb", "args": {"query": "refund window"}, "id": "call-s"}


def test_the_whole_stack_runs_under_ainvoke_and_every_gate_acts():
    """Before the twins, the first model call under ainvoke raised
    NotImplementedError (create_agent puts a sync-only wrap hook in the async
    chain). Now: the budget is charged, the call is audited, and the result
    comes back scanned and tagged."""
    store = InMemoryStore()
    agent = _stack_agent(_scripted(_SEARCH), store)

    out = asyncio.run(
        agent.ainvoke(
            {"messages": [("user", "how long is the refund window?")]},
            context=AtlasContext(role="support_agent", customer_id="C-1"),
        )
    )

    tool_result = next(m for m in out["messages"] if m.type == "tool")
    assert tool_result.content.startswith('<untrusted-content source="search_kb">')
    assert store.get(("audit", "C-1"), "call-s").value["result_status"] == "success"
    spent = store.search(("customer", "C-1", "budget"))
    assert spent and spent[0].value["tokens"] > 0


def test_under_ainvoke_the_role_gate_refuses_and_the_refusal_is_audited():
    store = InMemoryStore()
    agent = _stack_agent(_scripted(_SEARCH), store)

    out = asyncio.run(
        agent.ainvoke(
            {"messages": [("user", "how long is the refund window?")]},
            context=AtlasContext(role="anonymous", customer_id="C-1"),
        )
    )

    tool_result = next(m for m in out["messages"] if m.type == "tool")
    assert tool_result.status == "error"
    assert "not authorized" in tool_result.content
    assert store.get(("audit", "C-1"), "call-s").value["result_status"] == "error"


def test_under_ainvoke_a_revoked_subject_is_stopped_before_any_tool_runs():
    store = InMemoryStore()
    revoke(store, "C-1", reason="operator halt")
    agent = _stack_agent(_scripted(_SEARCH), store)

    with pytest.raises(RuntimeError, match="authority revoked"):
        asyncio.run(
            agent.ainvoke(
                {"messages": [("user", "hello")]},
                context=AtlasContext(role="support_agent", customer_id="C-1"),
            )
        )
    assert store.search(("audit", "C-1")) == []


def test_the_mcp_builder_carries_the_same_gates(monkeypatch):
    """An MCP result is untrusted content too, so the MCP-connected agent
    gets RESOLVE_MIDDLEWARE (InjectionGuard included) and AtlasContext."""

    class _FakeClient:
        def __init__(self, *_args, **_kwargs):
            pass

        async def get_tools(self):
            return [_fake_mcp_tool]

    seen: dict = {}
    real_create_agent = agent_module.create_agent

    def _spy(**kwargs):
        seen.update(kwargs)
        return real_create_agent(**kwargs)

    monkeypatch.setattr(agent_module, "MultiServerMCPClient", _FakeClient)
    monkeypatch.setattr(agent_module, "create_agent", _spy)

    asyncio.run(build_resolve_agent())

    assert seen["middleware"] is RESOLVE_MIDDLEWARE
    assert seen["context_schema"] is AtlasContext
