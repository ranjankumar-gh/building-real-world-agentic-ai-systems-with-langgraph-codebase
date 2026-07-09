"""Chapter 7, "Tools, Models, MCP, and create_agent" - atlas/agent.py.
Chapter 8, "The Middleware System", adds coverage for the middleware= wiring.

See "Binding the model" and "Reaching external tools with MCP". Building
`create_agent` (and `init_chat_model`) does not require a live API key -
only invoking it does - so these tests check construction, tool wiring, and
the MCP empty-tools guard, matching the no-live-call convention from
`tests/test_hello.py` (Chapter 2). No test here spins up a real MCP
client/server pair (no pytest-asyncio dependency either): `client.get_tools()`
is monkeypatched, and the async builder is driven with `asyncio.run`, the
same pattern `tests/test_graph.py` uses for `retrieve_async`."""

import asyncio

import pytest
from langchain_core.tools import tool

from atlas import agent as agent_module
from atlas.agent import (
    RESOLVE_PROMPT,
    RESOLVE_TOOLS,
    build_resolve_agent,
    build_resolve_agent_from_model_id,
    resolve_agent,
)
from atlas.middleware import AuthorityGate, approval, pii, summarizer
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
