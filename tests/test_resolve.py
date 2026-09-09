"""Chapter 17: the resolve agent mounted into Atlas's topology."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from atlas import agent as agent_module
from atlas import graph as graph_module
from atlas.breaks import ScriptedModel
from atlas.graph import build_graph
from atlas.resolve import make_resolve_node
from atlas.state import AtlasState


def _decision(route: str) -> SimpleNamespace:
    """Stand-in for a validated TriageResult - just enough shape (`.route`)
    for `triage` to read, without a live model call. Same helper shape as
    tests/test_graph.py's `_decision`, kept local rather than imported since
    no conftest.py shares fixtures across test modules in this repo."""
    return SimpleNamespace(route=route)


class FakeChatModel(BaseChatModel):
    """A BaseChatModel that returns canned replies. create_agent binds a
    real model interface, which ScriptedModel deliberately is not."""

    responses: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        reply = self.responses[0]
        return ChatResult(generations=[ChatGeneration(message=reply)])


class FakeAgent:
    """Stands in for a compiled create_agent graph: takes a messages dict,
    returns one with the reply appended."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def invoke(self, payload: dict, config: dict | None = None) -> dict:
        self.calls.append(payload)
        return {"messages": [*payload["messages"], AIMessage("resolved")]}


def test_resolve_node_passes_conversation_and_returns_only_the_reply() -> None:
    agent = FakeAgent()
    node = make_resolve_node(agent)
    state: AtlasState = {
        "messages": [{"role": "user", "content": "where is my order"}],
        "retrieved": [{"id": "kb:1", "text": "Orders ship in 2 days.", "score": 1.0}],
        "ticket": None,
        "route": "answer",
        "retrieve_attempts": 1,
        "error": None,
        "refund_done": False,
    }

    delta = node(state)

    assert agent.calls[0]["messages"] == state["messages"]
    assert [m.content for m in delta["messages"]] == ["resolved"]
    assert "retrieved" not in delta


def test_middleware_runs_when_the_agent_is_mounted_in_the_graph() -> None:
    """The assertion the repo never had: a middleware hook fires during
    graph.invoke, not only during a direct agent call."""
    fired: list[str] = []

    class ProbeMiddleware(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            fired.append("wrap_model_call")
            return handler(request)

    agent = create_agent(
        model=FakeChatModel(responses=[AIMessage("resolved")]),
        tools=[],
        system_prompt="probe",
        middleware=[ProbeMiddleware()],
    )
    graph = build_graph(
        model=ScriptedModel([AIMessage("answer")]),
        resolve_node=make_resolve_node(agent),
    )
    graph.invoke(
        {"messages": [{"role": "user", "content": "hi"}], "retrieved": []},
        {"configurable": {"thread_id": "test-a3"}},
    )

    assert fired == ["wrap_model_call"]


def test_build_resolved_graph_runs_the_mounted_agent_not_the_stub(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production assembly, in the package rather than in a test - and a
    behavioural proof rather than an internals inspection. Force triage to
    route to "answer" the same way tests/test_graph.py already does for
    `classify`, substitute a fake for `resolve_agent` before
    `build_resolved_graph()`'s lazy `from atlas.agent import resolve_agent`
    ever runs, then invoke the compiled graph for real: the fake recording a
    call and its reply landing in the final state is what "the answering
    node is the mounted agent, not atlas.graph.answer's deterministic stub"
    actually looks like at runtime. No model is constructed or invoked."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("answer")
    )
    agent = FakeAgent()
    monkeypatch.setattr(agent_module, "resolve_agent", agent)

    from atlas.resolve import build_resolved_graph

    graph = build_resolved_graph()
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "hi"}], "retrieved": []},
        {"configurable": {"thread_id": "test-a7"}},
    )

    assert agent.calls  # the mounted agent ran - not the deterministic stub
    assert result["messages"][-1].content == "resolved"
