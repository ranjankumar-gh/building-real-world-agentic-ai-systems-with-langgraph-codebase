"""Chapter 22, "Deployment and Scaling" - atlas/deploy/server.py, the graphs
`langgraph.json` serves.

Three claims, checked without a server: the served graphs compile with no
checkpointer and no store (the Agent Server supplies both, and `langgraph
dev` refuses a graph that brings its own); the served resolve node builds
the `AtlasContext` from the identity the server proved, never from the
caller's `context`; and the `monitor` graph runs Chapter 21's monitor with
the input the cron sends."""

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from atlas import graph as graph_module
from atlas.deploy import server
from atlas.security import AtlasContext


class _RecordingAgent:
    def __init__(self) -> None:
        self.contexts: list[Any] = []

    async def ainvoke(self, payload: dict, context: Any = None) -> dict:
        self.contexts.append(context)
        return {"messages": [*payload["messages"], AIMessage("resolved")]}


class _User:
    def __init__(self, identity: str, permissions: list[str]) -> None:
        self.identity = identity
        self.permissions = permissions


def _served(monkeypatch) -> tuple[Any, _RecordingAgent]:
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: SimpleNamespace(route="answer")
    )
    agent = _RecordingAgent()
    # the server would supply the store; recall and remember need one
    graph = server.build_served_graph(agent).copy(update={"store": InMemoryStore()})
    return graph, agent


def test_the_served_graphs_bring_no_checkpointer_and_no_store():
    for graph in (server.resolve, server.monitor, server.sla_watch):
        assert graph.checkpointer is None
        assert graph.store is None


def test_the_in_process_entry_keeps_its_own_persistence(monkeypatch):
    from atlas import resolve as resolve_module

    monkeypatch.setattr(resolve_module, "mount_resolve_agent", lambda: object())
    monkeypatch.setattr(resolve_module, "make_resolve_node", lambda agent: lambda s: {})
    graph = resolve_module.build_resolved_graph()

    assert isinstance(graph.checkpointer, InMemorySaver)
    assert graph.store is not None


def test_the_served_node_reads_the_role_off_the_proved_identity(monkeypatch):
    """The server puts the authenticated user in
    `config["configurable"]["langgraph_auth_user"]`; LangGraph surfaces it as
    `runtime.server_info.user`. A forged `context` from the caller is
    ignored."""
    graph, agent = _served(monkeypatch)
    user = _User("agent-7", ["role:support_agent"])

    asyncio.run(
        graph.ainvoke(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "ticket": {"id": "T-1", "amount": 10.0, "customer_id": "C-9"},
            },
            {"configurable": {"langgraph_auth_user": user}},
            context=AtlasContext(role="forged-admin", customer_id="C-other"),
        )
    )

    assert agent.contexts == [AtlasContext(role="support_agent", customer_id="C-9")]


def test_with_no_proved_identity_the_served_node_runs_anonymous(monkeypatch):
    graph, agent = _served(monkeypatch)

    asyncio.run(
        graph.ainvoke(
            {"messages": [{"role": "user", "content": "hi"}]},
            context=AtlasContext(role="support_agent", customer_id="C-1"),
        )
    )

    assert agent.contexts == [AtlasContext(role="anonymous", customer_id="unknown")]


def test_the_monitor_graph_runs_the_monitor_with_the_crons_input(monkeypatch):
    import atlas.monitor as monitor_module

    seen: list[float] = []
    monkeypatch.setattr(monitor_module, "run_quality_monitor", seen.append)

    server.monitor.invoke({"sample_rate": 0.05})

    assert seen == [0.05]


def test_the_served_resolve_node_is_async():
    """The server runs graphs with `astream`; the mounted agent is awaited,
    so its middleware run their async hooks on the server's loop."""
    node = server.resolve.builder.nodes["answer"].runnable
    assert inspect.iscoroutinefunction(node.afunc)
