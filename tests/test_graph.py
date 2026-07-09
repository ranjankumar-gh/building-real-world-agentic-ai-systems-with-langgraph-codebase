"""Chapter 4: atlas/graph.py - the Chapter 3 whiteboard wired into a real,
compiled StateGraph. helpers stay stubs (NotImplementedError) until
Chapter 7, so these tests exercise node/edge/reducer behavior with
monkeypatched stubs rather than a real knowledge base or model."""

import asyncio

import pytest
from langchain_core.messages import AIMessage

from atlas import graph as graph_module
from atlas.graph import AtlasState, answer, graph, retrieve, retrieve_async, triage


def _state(**overrides) -> AtlasState:
    base: AtlasState = {"messages": [], "ticket": None, "retrieved": [], "route": ""}
    base.update(overrides)
    return base


def test_triage_calls_classify_and_wraps_its_result_in_a_delta(monkeypatch):
    monkeypatch.setattr(graph_module, "classify", lambda messages: "retrieve")

    delta = triage(_state(messages=["hi"]))

    assert delta == {"route": "retrieve"}


def test_retrieve_calls_search_kb_and_wraps_its_result_in_a_delta(monkeypatch):
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: ["hit-1"])

    delta = retrieve(_state(messages=["hi"]))

    assert delta == {"retrieved": ["hit-1"]}


def test_answer_calls_compose_answer_and_wraps_its_result_in_a_delta(monkeypatch):
    monkeypatch.setattr(
        graph_module, "compose_answer", lambda messages, retrieved: "reply"
    )

    delta = answer(_state(messages=["hi"], retrieved=["hit-1"]))

    assert delta == {"messages": ["reply"]}


def test_retrieve_async_offloads_the_blocking_call_via_asyncio_to_thread(monkeypatch):
    """The asyncio.to_thread version from "Making it correct under load" -
    the offload happens off the event loop but the returned delta is
    identical in shape to the plain-def retrieve()."""
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: ["hit-1"])

    delta = asyncio.run(retrieve_async(_state(messages=["hi"])))

    assert delta == {"retrieved": ["hit-1"]}


def test_graph_compiles_with_the_linear_triage_retrieve_answer_topology():
    node_names = set(graph.get_graph().nodes) - {"__start__", "__end__"}

    assert node_names == {"triage", "retrieve", "answer"}


def test_invoking_the_compiled_graph_propagates_the_classify_stub_until_chapter_7():
    """No StateGraph wiring can paper over a stub: the runtime runs triage
    first, which still calls the real (unimplemented) classify()."""
    with pytest.raises(NotImplementedError):
        graph.invoke({"messages": [{"role": "user", "content": "hi"}]})


def test_messages_channel_accumulates_via_add_messages_instead_of_clobbering(
    monkeypatch,
):
    """The point of Annotated[list, add_messages]: the reply the answer node
    returns is APPENDED to the conversation, not swapped in for it."""
    monkeypatch.setattr(graph_module, "classify", lambda messages: "answer")
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])
    monkeypatch.setattr(
        graph_module,
        "compose_answer",
        lambda messages, retrieved: AIMessage(
            content="Refunds are available within 30 days of purchase."
        ),
    )

    result = graph.invoke({"messages": [{"role": "user", "content": "refund?"}]})

    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "refund?"
    assert result["messages"][-1].content == (
        "Refunds are available within 30 days of purchase."
    )


def test_retry_policy_is_attached_to_the_retrieve_node():
    """First look at durable execution (Chapter 10): retrieve carries a
    RetryPolicy so a transient ConnectionError re-runs the node instead of
    failing the whole run."""
    pregel_node = graph.nodes["retrieve"]

    assert pregel_node.retry_policy is not None
    assert pregel_node.retry_policy[0].max_attempts == 3
    assert pregel_node.retry_policy[0].retry_on == (ConnectionError,)
