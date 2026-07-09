"""Chapter 4: atlas/graph.py - the Chapter 3 whiteboard wired into a real,
compiled StateGraph. helpers stay stubs (NotImplementedError) until
Chapter 7, so these tests exercise node/edge/reducer behavior with
monkeypatched stubs rather than a real knowledge base or model.

Chapter 6, "Conditional Edges and Dynamic Control Flow", adds the branching
topology: route validation in `triage`, the two conditional routing
functions, the bounded `retrieve` retry, the `escalate` node, and the
`Command`-based `triage_with_command` alternative.

Chapter 7, "Tools, Models, MCP, and create_agent", replaces `classify` with
a real, structured-output version (atlas.triage.classify) that returns a
validated TriageResult instead of a raw string - `triage`/`triage_with_command`
now read `.route` off that object. These tests monkeypatch `classify` with a
small stand-in (`_decision`) that exposes the same `.route` attribute, so the
routing-boundary behavior can still be exercised without a live model call.
`KnowledgeBaseUnavailable` also moves, to atlas.tools.

Chapter 9, "Persistence and Checkpointing", compiles `graph` onto
`InMemorySaver`, which makes `thread_id` required in `config["configurable"]`
on every `invoke` call - the end-to-end tests below that call `graph.invoke`
directly now pass a `config` with a per-test `thread_id` for exactly that
reason (see tests/test_run.py for the checkpointer/thread_id behavior
itself)."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langgraph.types import Command

from atlas import graph as graph_module
from atlas.graph import (
    ALLOWED_ROUTES,
    MAX_RETRIEVE_ATTEMPTS,
    AtlasState,
    answer,
    escalate,
    graph,
    retrieve,
    retrieve_async,
    route_after_retrieve,
    route_from_triage,
    triage,
    triage_with_command,
)
from atlas.tools import KnowledgeBaseUnavailable


def _state(**overrides) -> AtlasState:
    base: AtlasState = {
        "messages": [],
        "ticket": None,
        "retrieved": [],
        "route": "",
        "retrieve_attempts": 0,
        "error": None,
    }
    base.update(overrides)
    return base


def _decision(route: str):
    """Stand-in for a validated TriageResult - just enough shape (`.route`)
    for `triage`/`triage_with_command` to read, without going through the
    real Pydantic-validated agent or a live model call."""
    return SimpleNamespace(route=route)


def test_triage_calls_classify_and_wraps_its_result_in_a_delta(monkeypatch):
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("retrieve")
    )

    delta = triage(_state(messages=["hi"]))

    assert delta == {"route": "retrieve"}


def test_triage_falls_back_to_escalate_when_the_model_proposes_an_off_menu_route(
    monkeypatch,
):
    """The routing boundary still holds as defense in depth even though a
    real TriageResult's `.route` is already Pydantic-constrained to the
    three legal values: an off-menu value on the object still collapses to
    the safe default rather than becoming an invalid transition."""
    for proposed in ("lookup_order", "", "RETRIEVE ", "Let me check on that."):
        monkeypatch.setattr(
            graph_module, "classify", lambda messages, p=proposed: _decision(p)
        )

        delta = triage(_state(messages=["hi"]))

        assert delta == {"route": "escalate"}


def test_triage_never_proposes_a_route_outside_the_allowed_set(monkeypatch):
    for allowed in ALLOWED_ROUTES:
        monkeypatch.setattr(
            graph_module, "classify", lambda messages, a=allowed: _decision(a)
        )

        delta = triage(_state(messages=["hi"]))

        assert delta["route"] == allowed


def test_triage_with_command_updates_state_and_names_the_next_node_together(
    monkeypatch,
):
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("retrieve")
    )

    result = triage_with_command(_state(messages=["hi"]))

    assert isinstance(result, Command)
    assert result.update == {"route": "retrieve"}
    assert result.goto == "retrieve"


def test_triage_with_command_also_falls_back_to_escalate_on_an_off_menu_route(
    monkeypatch,
):
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("lookup_order")
    )

    result = triage_with_command(_state(messages=["hi"]))

    assert result.update == {"route": "escalate"}
    assert result.goto == "escalate"


def test_route_from_triage_reads_the_route_key_and_decides_nothing_else():
    for route in ALLOWED_ROUTES:
        assert route_from_triage(_state(route=route)) == route


def test_route_after_retrieve_answers_once_results_come_back():
    state = _state(retrieved=["hit-1"], retrieve_attempts=1)

    assert route_after_retrieve(state) == "answer"


def test_route_after_retrieve_retries_while_under_the_cap():
    state = _state(retrieved=[], retrieve_attempts=MAX_RETRIEVE_ATTEMPTS - 1)

    assert route_after_retrieve(state) == "retrieve"


def test_route_after_retrieve_escalates_once_the_cap_is_reached():
    """The bounded retry's exit: this is what stands between a stuck
    retrieval loop and a GraphRecursionError."""
    state = _state(retrieved=[], retrieve_attempts=MAX_RETRIEVE_ATTEMPTS)

    assert route_after_retrieve(state) == "escalate"


def test_route_after_retrieve_escalates_on_a_recorded_error_even_with_attempts_left():
    state = _state(retrieved=[], retrieve_attempts=1, error="knowledge base is down")

    assert route_after_retrieve(state) == "escalate"


def test_escalate_returns_a_delta_only_and_does_not_touch_the_input_state():
    state = _state(ticket=None)

    delta = escalate(state)

    assert delta == {"ticket": {"status": "escalated"}}
    assert state["ticket"] is None


def test_retrieve_calls_search_kb_and_wraps_its_result_in_a_delta(monkeypatch):
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: ["hit-1"])

    delta = retrieve(_state(messages=["hi"]))

    assert delta == {"retrieved": ["hit-1"], "retrieve_attempts": 1}


def test_retrieve_increments_the_attempt_counter_on_every_call(monkeypatch):
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])

    delta = retrieve(_state(messages=["hi"], retrieve_attempts=2))

    assert delta == {"retrieved": [], "retrieve_attempts": 3}


def test_retrieve_records_a_knowledge_base_failure_instead_of_crashing(monkeypatch):
    def _boom(messages):
        raise KnowledgeBaseUnavailable("knowledge base is down")

    monkeypatch.setattr(graph_module, "search_kb", _boom)

    delta = retrieve(_state(messages=["hi"]))

    assert delta == {"error": "knowledge base is down"}


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


def test_graph_compiles_with_the_figure_3_1_branching_topology():
    """Chapter 6 replaces the linear chain with triage/retrieve/answer/
    escalate joined by conditional edges."""
    node_names = set(graph.get_graph().nodes) - {"__start__", "__end__"}

    assert node_names == {"triage", "retrieve", "answer", "escalate"}


def test_messages_channel_accumulates_via_add_messages_instead_of_clobbering(
    monkeypatch,
):
    """The point of Annotated[list, add_messages]: the reply the answer node
    returns is APPENDED to the conversation, not swapped in for it. Routing
    straight to "answer" also proves triage's conditional edge can bypass
    retrieve entirely."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("answer")
    )
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])
    monkeypatch.setattr(
        graph_module,
        "compose_answer",
        lambda messages, retrieved: AIMessage(
            content="Refunds are available within 30 days of purchase."
        ),
    )

    config = {"configurable": {"thread_id": "test-thread-messages-accumulate"}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "refund?"}]}, config
    )

    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "refund?"
    assert result["messages"][-1].content == (
        "Refunds are available within 30 days of purchase."
    )


def test_a_query_that_keeps_coming_back_empty_retries_then_escalates_gracefully(
    monkeypatch,
):
    """The chapter's central claim, exercised end to end: an empty
    retrieval retries exactly MAX_RETRIEVE_ATTEMPTS times and then escalates -
    it never raises GraphRecursionError."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("retrieve")
    )
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])

    config = {"configurable": {"thread_id": "test-thread-retry-then-escalate"}}
    result = graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)

    assert result["retrieve_attempts"] == MAX_RETRIEVE_ATTEMPTS
    assert result["ticket"] == {"status": "escalated"}


def test_a_failing_knowledge_base_routes_to_escalate_instead_of_a_fake_answer(
    monkeypatch,
):
    def _boom(messages):
        raise KnowledgeBaseUnavailable("knowledge base is down")

    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("retrieve")
    )
    monkeypatch.setattr(graph_module, "search_kb", _boom)

    config = {"configurable": {"thread_id": "test-thread-kb-failure"}}
    result = graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)

    assert result["error"] == "knowledge base is down"
    assert result["ticket"] == {"status": "escalated"}


def test_an_off_menu_triage_route_escalates_without_ever_reaching_a_bad_node(
    monkeypatch,
):
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("lookup_order")
    )

    config = {"configurable": {"thread_id": "test-thread-off-menu-route"}}
    result = graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)

    assert result["route"] == "escalate"
    assert result["ticket"] == {"status": "escalated"}


def test_retry_policy_is_attached_to_the_retrieve_node():
    """First look at durable execution (Chapter 10): retrieve carries a
    RetryPolicy so a transient ConnectionError re-runs the node instead of
    failing the whole run."""
    pregel_node = graph.nodes["retrieve"]

    assert pregel_node.retry_policy is not None
    assert pregel_node.retry_policy[0].max_attempts == 3
    assert pregel_node.retry_policy[0].retry_on == (ConnectionError,)
