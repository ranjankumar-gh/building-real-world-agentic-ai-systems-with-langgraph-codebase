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
itself).

Chapter 10, "Durable Execution, Long-Running Workflows, and State
Migration", adds the `refund` node (Atlas's first crossing of the checkpoint
membrane), its `retry_policy` + `error_handler` (`refund_failed`,
compensating by routing to `escalate`), the additive-migration-safe
`refund_already_done` read, and `RETRIEVE_TIMEOUT` (a `TimeoutPolicy`
exercised against `retrieve_async`, since Atlas's compiled topology has no
async node of its own and the chapter's own example node, "research", does
not exist in this repo)."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from atlas import graph as graph_module
from atlas.effects import RefundError
from atlas.graph import (
    ALLOWED_ROUTES,
    MAX_RETRIEVE_ATTEMPTS,
    RETRIEVE_TIMEOUT,
    AtlasState,
    answer,
    escalate,
    graph,
    refund,
    refund_already_done,
    refund_failed,
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
        "refund_done": False,
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
    escalate joined by conditional edges. Chapter 10 adds `refund` - the
    checkpoint-membrane crossing - and its `error_handler` shows up as an
    internal `__error_handler__refund` pseudo-node, filtered out here the
    same way `__start__`/`__end__` are."""
    node_names = {
        name for name in graph.get_graph().nodes if not name.startswith("__")
    }

    assert node_names == {"triage", "retrieve", "answer", "escalate", "refund"}


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


# --- Chapter 10: durable execution, the checkpoint membrane, refund -------


def test_refund_already_done_reads_with_a_default_for_pre_chapter_10_checkpoints():
    """"State migration without downtime": a checkpoint written before this
    chapter has no `refund_done` key at all - `.get` with a default reads it
    as False instead of raising KeyError."""
    old_checkpoint_values = {
        "messages": [],
        "ticket": None,
        "route": "",
        "retrieve_attempts": 0,
        "error": None,
        # no "refund_done" key - this is what an old checkpoint looks like.
    }

    assert refund_already_done(old_checkpoint_values) is False


def test_refund_already_done_reads_true_once_the_refund_node_has_run():
    assert refund_already_done(_state(refund_done=True)) is True


def test_refund_computes_a_stable_key_and_returns_a_message_plus_refund_done(
    monkeypatch,
):
    """The node reads ticket_id + thread_id, calls the idempotent operation,
    and records `refund_done` in state so the rest of the graph can see the
    membrane was crossed."""
    seen_keys = []

    def _fake_charge(key, ticket_id):
        seen_keys.append((key, ticket_id))
        return f"Refund issued for {ticket_id}."

    monkeypatch.setattr(graph_module, "charge_refund", _fake_charge)

    state = _state(ticket={"id": "T-1001"})
    config = {"configurable": {"thread_id": "thread-refund-1"}}

    delta = refund(state, config)

    assert seen_keys == [("refund:thread-refund-1:T-1001", "T-1001")]
    assert delta["refund_done"] is True
    assert isinstance(delta["messages"][0], AIMessage)
    assert delta["messages"][0].content == "Refund issued for T-1001."


def test_refund_recomputes_the_same_key_across_two_calls_on_the_same_thread_and_ticket(
    monkeypatch,
):
    """The whole point of a stable key: a retry or a resume recomputes it
    identically, so the SAME key reaches charge_refund both times - the
    provider-side dedup (tested in tests/test_effects.py) is what actually
    stops the double charge."""
    seen_keys = []
    monkeypatch.setattr(
        graph_module,
        "charge_refund",
        lambda key, ticket_id: seen_keys.append(key) or "ok",
    )

    state = _state(ticket={"id": "T-1001"})
    config = {"configurable": {"thread_id": "thread-refund-resume"}}

    refund(state, config)
    refund(state, config)  # simulates a retry/resume of the same logical step

    assert seen_keys == [seen_keys[0], seen_keys[0]]


def test_refund_failed_compensates_by_recording_the_error_and_routing_to_escalate():
    """"When the side effect fails: compensation" - error_handler runs after
    retries are exhausted and returns a Command that both updates state and
    routes, in one move."""
    result = refund_failed(_state())

    assert isinstance(result, Command)
    assert result.update == {
        "error": "refund failed after retries; needs manual review"
    }
    assert result.goto == "escalate"


def test_refund_node_carries_a_retry_policy_targeting_refund_error_and_an_error_handler():
    """Chapter 4 forbade retry_policy on side-effecting nodes; Chapter 10
    earns it back because refund is now idempotent. error_handler is the
    compensation hook that runs once the retry policy gives up."""
    pregel_node = graph.nodes["refund"]

    assert pregel_node.retry_policy is not None
    assert pregel_node.retry_policy[0].max_attempts == 3
    assert pregel_node.retry_policy[0].retry_on == (RefundError,)
    assert pregel_node.error_handler_node == "__error_handler__refund"


def test_allowed_routes_grows_to_include_refund_across_the_membrane():
    assert ALLOWED_ROUTES == ("answer", "retrieve", "escalate", "refund")


def test_refund_charges_exactly_once_end_to_end_through_the_compiled_graph(
    monkeypatch,
):
    """End-to-end: triage routes to "refund", the node calls the real
    charge_refund from atlas.effects, and the completed run carries the
    refund message plus refund_done=True."""
    monkeypatch.setattr(graph_module, "classify", lambda messages: _decision("refund"))

    config = {"configurable": {"thread_id": "test-thread-refund-e2e"}}
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "refund please"}],
            "ticket": {"id": "T-1001"},
        },
        config,
    )

    assert result["refund_done"] is True
    assert result["messages"][-1].content.startswith("Refund of $")


def test_a_refund_that_keeps_failing_exhausts_retries_then_escalates_end_to_end(
    monkeypatch,
):
    """The retry_policy/error_handler composition, end to end: three failed
    attempts at charge_refund, then refund_failed compensates by routing to
    escalate - the run finishes gracefully instead of crashing."""
    monkeypatch.setattr(graph_module, "classify", lambda messages: _decision("refund"))
    attempts = []

    def _always_fails(key, ticket_id):
        attempts.append(key)
        raise RefundError("payment backend rejected the refund")

    monkeypatch.setattr(graph_module, "charge_refund", _always_fails)

    config = {"configurable": {"thread_id": "test-thread-refund-fails-e2e"}}
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "refund please"}],
            "ticket": {"id": "T-1001"},
        },
        config,
    )

    assert len(attempts) == 3  # max_attempts on the refund node's RetryPolicy
    assert result["ticket"] == {"status": "escalated"}
    assert result["error"] == "refund failed after retries; needs manual review"


def test_timeout_policy_is_accepted_on_an_async_node():
    """"Timeouts are async-only": TimeoutPolicy attaches cleanly to an async
    node - retrieve_async, Atlas's real async candidate - and compiles."""
    builder = StateGraph(AtlasState)
    builder.add_node("retrieve", retrieve_async, timeout=RETRIEVE_TIMEOUT)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", END)

    compiled = builder.compile()  # must not raise

    assert compiled.nodes["retrieve"].timeout == RETRIEVE_TIMEOUT


def test_timeout_policy_is_rejected_at_compile_time_on_a_sync_node():
    """The other half of the same claim: attaching `timeout=` to a sync node
    (retrieve, Atlas's real sync node) is rejected at compile, not at run
    time - the chapter's "async-only" rule enforced by LangGraph itself."""
    builder = StateGraph(AtlasState)
    builder.add_node("retrieve", retrieve, timeout=RETRIEVE_TIMEOUT)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", END)

    with pytest.raises(ValueError, match="async"):
        builder.compile()
