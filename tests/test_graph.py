"""Chapter 4: atlas/graph.py - the Chapter 3 whiteboard wired into a real,
compiled StateGraph. Most tests here monkeypatch the helpers so node, edge
and reducer behavior can be exercised in isolation; the two `..._end_to_end`
tests at the foot of the file drive the real seeded knowledge base through
the compiled graph, standing in only for `classify` (the one live model call).

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
async node of its own; Chapter 10's own example node, "research", did not
yet exist in this repo - Chapter 17 is what finally adds it, as the mounted
map-reduce subgraph below).

Chapter 11, "Human-in-the-Loop", adds `approval_gate` in front of the
membrane `refund` crosses. `approval_gate`'s own routing logic (approve /
edit / reject / unknown-decision) is tested in isolation below by
monkeypatching `interrupt` - the same style already used for `classify` and
`search_kb` - so these tests exercise the gate's decision logic without a
real suspend/resume round trip. The end-to-end tests further down drive the
REAL `interrupt()`/`Command(resume=...)` cycle through the compiled graph,
proving the suspension is real (the run returns with `result["__interrupt__"]`
set) and that resuming lands exactly where the chapter promises."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from atlas import graph as graph_module
from atlas.breaks import ScriptedModel
from atlas.effects import RefundError
from atlas.graph import (
    ALLOWED_ROUTES,
    MAX_RETRIEVE_ATTEMPTS,
    RETRIEVE_TIMEOUT,
    AtlasState,
    answer,
    approval_gate,
    build_graph,
    derive_sources,
    escalate,
    graph,
    recall,
    refund,
    refund_already_done,
    refund_failed,
    remember,
    research,
    retrieve,
    retrieve_async,
    route_after_retrieve,
    route_from_triage,
    summarize_findings,
    triage,
    triage_with_command,
)
from atlas.memory import profile_ns
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

    assert delta == {"route": "retrieve", "retrieve_attempts": 0, "error": None}


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

        assert delta["route"] == "escalate"


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
    assert result.update == {"route": "retrieve", "retrieve_attempts": 0, "error": None}
    assert result.goto == "retrieve"


def test_triage_with_command_also_falls_back_to_escalate_on_an_off_menu_route(
    monkeypatch,
):
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("lookup_order")
    )

    result = triage_with_command(_state(messages=["hi"]))

    assert result.update["route"] == "escalate"
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

    # Wrapped in an AIMessage: add_messages would coerce a bare str into a
    # HumanMessage and file Atlas's answer as a user turn.
    [message] = delta["messages"]
    assert isinstance(message, AIMessage)
    assert message.content == "reply"


def test_retrieve_async_offloads_the_blocking_call_via_asyncio_to_thread(monkeypatch):
    """The asyncio.to_thread version from "Making it correct under load" -
    the offload happens off the event loop but the returned delta is
    identical in shape to the plain-def retrieve()."""
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: ["hit-1"])

    delta = asyncio.run(retrieve_async(_state(messages=["hi"])))

    assert delta == {"retrieved": ["hit-1"]}


def test_remember_persists_a_durable_customer_fact_to_the_store():
    """Chapter 13: `remember` writes through `runtime.store`, not through
    the checkpointer - it reaches the store the same way every node does."""
    store = InMemoryStore()
    runtime = Runtime(store=store)
    state = _state(ticket={"id": "T-1001", "customer_id": "cust-1"})

    delta = remember(state, runtime)

    assert delta == {}  # no state channel changes - the store, not state, holds the fact
    item = store.get(profile_ns("cust-1"), "plan")
    assert item.value == {"tier": "enterprise"}


def test_recall_reads_a_previously_remembered_fact_on_a_fresh_thread():
    """Exercise 1: write in one thread, read back on what is logically a new
    one - `recall` never touches thread_id, only the customer's namespace."""
    store = InMemoryStore()
    runtime = Runtime(store=store)
    state = _state(ticket={"id": "T-1001", "customer_id": "cust-1"})
    remember(state, runtime)

    delta = recall(state, runtime)

    assert delta == {"customer_plan": "enterprise"}


def test_recall_returns_unknown_when_the_store_has_never_seen_this_customer():
    store = InMemoryStore()
    runtime = Runtime(store=store)
    state = _state(ticket={"id": "T-2002", "customer_id": "cust-never-seen"})

    delta = recall(state, runtime)

    assert delta == {"customer_plan": "unknown"}


def test_recall_never_returns_a_different_customers_memory():
    """Exercise 2's isolation claim, from the node side: two customers'
    facts never cross, because the namespace is keyed by customer_id."""
    store = InMemoryStore()
    runtime = Runtime(store=store)
    remember(_state(ticket={"id": "T-1", "customer_id": "cust-a"}), runtime)

    delta = recall(_state(ticket={"id": "T-2", "customer_id": "cust-b"}), runtime)

    assert delta == {"customer_plan": "unknown"}  # cust-b has no memory of its own


def test_graph_compiles_with_the_figure_3_1_branching_topology():
    """Chapter 6 replaces the linear chain with triage/retrieve/answer/
    escalate joined by conditional edges. Chapter 10 adds `refund` - the
    checkpoint-membrane crossing - and its `error_handler` shows up as an
    internal `__error_handler__refund` pseudo-node, filtered out here the
    same way `__start__`/`__end__` are. Chapter 11 adds `approval_gate`,
    sitting in front of `refund`. Chapter 17 adds `research`, the mounted
    map-reduce subgraph - present as a node but, like `remember`/`recall`
    before it, not wired into any edge, so it is an orphan in this topology
    on purpose (LangGraph compiles unreachable nodes without error)."""
    node_names = {
        name for name in graph.get_graph().nodes if not name.startswith("__")
    }

    assert node_names == {
        "triage",
        "retrieve",
        "answer",
        "escalate",
        "approval_gate",
        "refund",
        "research",
    }


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
        lambda messages, retrieved: (
            "Refunds are available within 30 days of purchase."
        ),
    )

    config = {"configurable": {"thread_id": "test-thread-messages-accumulate"}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "refund?"}]}, config
    )

    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "refund?"
    assert isinstance(result["messages"][-1], AIMessage)
    assert result["messages"][-1].content == (
        "Refunds are available within 30 days of purchase."
    )


def test_a_query_that_keeps_coming_back_empty_retries_then_escalates_gracefully(
    monkeypatch,
):
    """The chapter's central claim, exercised end to end: an empty
    retrieval runs retrieve exactly MAX_RETRIEVE_ATTEMPTS times (the first
    attempt plus two retries) and then escalates - it never raises
    GraphRecursionError."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("retrieve")
    )
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])

    config = {"configurable": {"thread_id": "test-thread-retry-then-escalate"}}
    result = graph.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)

    assert result["retrieve_attempts"] == MAX_RETRIEVE_ATTEMPTS
    assert result["ticket"] == {"status": "escalated"}


def test_triage_resets_the_loop_guard_for_every_new_question(monkeypatch):
    """Chapter 6, "The bounded retry": triage starts each question with a
    fresh guard, whatever the previous question left in state."""
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("retrieve")
    )
    stale = _state(
        messages=["hi"],
        retrieve_attempts=MAX_RETRIEVE_ATTEMPTS,
        error="knowledge base is down",
    )

    delta = triage(stale)

    assert delta["retrieve_attempts"] == 0
    assert delta["error"] is None


def test_a_second_question_on_the_same_thread_gets_its_own_retries(monkeypatch):
    """The per-question cap survives a checkpointer. `graph` is compiled on
    an InMemorySaver, so the second invoke on one thread_id starts from the
    first question's saved state. Without triage's reset it would start at
    retrieve_attempts == MAX_RETRIEVE_ATTEMPTS, run retrieve once, and
    escalate with no retry."""
    calls: list[str] = []

    def _empty_kb(messages):
        calls.append("search")
        return []

    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("retrieve")
    )
    monkeypatch.setattr(graph_module, "search_kb", _empty_kb)
    config = {"configurable": {"thread_id": "test-thread-guard-per-question"}}

    for question in ("first question", "second question"):
        calls.clear()
        result = graph.invoke(
            {"messages": [{"role": "user", "content": question}]}, config
        )

        assert len(calls) == MAX_RETRIEVE_ATTEMPTS
        assert result["retrieve_attempts"] == MAX_RETRIEVE_ATTEMPTS
        assert result["ticket"] == {"status": "escalated"}


def test_a_recorded_failure_does_not_escalate_the_next_question(monkeypatch):
    """A stale `error` from one question must not escalate the next one on
    the same thread: triage clears it, so the second question reaches the
    (now healthy) knowledge base and answers."""
    def _boom(messages):
        raise KnowledgeBaseUnavailable("knowledge base is down")

    monkeypatch.setattr(
        graph_module, "classify", lambda messages: _decision("retrieve")
    )
    monkeypatch.setattr(
        graph_module, "compose_answer", lambda messages, retrieved: "the answer"
    )
    config = {"configurable": {"thread_id": "test-thread-stale-error"}}

    monkeypatch.setattr(graph_module, "search_kb", _boom)
    first = graph.invoke({"messages": [{"role": "user", "content": "q1"}]}, config)
    assert first["error"] == "knowledge base is down"

    hit = {"id": "doc-1", "text": "Refunds within 30 days.", "score": 1.0}
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [hit])
    second = graph.invoke({"messages": [{"role": "user", "content": "q2"}]}, config)

    assert second["error"] is None
    assert second["messages"][-1].content == "the answer"


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
    RetryPolicy so a transient failure re-runs the node instead of failing
    the whole run. Chapter 4 drops the narrowed retry_on=(ConnectionError,)
    it first shows, so the policy keeps the default predicate."""
    from langgraph.types import default_retry_on

    pregel_node = graph.nodes["retrieve"]

    assert pregel_node.retry_policy is not None
    assert pregel_node.retry_policy[0].max_attempts == 3
    assert pregel_node.retry_policy[0].retry_on is default_retry_on


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
    """End-to-end, through the Chapter 11 approval gate: triage routes to
    "refund", which now lands on `approval_gate` first. Resuming with an
    approve decision continues on to the real `refund` node, which calls
    the real charge_refund from atlas.effects - the completed run carries
    the refund message plus refund_done=True."""
    monkeypatch.setattr(graph_module, "classify", lambda messages: _decision("refund"))

    config = {"configurable": {"thread_id": "test-thread-refund-e2e"}}
    graph.invoke(
        {
            "messages": [{"role": "user", "content": "refund please"}],
            "ticket": {"id": "T-1001", "amount": 49.0},
        },
        config,
    )
    result = graph.invoke(Command(resume={"type": "approve"}), config)

    assert result["refund_done"] is True
    assert result["messages"][-1].content.startswith("Refund of $")


def test_a_refund_that_keeps_failing_exhausts_retries_then_escalates_end_to_end(
    monkeypatch,
):
    """The retry_policy/error_handler composition, end to end, past an
    approved gate: three failed attempts at charge_refund, then
    refund_failed compensates by routing to escalate - the run finishes
    gracefully instead of crashing."""
    monkeypatch.setattr(graph_module, "classify", lambda messages: _decision("refund"))
    attempts = []

    def _always_fails(key, ticket_id):
        attempts.append(key)
        raise RefundError("payment backend rejected the refund")

    monkeypatch.setattr(graph_module, "charge_refund", _always_fails)

    config = {"configurable": {"thread_id": "test-thread-refund-fails-e2e"}}
    graph.invoke(
        {
            "messages": [{"role": "user", "content": "refund please"}],
            "ticket": {"id": "T-1001", "amount": 49.0},
        },
        config,
    )
    result = graph.invoke(Command(resume={"type": "approve"}), config)

    assert len(attempts) == 3  # max_attempts on the refund node's RetryPolicy
    assert result["ticket"] == {"status": "escalated"}
    assert result["error"] == "refund failed after retries; needs manual review"


# --- Chapter 11: the approval gate ----------------------------------------


def test_approval_gate_surfaces_the_proposed_refund_and_approves_to_refund(
    monkeypatch,
):
    """"Suspend, surface, resume": the gate calls interrupt() with the
    proposed action, and an approve decision routes onward to refund with no
    state update of its own."""
    seen_payloads = []
    monkeypatch.setattr(
        graph_module,
        "interrupt",
        lambda payload: seen_payloads.append(payload) or {"type": "approve"},
    )
    state = _state(ticket={"id": "T-1001", "amount": 49.0})

    result = approval_gate(state)

    assert seen_payloads == [
        {"action": "issue_refund", "ticket_id": "T-1001", "amount": 49.0}
    ]
    assert isinstance(result, Command)
    assert result.goto == "refund"
    assert result.update is None


def test_approval_gate_rejects_and_routes_to_escalate_with_the_reason_recorded(
    monkeypatch,
):
    monkeypatch.setattr(
        graph_module,
        "interrupt",
        lambda payload: {"type": "reject", "reason": "duplicate refund request"},
    )
    state = _state(ticket={"id": "T-1001", "amount": 49.0})

    result = approval_gate(state)

    assert result.goto == "escalate"
    assert result.update == {"error": "refund rejected: duplicate refund request"}


def test_approval_gate_accepts_an_edit_within_policy_and_updates_the_ticket_amount(
    monkeypatch,
):
    """The edit decision is untrusted input, re-validated against policy -
    here it passes (the edited amount is within the original amount) and the
    gate updates `ticket` before routing to refund."""
    monkeypatch.setattr(
        graph_module, "interrupt", lambda payload: {"type": "edit", "amount": 24.0}
    )
    state = _state(ticket={"id": "T-1001", "amount": 49.0})

    result = approval_gate(state)

    assert result.goto == "refund"
    assert result.update == {"ticket": {"id": "T-1001", "amount": 24.0}}


def test_approval_gate_rejects_an_out_of_policy_edit_and_escalates_instead(
    monkeypatch,
):
    """A fat-fingered (or malicious) edit above the original amount must be
    caught the same way a hallucinated tool argument is - it never reaches
    refund."""
    monkeypatch.setattr(
        graph_module, "interrupt", lambda payload: {"type": "edit", "amount": 4900.0}
    )
    state = _state(ticket={"id": "T-1001", "amount": 49.0})

    result = approval_gate(state)

    assert result.goto == "escalate"
    assert result.update == {"error": "edited amount 4900.0 out of policy"}


def test_approval_gate_raises_on_an_unrecognized_decision_type(monkeypatch):
    monkeypatch.setattr(
        graph_module, "interrupt", lambda payload: {"type": "shrug"}
    )
    state = _state(ticket={"id": "T-1001", "amount": 49.0})

    with pytest.raises(ValueError, match="unknown decision"):
        approval_gate(state)


def test_the_refund_route_suspends_at_the_approval_gate_end_to_end(monkeypatch):
    """The durable-pause claim, exercised for real: no monkeypatched
    interrupt here - the compiled graph's own checkpointer is what makes the
    suspension possible. The run returns with `__interrupt__` set instead of
    a finished answer, carrying the exact payload the gate proposed."""
    monkeypatch.setattr(graph_module, "classify", lambda messages: _decision("refund"))

    config = {"configurable": {"thread_id": "test-thread-approval-suspend"}}
    result = graph.invoke(
        {
            "messages": [{"role": "user", "content": "refund please"}],
            "ticket": {"id": "T-1001", "amount": 49.0},
        },
        config,
    )

    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value == {
        "action": "issue_refund",
        "ticket_id": "T-1001",
        "amount": 49.0,
    }


def test_resuming_with_an_edit_re_validates_before_crossing_the_membrane(
    monkeypatch,
):
    """Exercise 2: submit an out-of-policy edit against a REAL suspended
    gate and confirm it never reaches refund - the run escalates instead,
    and refund_done stays unset."""
    monkeypatch.setattr(graph_module, "classify", lambda messages: _decision("refund"))

    config = {"configurable": {"thread_id": "test-thread-approval-bad-edit"}}
    graph.invoke(
        {
            "messages": [{"role": "user", "content": "refund please"}],
            "ticket": {"id": "T-1001", "amount": 49.0},
        },
        config,
    )
    result = graph.invoke(
        Command(resume={"type": "edit", "amount": 4900.0}), config
    )

    assert result.get("refund_done") is not True
    assert result["ticket"] == {"status": "escalated"}
    assert result["error"] == "edited amount 4900.0 out of policy"


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


# --- Chapter 17: subgraphs, parallelism, and map-reduce --------------------


def test_derive_sources_reads_the_source_list_off_the_ticket():
    """The parent-to-subgraph input adapter: AtlasState has no native
    "sources" key, so it comes from the same free-form ticket dict
    approval_gate/refund already read ticket-scoped fields from."""
    state = _state(ticket={"id": "T-1001", "sources": ["docs.internal/sla"]})

    assert derive_sources(state) == ["docs.internal/sla"]


def test_derive_sources_defaults_to_empty_when_the_ticket_has_no_sources():
    assert derive_sources(_state(ticket={"id": "T-1001"})) == []


def test_derive_sources_tolerates_a_missing_ticket_entirely():
    assert derive_sources(_state(ticket=None)) == []


def test_summarize_findings_reports_a_result_line_per_successful_source():
    message = summarize_findings(
        [{"source": "docs.internal/sla", "result": "4-hour first response."}]
    )

    assert isinstance(message, AIMessage)
    assert "docs.internal/sla: 4-hour first response." in message.content


def test_summarize_findings_reports_an_unavailable_line_for_an_error_finding():
    """Partial failure surfaces in the summary rather than being dropped -
    the reduce step's job, per "Handle partial failure in the worker, not
    around it"."""
    message = summarize_findings(
        [{"source": "missing/source", "error": "source unreachable: missing/source"}]
    )

    assert "missing/source: unavailable (source unreachable: missing/source)" in (
        message.content
    )


def test_research_node_adapts_atlas_state_into_the_subgraph_and_back():
    """End-to-end through the REAL compiled research_graph (Chapter 17's
    seeded backend, no mocking needed) - proof the wrapping node in
    atlas/graph.py actually round-trips AtlasState through ResearchState."""
    state = _state(ticket={"id": "T-1001", "sources": ["docs.internal/sla"]})

    delta = research(state)

    [message] = delta["messages"]
    assert isinstance(message, AIMessage)
    assert "docs.internal/sla" in message.content
    assert "4-hour first response" in message.content


def test_research_node_survives_an_unreachable_source_without_crashing():
    state = _state(ticket={"id": "T-1001", "sources": ["nope/does-not-exist"]})

    delta = research(state)

    assert "unavailable" in delta["messages"][0].content


def test_research_is_registered_as_a_node_but_not_wired_into_any_edge():
    """Like remember/recall before it: the chapter's own code adds the node
    (`builder.add_node("research", research)`) but names no place in the
    routing topology to reach it from."""
    assert "research" in graph.nodes
    graph_edges = graph.get_graph().edges
    assert not any(edge.source == "research" or edge.target == "research"
                   for edge in graph_edges)


def test_build_graph_mounts_a_custom_resolve_node() -> None:
    """The answering node is a seam: pass one in and the graph uses it."""
    seen: list[str] = []

    def fake_resolve(state: AtlasState) -> dict:
        seen.append("called")
        return {"messages": [AIMessage("from the mounted node")]}

    graph = build_graph(
        model=ScriptedModel([AIMessage("answer")]),
        resolve_node=fake_resolve,
    )
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "hello"}], "retrieved": []},
        {"configurable": {"thread_id": "test-a1"}},
    )
    assert seen == ["called"]
    assert result["messages"][-1].content == "from the mounted node"


def test_atlas_answers_a_knowledge_base_question_end_to_end(monkeypatch):
    """The answer path, end to end, through the real compiled graph.

    Chapter 4 promised `search_kb`/`compose_answer` would be filled in for
    real; until they were, `retrieve` raised NotImplementedError and this
    path had no coverage at all - every other test in this file monkeypatches
    the helpers, so a stubbed answer path stayed green. Only `classify` is
    stood in for here (it is the one live model call); everything downstream
    is the real seeded knowledge base.
    """
    monkeypatch.setattr(graph_module, "classify", lambda messages: _decision("retrieve"))

    config = {"configurable": {"thread_id": "test-thread-answer-e2e"}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "what is the refund window?"}]},
        config,
    )

    assert result["retrieved"], "the seeded KB should have matched"
    assert result["retrieved"][0]["text"].startswith("Refunds are available")
    assert "30 days" in result["messages"][-1].content


def test_a_knowledge_base_miss_retries_then_escalates_end_to_end(monkeypatch):
    """The other half of the same path: a real miss returns no documents, so
    `route_after_retrieve` drives the bounded retry and then escalates
    gracefully rather than composing an answer it cannot ground."""
    monkeypatch.setattr(graph_module, "classify", lambda messages: _decision("retrieve"))

    config = {"configurable": {"thread_id": "test-thread-answer-miss-e2e"}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "do you sell submarines?"}]},
        config,
    )

    assert result["retrieved"] == []
    assert result["retrieve_attempts"] == MAX_RETRIEVE_ATTEMPTS
    assert result["ticket"]["status"] == "escalated"
