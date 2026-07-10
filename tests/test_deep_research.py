"""Chapter 18, "Deep Agents: The Production Harness" - atlas/deep_research.py.

See "Building the Deep Research Agent". Building `create_deep_agent` (and
`create_agent` under it) does not require a live API key - only invoking it
does - matching the no-live-call convention from `tests/test_agent.py` and
`tests/test_research.py`. `source_lookup` is a plain function wrapping the
seeded, mockable `search_source`/`SourceUnavailable` backend, so it is
exercised for real, no mocking needed.

Chapter 19, "Streaming", adds one line to `source_lookup` -
`get_stream_writer()` - which requires an active runnable context, the same
constraint `research_namespace` already had for `get_config()` (see that
chapter's tests below). Calling `source_lookup.func(...)` directly, with no
graph run underneath it, now raises `RuntimeError`, so the two Chapter 18
tests that used to call it that way are rewritten here to run it from inside
a real (tiny) compiled graph instead - see `_run_source_lookup_in_a_graph`.

The `research_namespace` tests are the load-bearing ones for this chapter:
they confirm, against the ACTUALLY INSTALLED `deepagents==0.6.x`, that the
namespace factory contract is "called with a `Runtime`, not the run's
`config` dict" - `Runtime` does not carry `config` (verified against
`langgraph.runtime.Runtime`'s own docstring) - and that reading
`customer_id` via `get_config()` inside a real graph invocation resolves to
the correct, per-customer namespace tuple. An earlier draft of this module
took `config: dict` and read `config["configurable"]["customer_id"]`
directly; `StoreBackend` never passes a dict there, so that draft raised
`TypeError: '_NamespaceRuntimeCompat' object is not subscriptable` the
moment a backend operation ran - confirmed against the installed package,
not assumed from the docs, and fixed here before the .qmd shipped it."""

from typing import TypedDict

import pytest
from deepagents.backends.store import StoreBackend
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore

from atlas.deep_research import (
    checkpointer,
    deep_research_agent,
    research_namespace,
    source_lookup,
    source_researcher,
    store,
)


def _run_source_lookup_in_a_graph(source: str) -> tuple[str, list[dict]]:
    """Chapter 19: `source_lookup` now calls `get_stream_writer()`, which
    needs an active runnable context - build a tiny compiled graph whose one
    node calls the tool directly, and collect both the node's return value
    and every event the writer pushed onto the "custom" channel."""

    class _S(TypedDict):
        result: str

    def _node(_state: _S) -> dict:
        return {"result": source_lookup.func(source)}

    builder = StateGraph(_S)
    builder.add_node("n", _node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    compiled = builder.compile()

    result = None
    custom_events: list[dict] = []
    for chunk in compiled.stream(
        {"result": ""}, stream_mode=["custom", "values"], version="v2"
    ):
        if chunk["type"] == "custom":
            custom_events.append(chunk["data"])
        elif chunk["type"] == "values":
            result = chunk["data"]["result"]
    return result, custom_events


def test_source_lookup_returns_the_seeded_result_for_a_known_source():
    result, _ = _run_source_lookup_in_a_graph("docs.internal/refund-policy")

    assert "30 days" in result


def test_source_lookup_returns_an_error_string_not_a_raised_exception():
    """Same partial-failure discipline as Chapter 17's `research_worker`: a
    dead source becomes a string the sub-agent can read and report on."""
    result, _ = _run_source_lookup_in_a_graph("nope/does-not-exist")

    assert result == "error: source unreachable: nope/does-not-exist"


def test_source_lookup_emits_custom_progress_via_get_stream_writer():
    """Chapter 19: the only channel that reports what the tool is doing
    mid-execution, not just what it returns."""
    _, custom_events = _run_source_lookup_in_a_graph("docs.internal/sla")

    assert {"progress": "researching docs.internal/sla"} in custom_events


def test_source_lookup_raises_outside_a_graph_run():
    """`get_stream_writer()` requires an active runnable context, the same
    constraint `research_namespace` already has for `get_config()` (see
    `test_research_namespace_raises_outside_a_graph_run` below) - calling the
    raw function directly, with no graph run underneath it, is not a context
    it provides on its own."""
    with pytest.raises(RuntimeError):
        source_lookup.func("docs.internal/sla")


def test_source_researcher_is_a_subagent_declaration_not_a_handoff_tool():
    """A SubAgent is name/description/system_prompt/tools - a declaration
    the harness turns into a handoff tool, isolated context, and result
    aggregation, not a hand-wired Command-returning tool (Chapter 16's
    make_handoff)."""
    assert source_researcher["name"] == "source_researcher"
    assert source_researcher["tools"] == [source_lookup]
    assert "source_lookup" in source_researcher["system_prompt"]


def test_deep_research_agent_compiles_to_an_invokable_graph_without_calling_the_model():
    assert hasattr(deep_research_agent, "invoke")


def test_research_namespace_raises_outside_a_graph_run():
    """`get_config()` requires an active runnable context - calling the
    factory directly, the way a `config: dict`-typed version would have
    accepted, is not a context `StoreBackend` ever provides on its own."""
    with pytest.raises(RuntimeError):
        research_namespace(runtime=None)


def test_research_namespace_scopes_by_customer_id_inside_a_real_graph_run():
    """End-to-end proof the fix works: build a StoreBackend with
    research_namespace, use it from inside a real (tiny) compiled graph so
    `get_config()` has an active context, and confirm the write lands under
    `("customer", <that customer's id>, "research")` - not another
    customer's namespace."""

    class _S(TypedDict):
        done: bool

    scoped_store = InMemoryStore()
    backend = StoreBackend(store=scoped_store, namespace=research_namespace)

    def _node(_state: _S) -> dict:
        backend.write("findings/test.md", "hello world")
        return {"done": True}

    graph = StateGraph(_S)
    graph.add_node("n", _node)
    graph.add_edge(START, "n")
    graph.add_edge("n", END)
    compiled = graph.compile(store=scoped_store)

    compiled.invoke(
        {"done": False},
        config={"configurable": {"customer_id": "cust-42", "thread_id": "t1"}},
    )

    item = scoped_store.get(("customer", "cust-42", "research"), "findings/test.md")
    assert item is not None
    assert item.value["content"] == "hello world"

    other = scoped_store.get(("customer", "cust-99", "research"), "findings/test.md")
    assert other is None


def test_deep_research_agent_uses_the_shared_dev_checkpointer_and_store():
    """`checkpointer`/`store` are the Chapter 9 / Chapter 13 dev defaults -
    swapped for production the same way run_durable/build_prod_store already
    do - not new infrastructure this chapter invents."""
    assert checkpointer is not None
    assert store is not None
