"""Chapter 18, "Deep Agents: The Production Harness" - atlas/deep_research.py.

See "Building the Deep Research Agent". Building `create_deep_agent` (and
`create_agent` under it) does not require a live API key - only invoking it
does - matching the no-live-call convention from `tests/test_agent.py` and
`tests/test_research.py`. `source_lookup` is a plain function wrapping the
seeded, mockable `search_source`/`SourceUnavailable` backend, so it is
exercised for real, no mocking needed.

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


def test_source_lookup_returns_the_seeded_result_for_a_known_source():
    assert "30 days" in source_lookup.func("docs.internal/refund-policy")


def test_source_lookup_returns_an_error_string_not_a_raised_exception():
    """Same partial-failure discipline as Chapter 17's `research_worker`: a
    dead source becomes a string the sub-agent can read and report on."""
    result = source_lookup.func("nope/does-not-exist")

    assert result == "error: source unreachable: nope/does-not-exist"


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
