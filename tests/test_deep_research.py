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

import re
from collections import Counter
from typing import TypedDict

import pytest
from deepagents import create_deep_agent
from deepagents.backends.store import StoreBackend
from langchain.tools import tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.base import _namespace_to_text

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


# --- Chapter 18, "Production considerations": research_namespace refuses an
# --- id that could widen a PostgresStore prefix match (Chapter 13's SAFE_ID).


def _namespace_in_graph(customer_id: str) -> tuple[str, ...]:
    """Call research_namespace inside a real (tiny) graph run, so
    get_config() has the customer_id in context."""

    class _S(TypedDict):
        ns: tuple

    def _node(_state: _S) -> dict:
        return {"ns": research_namespace(runtime=None)}

    builder = StateGraph(_S)
    builder.add_node("n", _node)
    builder.add_edge(START, "n")
    builder.add_edge("n", END)
    out = builder.compile().invoke(
        {"ns": ()}, config={"configurable": {"customer_id": customer_id}}
    )
    return out["ns"]


def _postgres_prefix_matches(prefix: tuple[str, ...], stored: tuple[str, ...]) -> bool:
    """PostgresStore's search: `store.prefix LIKE '<dot-joined prefix>%'`,
    where "_" matches any one character and "%" any run."""
    pattern = _namespace_to_text(prefix) + "%"
    regex = "".join(
        ".*" if c == "%" else "." if c == "_" else re.escape(c) for c in pattern
    )
    return re.fullmatch(regex, _namespace_to_text(stored), flags=re.DOTALL) is not None


@pytest.mark.parametrize("bad_id", ["cust_42", "cust.42", "cust%", "cust-42.research"])
def test_research_namespace_refuses_an_unsafe_customer_id(bad_id):
    with pytest.raises(ValueError, match="unsafe customer id"):
        _namespace_in_graph(bad_id)


def test_unsafe_id_would_leak_on_a_postgres_prefix_match():
    """Why the check exists: unchecked, "cust_42" lists "cust-42"'s files."""
    assert _postgres_prefix_matches(
        ("customer", "cust_42", "research"), ("customer", "cust-42", "research")
    )


def test_a_safe_id_cannot_prefix_match_a_longer_id():
    """The 12-vs-123 case: the namespace ends in "research", so customer
    12's prefix never reaches customer 123's files."""
    ns_12 = _namespace_in_graph("12")
    ns_123 = _namespace_in_graph("123")

    assert ns_12 == ("customer", "12", "research")
    assert not _postgres_prefix_matches(ns_12, ns_123)
    assert _postgres_prefix_matches(ns_12, ns_12)


# --- "The task tool fans out through Send" and "StateBackend vs StoreBackend":
# --- a content-driven fake model (deterministic under concurrent sub-agents)
# --- drives create_deep_agent with the chapter's shape. No network.

_lookups: Counter = Counter()
_fail_once: set[str] = set()


@tool
def flaky_lookup(source: str) -> str:
    """Look up one source; raises once for a source in _fail_once."""
    _lookups[source] += 1
    if source in _fail_once:
        _fail_once.discard(source)
        raise RuntimeError(f"source crashed: {source}")
    return f"finding for {source}"


def _call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": f"{name}-{call_id}"}


class _ResearchModel(BaseChatModel):
    """Main agent: three `task` calls in one message, then `ls /findings`,
    then a report (a "list" request only lists). Sub-agent: look up its
    source, write /findings/<source>.md, finish."""

    @property
    def _llm_type(self) -> str:
        return "scripted-research"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = " ".join(
            str(m.content) for m in messages if isinstance(m, SystemMessage)
        )
        first = next(m for m in messages if isinstance(m, HumanMessage))
        last = messages[-1]
        seen = [m.name for m in messages if isinstance(m, ToolMessage)]
        if "Investigate the assigned source" in system:
            src = first.content.split()[-1]
            if not seen:
                msg = AIMessage(
                    "", tool_calls=[_call("flaky_lookup", {"source": src}, src)]
                )
            elif seen == ["flaky_lookup"]:
                args = {"file_path": f"/findings/{src}.md", "content": last.content}
                msg = AIMessage("", tool_calls=[_call("write_file", args, src)])
            else:
                msg = AIMessage(f"done {src}")
        elif not seen and first.content != "list":
            calls = [
                _call(
                    "task",
                    {"description": f"investigate {s}", "subagent_type": "researcher"},
                    s,
                )
                for s in ("s1", "s2", "s3")
            ]
            msg = AIMessage("", tool_calls=calls)
        elif "ls" not in seen:
            msg = AIMessage("", tool_calls=[_call("ls", {"path": "/findings"}, "ls")])
        else:
            msg = AIMessage("report")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _research_agent(backend=None, store=None):
    researcher = {
        "name": "researcher",
        "description": "Investigates one source.",
        "system_prompt": "Investigate the assigned source using flaky_lookup.",
        "tools": [flaky_lookup],
    }
    extra = {"backend": backend} if backend is not None else {}
    return create_deep_agent(
        model=_ResearchModel(),
        tools=[flaky_lookup],
        system_prompt="You are a deep research agent.",
        subagents=[researcher],
        checkpointer=InMemorySaver(),
        store=store,
        **extra,
    )


def _last_ls(result: dict) -> str | None:
    listings = [
        m.content
        for m in result["messages"]
        if isinstance(m, ToolMessage) and m.name == "ls"
    ]
    return listings[-1] if listings else None


def test_parallel_task_calls_are_separate_sends_and_resume_reruns_only_the_failed_one():
    """Three `task` calls in one message are three tasks in one superstep
    (create_agent sends each tool call as its own Send). One sub-agent
    crashes: its siblings' results stay checkpointed, and a resume re-runs
    only the call that failed."""
    _lookups.clear()
    _fail_once.clear()
    _fail_once.add("s2")
    agent = _research_agent()
    config = {"configurable": {"thread_id": "t-send"}}

    with pytest.raises(RuntimeError, match="source crashed: s2"):
        agent.invoke({"messages": [{"role": "user", "content": "report"}]}, config)
    snapshot = agent.get_state(config)
    assert snapshot.next == ("tools",)
    assert sorted(t.error is not None for t in snapshot.tasks) == [False, False, True]
    assert _lookups == Counter({"s1": 1, "s2": 1, "s3": 1})

    result = agent.invoke(None, config)
    assert _lookups == Counter({"s1": 1, "s2": 2, "s3": 1})
    assert result["messages"][-1].content == "report"


def test_state_backend_files_stay_in_the_writing_threads_checkpoint():
    """Default backend: the files are still in thread A's checkpoint after
    the run; a new thread for the same customer cannot see them."""
    _lookups.clear()
    _fail_once.clear()
    agent = _research_agent()
    thread_a = {
        "configurable": {"thread_id": "research-9001", "customer_id": "cust-42"}
    }
    thread_b = {
        "configurable": {"thread_id": "research-9002", "customer_id": "cust-42"}
    }

    agent.invoke({"messages": [{"role": "user", "content": "report"}]}, thread_a)
    files = agent.get_state(thread_a).values["files"]
    assert sorted(files) == ["/findings/s1.md", "/findings/s2.md", "/findings/s3.md"]

    later = agent.invoke({"messages": [{"role": "user", "content": "list"}]}, thread_b)
    assert _last_ls(later) == "[]"


def test_store_backend_files_reach_a_new_thread_for_the_same_customer():
    _lookups.clear()
    _fail_once.clear()
    scoped_store = InMemoryStore()
    backend = StoreBackend(store=scoped_store, namespace=research_namespace)
    agent = _research_agent(backend=backend, store=scoped_store)
    thread_a = {
        "configurable": {"thread_id": "research-9001", "customer_id": "cust-42"}
    }
    thread_b = {
        "configurable": {"thread_id": "research-9002", "customer_id": "cust-42"}
    }
    other = {"configurable": {"thread_id": "research-9003", "customer_id": "cust-99"}}

    agent.invoke({"messages": [{"role": "user", "content": "report"}]}, thread_a)
    later = agent.invoke({"messages": [{"role": "user", "content": "list"}]}, thread_b)
    stranger = agent.invoke({"messages": [{"role": "user", "content": "list"}]}, other)

    assert "/findings/s1.md" in _last_ls(later)
    assert _last_ls(stranger) == "[]"
