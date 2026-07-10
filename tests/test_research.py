"""Chapter 16, "The Supervisor Pattern (and Swarm as Contrast)" -
atlas/research.py.

See "Building the supervisor by hand". Building `create_agent` (and the
handoff tools) does not require a live API key - only invoking the model
does - matching the no-live-call convention already used for
`atlas/agent.py` (see tests/test_agent.py). What's actually invoked and
checked here is the handoff `Command` shape - the chapter's central claim is
that the payload it carries is a SCOPED assignment, not the transcript - and
`web_research`, with its own scoped `create_agent` call mocked so no model
runs.

Chapter 17, "Subgraphs, Parallelism, and Map-Reduce", adds the Send-based
map-reduce tests below: `search_source`/`SourceUnavailable` is a seeded,
mockable backend (no live call, no mocking needed), so `fan_out`,
`research_worker`, and the compiled `research_graph` are all exercised for
real, including the end-to-end fan-out/reduce/partial-failure behavior.

Chapter 20, "Observability and Debugging with LangSmith", adds `name=` to
`supervisor` and to the scoped `create_agent` each of `web_research`/
`doc_research` builds - see "Naming the fleet: attribution across the
supervisor topology" - and finally supplies `doc_research` as code (Chapter
16 deferred it to prose only). `doc_research`'s own scoped agent is faked
the same way `web_research`'s already is, so no live model call happens.

Chapter 24, "Patterns from Production", adds the memory-horizon retrofit:
`research_ns`/`recall_finding`/`remember_finding`, Chapter 13's
`profile_ns`/`compact`/`reflect` pattern repointed at research findings. No
live call and no external service - `store.put`/`store.get` against a real
`InMemoryStore`, same convention as tests/test_memory.py."""

from datetime import datetime, timedelta, timezone

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, Send

from atlas import research as research_module
from atlas.research import (
    DEFAULT_TTL_DAYS,
    MAX_HANDOFFS,
    SourceUnavailable,
    doc_research,
    fan_out,
    make_handoff,
    recall_finding,
    remember_finding,
    research_graph,
    research_ns,
    research_worker,
    route_from_specialist,
    search_source,
    supervisor,
    web_research,
)


def _tool_runtime(state: dict, tool_call_id: str = "call_1") -> ToolRuntime:
    """A real `ToolRuntime`, built directly rather than through a live tool
    node - the same shape the tool-calling runtime would inject, without
    running a graph."""
    return ToolRuntime(
        state=state,
        tool_call_id=tool_call_id,
        context=None,
        config={},
        stream_writer=lambda _: None,
        store=None,
    )


def test_make_handoff_names_the_tool_for_its_specialist_not_the_wrapper_function():
    """Every call to `make_handoff` defines a function literally named
    `handoff` - without an explicit per-specialist tool name, two specialists'
    handoff tools would collide under that one name (verified: `create_agent`
    silently keeps only the last-registered one, no error at construction).
    The explicit `delegate_to_{specialist}` name is the fix."""
    handoff = make_handoff("web_research", "Delegate a web-search sub-task.")

    assert handoff.name == "delegate_to_web_research"
    assert handoff.description == "Delegate a web-search sub-task."


def test_two_handoffs_for_different_specialists_do_not_collide_on_tool_name():
    web = make_handoff("web_research", "Delegate a web-search sub-task.")
    doc = make_handoff("doc_research", "Delegate an internal-docs sub-task.")

    assert web.name != doc.name


def test_handoff_command_scopes_the_payload_to_the_assignment_not_the_transcript():
    """The chapter's central fix: the payload is `assignment=task`, a single
    scoped sub-task, never `state["messages"]` (the prebuilt library's
    default, and the hook's tripled-bill bug)."""
    handoff = make_handoff("web_research", "Delegate a web-search sub-task.")
    runtime = _tool_runtime(state={"handoffs": 0})

    result = handoff.func(task="Find the refund policy.", runtime=runtime)

    assert isinstance(result, Command)
    assert result.goto == "web_research"
    assert result.update["assignment"] == "Find the refund policy."
    assert list(result.update.keys()) == ["assignment", "messages", "handoffs"]


def test_handoff_command_routes_in_the_parent_graph():
    handoff = make_handoff("doc_research", "Delegate an internal-docs sub-task.")
    runtime = _tool_runtime(state={"handoffs": 0})

    result = handoff.func(task="Look up the SLA.", runtime=runtime)

    assert result.graph == Command.PARENT


def test_handoff_command_increments_the_explicit_bound():
    handoff = make_handoff("web_research", "Delegate a web-search sub-task.")
    runtime = _tool_runtime(state={"handoffs": 3})

    result = handoff.func(task="Find the refund policy.", runtime=runtime)

    assert result.update["handoffs"] == 4


def test_handoff_command_acknowledges_with_a_tool_message_matching_the_call_id():
    """The ack keeps the coordinator's own message history valid - every
    tool call needs a matching tool result."""
    handoff = make_handoff("web_research", "Delegate a web-search sub-task.")
    runtime = _tool_runtime(state={"handoffs": 0}, tool_call_id="call_42")

    result = handoff.func(task="Find the refund policy.", runtime=runtime)
    [ack] = result.update["messages"]

    assert isinstance(ack, ToolMessage)
    assert ack.tool_call_id == "call_42"
    assert "web_research" in ack.content


def test_supervisor_compiles_to_an_invokable_graph_without_calling_the_model():
    assert hasattr(supervisor, "invoke")


def test_supervisor_carries_both_distinct_handoff_tools():
    """The coordinator never researches itself (see the system prompt) - its
    tools node holds exactly the two handoffs `make_handoff` built, each
    under its own distinct name, not a research tool like `web_search_tool`."""
    tool_names = set(supervisor.nodes["tools"].bound.tools_by_name)

    assert tool_names == {"delegate_to_web_research", "delegate_to_doc_research"}


def test_web_research_reads_only_the_scoped_assignment_not_the_full_history(monkeypatch):
    """The specialist's entire input is `state["assignment"]` - a scoped
    sub-task - never the shared conversation. The scoped agent's own
    `.invoke` is faked so no model call happens."""

    captured_input = {}

    class _FakeAgent:
        def invoke(self, input_):
            captured_input.update(input_)
            return {"messages": [AIMessage("Refunds: 30-day window.")]}

    monkeypatch.setattr(research_module, "create_agent", lambda **_: _FakeAgent())

    state = {
        "messages": [{"role": "user", "content": "irrelevant prior turns"}],
        "assignment": "Find the refund policy.",
        "findings": [],
        "handoffs": 1,
    }

    result = web_research(state)

    assert captured_input["messages"] == [
        {"role": "user", "content": "Find the refund policy."}
    ]
    assert result == {"findings": ["Refunds: 30-day window."]}


def test_web_research_names_its_scoped_agent_for_the_trace_tree(monkeypatch):
    """Without name="web-research", the specialist would still trace
    correctly - just under create_agent's generic default, indistinguishable
    from doc_research calling itself twice."""
    captured_kwargs = {}

    class _FakeAgent:
        def invoke(self, input_):
            return {"messages": [AIMessage("ok")]}

    def _fake_create_agent(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeAgent()

    monkeypatch.setattr(research_module, "create_agent", _fake_create_agent)

    web_research({"assignment": "task", "findings": [], "handoffs": 0})

    assert captured_kwargs["name"] == "web-research"


def test_doc_research_reads_only_the_scoped_assignment_not_the_full_history(
    monkeypatch,
):
    """Chapter 20 finally supplies doc_research as code - identical shape to
    web_research, against search_kb instead of web_search_tool."""
    captured_input = {}

    class _FakeAgent:
        def invoke(self, input_):
            captured_input.update(input_)
            return {"messages": [AIMessage("Refunds: 30-day window.")]}

    monkeypatch.setattr(research_module, "create_agent", lambda **_: _FakeAgent())

    state = {
        "messages": [{"role": "user", "content": "irrelevant prior turns"}],
        "assignment": "Find the refund policy.",
        "findings": [],
        "handoffs": 1,
    }

    result = doc_research(state)

    assert captured_input["messages"] == [
        {"role": "user", "content": "Find the refund policy."}
    ]
    assert result == {"findings": ["Refunds: 30-day window."]}


def test_doc_research_names_its_scoped_agent_for_the_trace_tree(monkeypatch):
    captured_kwargs = {}

    class _FakeAgent:
        def invoke(self, input_):
            return {"messages": [AIMessage("ok")]}

    def _fake_create_agent(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeAgent()

    monkeypatch.setattr(research_module, "create_agent", _fake_create_agent)

    doc_research({"assignment": "task", "findings": [], "handoffs": 0})

    assert captured_kwargs["name"] == "doc-research"


def test_supervisor_is_named_for_the_trace_tree():
    """`name="supervisor"` turns the coordinator's generic AgentExecutor
    span into a labeled one - the same fix web_research/doc_research get."""
    assert supervisor.name == "supervisor"


def test_route_from_specialist_returns_to_the_supervisor_below_the_bound():
    assert route_from_specialist({"handoffs": MAX_HANDOFFS - 1}) == "supervisor"


def test_route_from_specialist_degrades_to_compile_at_the_bound():
    """The graceful exit: hitting the explicit bound compiles partial
    findings instead of looping forever or relying on the recursion limit."""
    assert route_from_specialist({"handoffs": MAX_HANDOFFS}) == "compile"


# --- Chapter 17: Send-based map-reduce -------------------------------------


def test_search_source_returns_the_seeded_result_for_a_known_source():
    assert "30 days" in search_source("docs.internal/refund-policy")


def test_search_source_raises_source_unavailable_for_an_unknown_source():
    with pytest.raises(SourceUnavailable):
        search_source("nope/does-not-exist")


def test_fan_out_returns_one_send_per_source_scoped_to_that_source():
    """The map: a routing function returning list[Send] instead of a node
    name - each Send names research_worker and carries exactly one
    source, not the whole list."""
    sends = fan_out({"sources": ["a", "b", "c"]})

    assert all(isinstance(s, Send) for s in sends)
    assert [s.node for s in sends] == ["research_worker", "research_worker", "research_worker"]
    assert [s.arg["source"] for s in sends] == ["a", "b", "c"]


def test_research_worker_returns_a_result_finding_for_a_reachable_source():
    delta = research_worker({"source": "docs.internal/sla"})

    assert delta == {
        "findings": [
            {
                "source": "docs.internal/sla",
                "result": "Enterprise SLA guarantees a 4-hour first response.",
            }
        ]
    }


def test_research_worker_returns_an_error_finding_instead_of_raising():
    """"Handle partial failure in the worker, not around it": a dead source
    becomes a finding WITH an error key, never an unhandled exception -
    so one bad source cannot fail the whole superstep."""
    delta = research_worker({"source": "nope/does-not-exist"})

    assert delta == {
        "findings": [
            {"source": "nope/does-not-exist", "error": "source unreachable: nope/does-not-exist"}
        ]
    }


def test_research_graph_fans_out_and_reduces_findings_for_every_source():
    """End-to-end through the compiled subgraph: the barrier waits for every
    fanned-out worker, and the `add` reducer on `findings` merges all of
    their writes - no gather, no lock, no manual join."""
    result = research_graph.invoke(
        {"sources": ["docs.internal/refund-policy", "docs.internal/sla"]}
    )

    sources_seen = {f["source"] for f in result["findings"]}
    assert sources_seen == {"docs.internal/refund-policy", "docs.internal/sla"}
    assert all("result" in f for f in result["findings"])


def test_research_graph_survives_one_dead_source_among_several():
    """A mixed batch - two reachable sources and one dead one - completes
    with three findings, not a crash: the reduce step sees the error and the
    graph never raises."""
    result = research_graph.invoke(
        {
            "sources": [
                "docs.internal/refund-policy",
                "nope/does-not-exist",
                "docs.internal/sla",
            ]
        }
    )

    findings_by_source = {f["source"]: f for f in result["findings"]}
    assert len(findings_by_source) == 3
    assert "error" in findings_by_source["nope/does-not-exist"]
    assert "result" in findings_by_source["docs.internal/refund-policy"]


def test_research_graph_honors_max_concurrency_in_the_invoke_config():
    """"Bound the fan-out": max_concurrency is accepted on invoke's config
    and the run still completes correctly with it set."""
    result = research_graph.invoke(
        {"sources": ["docs.internal/refund-policy", "docs.internal/sla"]},
        config={"max_concurrency": 1},
    )

    assert len(result["findings"]) == 2


# --- Chapter 24: the memory horizon this extension was missing -------------


def test_research_ns_scopes_by_customer_id():
    """Same privacy-boundary shape as atlas/memory.py's profile_ns, pointed
    at a "research" namespace instead of "profile"."""
    assert research_ns("cust-1") == ("customer", "cust-1", "research")
    assert research_ns("cust-1") != research_ns("cust-2")


def test_recall_finding_returns_none_on_a_genuine_miss():
    store = InMemoryStore()

    assert recall_finding(store, "cust-1", "return policy?") is None


def test_remember_finding_then_recall_finding_round_trips():
    """The write/read pair: a fresh remember_finding is recallable
    immediately - no TTL has elapsed yet."""
    store = InMemoryStore()

    remember_finding(store, "cust-1", "return policy?", ["30-day window."])

    assert recall_finding(store, "cust-1", "return policy?") == ["30-day window."]


def test_recall_finding_is_scoped_per_customer():
    store = InMemoryStore()
    remember_finding(store, "cust-1", "return policy?", ["30-day window."])

    assert recall_finding(store, "cust-2", "return policy?") is None


def test_recall_finding_treats_a_stale_hit_as_a_miss():
    """A hit older than DEFAULT_TTL_DAYS is treated exactly like a miss -
    both mean re-derive, per the chapter's recall_finding docstring."""
    store = InMemoryStore()
    stale = datetime.now(timezone.utc) - timedelta(days=DEFAULT_TTL_DAYS + 1)
    store.put(
        research_ns("cust-1"),
        "return policy?",
        {"findings": ["30-day window."], "recorded_at": stale.isoformat()},
    )

    assert recall_finding(store, "cust-1", "return policy?") is None


def test_recall_finding_returns_a_hit_just_inside_the_ttl_window():
    store = InMemoryStore()
    fresh = datetime.now(timezone.utc) - timedelta(days=DEFAULT_TTL_DAYS - 1)
    store.put(
        research_ns("cust-1"),
        "return policy?",
        {"findings": ["30-day window."], "recorded_at": fresh.isoformat()},
    )

    assert recall_finding(store, "cust-1", "return policy?") == ["30-day window."]
