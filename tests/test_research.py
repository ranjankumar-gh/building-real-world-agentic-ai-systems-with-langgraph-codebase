"""Chapter 16, "The Supervisor Pattern (and Swarm as Contrast)" -
atlas/research.py.

See "Building the supervisor by hand". Building `create_agent` (and the
handoff tools) does not require a live API key - only invoking the model
does - matching the no-live-call convention already used for
`atlas/agent.py` (see tests/test_agent.py). What's actually invoked and
checked here is the handoff `Command` shape - the chapter's central claim is
that the payload it carries is a SCOPED assignment, not the transcript - and
`web_research`, with its own scoped `create_agent` call mocked so no model
runs."""

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.types import Command

from atlas import research as research_module
from atlas.research import (
    MAX_HANDOFFS,
    make_handoff,
    route_from_specialist,
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


def test_route_from_specialist_returns_to_the_supervisor_below_the_bound():
    assert route_from_specialist({"handoffs": MAX_HANDOFFS - 1}) == "supervisor"


def test_route_from_specialist_degrades_to_compile_at_the_bound():
    """The graceful exit: hitting the explicit bound compiles partial
    findings instead of looping forever or relying on the recursion limit."""
    assert route_from_specialist({"handoffs": MAX_HANDOFFS}) == "compile"
