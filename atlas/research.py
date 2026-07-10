"""Chapter 16, "The Supervisor Pattern (and Swarm as Contrast)" - Atlas's
research extension, built as a hand-rolled tool-calling supervisor.

See "Building the supervisor by hand". Chapter 15 gave the split decision
its blessing (`atlas/multiagent_decision.py`): Atlas's research workload is
parallel-independent, so it earns a multi-agent split. This module builds
that split the way the chapter argues it should be built - by hand, from
`create_agent` and `Command`-based handoff tools - specifically so the
*handoff payload* is owned rather than defaulted away by a prebuilt library.

`ResearchState` carries the shared conversation plus two extras: `assignment`,
the scoped sub-task a specialist reads instead of the full transcript, and
`findings`, the accumulator specialists append into. `handoffs` is the
explicit bound from "Bounding the handoffs" - the Chapter 6 cycle-guard
lesson, now spanning agents.

`make_handoff` is a factory, not a single tool: each specialist gets its own
handoff tool, built with a specialist-specific routing description (see the
"Vague specialist descriptions produce random routing" warning - the
descriptions are the supervisor's real routing interface). Every handoff tool
it builds returns a `Command` that (a) routes to the named specialist via
`goto=specialist`, (b) carries the scoped `assignment` - not `state["messages"]`
- as the payload, and (c) uses `graph=Command.PARENT` because the specialist
nodes live in the parent graph, not inside the tool-calling coordinator's own
subgraph. The `ToolMessage` acknowledgment keeps the coordinator's own message
history valid (every tool call needs a matching tool result).

The tool is named explicitly - `@tool(f"delegate_to_{specialist}", ...)` -
rather than left to default to the wrapper function's own name. Every call to
`make_handoff` defines a function literally named `handoff`, so without an
explicit name every specialist's handoff tool would register under the same
name; `create_agent`'s tool node keeps only the last one added
(`tools_by_name` is a dict keyed by name), and the model would see a single
delegation tool that always routed to whichever specialist happened to be
registered last - silently, with no error at construction. Verified against
the pinned `langchain==1.3.0` build: two `@tool(description=...)`-only
handoffs collide exactly this way; the explicit per-specialist name fixes it.

`supervisor` is a `create_agent` whose only tools are handoffs - it never
researches itself, only routes. `web_research` is the specialist node the
chapter writes out in full: it reads `state["assignment"]` as the *entire*
input to its own scoped `create_agent`, not the shared history, which is
payload control made concrete. `atlas/tools.py`'s `web_search_tool` (added in
this same chapter, alongside the Chapter 7 `search_kb`) is its tool.

The chapter's own code stops at `web_research`; a `doc_research` specialist
and a `compile` node are named in prose ("Wire it with StateGraph...") but
not given as code, so they are not invented here - see the chapter's "What's
Next" for why the full parallel wiring (subgraphs, `Send`-based map-reduce)
is deferred to Chapter 17, where the supervisor's one-at-a-time delegation
becomes genuine concurrency.

`route_from_specialist` is the routing function for the bound: once
`handoffs` reaches `MAX_HANDOFFS`, degrade gracefully to a `"compile"` node
rather than let the run loop forever or hit LangGraph's `recursion_limit` -
the chapter's point that a multi-agent recursion crash takes down the
coordinator and every specialist with it, worse than a single-agent runaway.

Chapter 17, "Subgraphs, Parallelism, and Map-Reduce", is where the "full
parallel wiring" this docstring deferred above finally lands - but not as a
literal refactor of `web_research`/`doc_research` into fanned-out workers.
The chapter's own code ("Building the map-reduce") is a fresh, self-contained
illustration of the `Send` + reducer shape against a generic `sources` list,
so that is what is built here too: `search_source`/`SourceUnavailable` (a
seeded, mockable backend, same convention as `atlas/tools.py`'s `_KB`/`_WEB`),
`fan_out`, `research_worker`, and a compiled `research_graph` subgraph
(`plan` -> N parallel `research_worker`s -> `END`) built from exactly the
fragments the chapter shows. `ResearchState` is *extended* rather than
duplicated under a colliding second definition: `sources` is new, and
`findings` widens from `list[str]` (a specialist's prose finding) to
`list[dict]` (a worker's structured `{"source", "result"}` or
`{"source", "error"}` finding) - both still merge through the same `add`
reducer, since neither TypedDict field nor `add` enforce element type at
runtime. `atlas/graph.py` mounts `research_graph` as a wrapped node (Chapter
18's `atlas/deep_research.py` and Chapter 19's streaming example both import
`search_source`/`SourceUnavailable` from here directly, so those two names -
unlike the still-uncoded `doc_research`/`compile` - are load-bearing beyond
this module and are not optional).
"""

from operator import add
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, Send

from atlas.tools import text_of, web_search_tool


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    assignment: str  # the scoped handoff payload
    sources: list[str]  # NEW (Chapter 17): the map-reduce fan-out list
    findings: Annotated[list[dict], add]  # widened: workers now write dicts
    handoffs: int  # explicit bound (Chapter 6)


def make_handoff(specialist: str, description: str):
    """Build a handoff tool that routes to a specialist with a SCOPED payload."""

    @tool(f"delegate_to_{specialist}", description=description)
    def handoff(task: str, runtime: ToolRuntime) -> Command:
        ack = ToolMessage(
            f"Delegated to {specialist}.", tool_call_id=runtime.tool_call_id
        )
        return Command(
            goto=specialist,
            update={
                "assignment": task,  # the scoped sub-task, not the transcript
                "messages": [ack],
                "handoffs": runtime.state["handoffs"] + 1,
            },
            graph=Command.PARENT,  # routes in the parent graph, where specialists live
        )

    return handoff


supervisor = create_agent(
    model="claude-sonnet-4-6",
    tools=[
        make_handoff("web_research", "Delegate a web-search sub-task."),
        make_handoff("doc_research", "Delegate an internal-docs sub-task."),
    ],
    system_prompt=(
        "You coordinate research specialists. Break the request into "
        "sub-tasks and delegate each with a precise, self-contained task "
        "description. Do not research yourself."
    ),
)


def web_research(state: ResearchState) -> dict:
    """A specialist node: reads its SCOPED assignment, not the transcript."""
    agent = create_agent(model="claude-sonnet-4-6", tools=[web_search_tool])
    result = agent.invoke(
        {"messages": [{"role": "user", "content": state["assignment"]}]}
    )
    return {"findings": [text_of(result["messages"][-1])]}


MAX_HANDOFFS = 6


def route_from_specialist(state: ResearchState) -> str:
    """Specialists return to the supervisor - unless the bound is hit."""
    if state["handoffs"] >= MAX_HANDOFFS:
        return "compile"  # degrade gracefully: compile what we have
    return "supervisor"


# --- Chapter 17: Send-based map-reduce -------------------------------------
#
# "The supervisor from Chapter 16 is correct and slow" - one specialist at a
# time via make_handoff's Command routing. This section is the parallel
# alternative: a routing function returns a list[Send] instead of a node
# name, LangGraph fans the workers out into one superstep (the barrier), and
# the findings reducer above does the fan-in. See "Building the map-reduce".

# Seeded, mockable backend - same convention as atlas/tools.py's _KB/_WEB: an
# in-repo dict standing in for whatever real source lookup research_worker
# would call, so the fan-out runs fully offline and deterministically in
# tests. A source not in the dict is treated as unreachable.
_SOURCES: dict[str, str] = {
    "docs.internal/refund-policy": "Refunds are honored within 30 days of purchase.",
    "docs.internal/sla": "Enterprise SLA guarantees a 4-hour first response.",
    "web/langgraph-overview": "LangGraph is a low-level orchestration runtime.",
}


class SourceUnavailable(RuntimeError):
    """Raised when a research source cannot be reached."""


def search_source(source: str) -> str:
    """Look up one research source. Raises `SourceUnavailable` for any
    source not seeded above - the failure `research_worker` below is built
    to survive without failing the whole fan-out."""
    if source not in _SOURCES:
        raise SourceUnavailable(f"source unreachable: {source}")
    return _SOURCES[source]


def fan_out(state: ResearchState) -> list[Send]:
    """Map: one worker per source, each with a scoped payload."""
    return [Send("research_worker", {"source": src}) for src in state["sources"]]


def research_worker(state: dict) -> dict:
    """Reduce-side input: one worker, one source, returns one finding."""
    src = state["source"]
    try:
        result = search_source(src)
        return {"findings": [{"source": src, "result": result}]}
    except SourceUnavailable as exc:
        # Partial-failure handling lives here: a dead source returns a
        # finding WITH an error, not an exception, so one bad source cannot
        # fail the superstep - the reduce step downstream sees the error and
        # decides what to do with it.
        return {"findings": [{"source": src, "error": str(exc)}]}


def plan(state: ResearchState) -> dict:
    """Entry node for the map-reduce subgraph: `sources` already arrives as
    input (see `atlas/graph.py`'s `derive_sources` adapter), so `plan` has
    nothing to add yet - it exists to give `fan_out` a named node to hang
    `add_conditional_edges` off of, exactly as the chapter's own
    `builder.add_conditional_edges("plan", fan_out)` shows. A real planner
    that decomposes a request into sources would live here."""
    return {}


research_builder = StateGraph(ResearchState)
research_builder.add_node("plan", plan)
research_builder.add_node("research_worker", research_worker)
research_builder.add_edge(START, "plan")
research_builder.add_conditional_edges("plan", fan_out)  # plan -> N parallel workers
research_builder.add_edge("research_worker", END)

research_graph = research_builder.compile()  # the map-reduce pipeline
