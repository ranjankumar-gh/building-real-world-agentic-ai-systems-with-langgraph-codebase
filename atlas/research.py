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
"""

from operator import add
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command

from atlas.tools import text_of, web_search_tool


class ResearchState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    assignment: str  # the scoped handoff payload
    findings: Annotated[list[str], add]  # specialists accumulate results
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
