"""The Chapter 3 whiteboard, wired into a real, running StateGraph.

See Chapter 4, "StateGraph, Nodes, and Edges". This module supersedes
atlas/graph_sketch.py's role as Atlas's *executable* graph - graph_sketch.py
stays in the repo unchanged as the Chapter 3 node-SHAPE artifact (no
StateGraph, no compile()); this is the transcription of that whiteboard into
something `compile()` validates and the runtime actually runs.

Chapter 5 moved the state schema out to atlas/state.py and gave every channel
its own deliberately chosen reducer - `AtlasState` is imported from there now,
not defined inline.

Chapter 6, "Conditional Edges and Dynamic Control Flow", turns the linear
START -> triage -> retrieve -> answer -> END chain into the Figure 3.1
branching topology: `triage` validates the model's proposed route against
`ALLOWED_ROUTES` (off-menu -> "escalate", the routing boundary from that
chapter), `add_conditional_edges` replaces the fixed triage->retrieve and
retrieve->answer edges, and a new `escalate` node gives the graph a graceful
exit. `retrieve` is now a bounded retry cycle guarded by the explicit
`retrieve_attempts` state counter - not by LangGraph's `recursion_limit` -
and records a `KnowledgeBaseUnavailable` failure in state instead of crashing
or answering on top of it. `triage_with_command` is the chapter's `Command`
alternative: update state and route in one move. It is intentionally not
wired into `builder` below, the same way Chapter 4's `retrieve_async` is
defined but unused - the chapter's own guidance is to default to conditional
edges and reach for `Command` only when the update and the route are
genuinely one decision.

The node bodies originally called stubs in atlas.helpers
(classify/search_kb/compose_answer), which raised NotImplementedError.

Chapter 7, "Tools, Models, MCP, and create_agent", replaces `classify` with
the real, structured-output version in `atlas/triage.py` - `triage` below
now reads a validated `TriageResult` instead of parsing a raw string, but
still validates `.route` against `ALLOWED_ROUTES` (the routing boundary
holds; structured output narrows the input, it does not dissolve the
boundary). `search_kb` also went real in that chapter, as a `@tool` in
`atlas/tools.py` - along with `KnowledgeBaseUnavailable`, imported from
there now instead of atlas.helpers - but `retrieve` and `answer` below are
not yet rewired to call it: that integration (folding a tool-calling agent
into this graph) is deferred to a later chapter, so `retrieve`/`answer`
still call the atlas.helpers stubs for `search_kb`/`compose_answer` shape
continuity until then. See atlas/helpers.py's module docstring.
"""

import asyncio
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

from atlas.helpers import compose_answer, search_kb
from atlas.state import AtlasState
from atlas.tools import KnowledgeBaseUnavailable
from atlas.triage import classify

ALLOWED_ROUTES = ("answer", "retrieve", "escalate")
MAX_RETRIEVE_ATTEMPTS = 3


def triage(state: AtlasState) -> dict:
    decision = classify(state["messages"])  # typed; route is already constrained
    route = decision.route if decision.route in ALLOWED_ROUTES else "escalate"
    return {"route": route}


def triage_with_command(
    state: AtlasState,
) -> Command[Literal["answer", "retrieve", "escalate"]]:
    """The `Command` alternative to `triage` + `route_from_triage`: update
    state and name the next node in one move, for when the two are genuinely
    the same decision. Not wired into `builder` - `triage` above stays the
    default, since it keeps routing visible on a separate edge."""
    decision = classify(state["messages"])
    route = decision.route if decision.route in ALLOWED_ROUTES else "escalate"
    return Command(update={"route": route}, goto=route)


def route_from_triage(state: AtlasState) -> Literal["answer", "retrieve", "escalate"]:
    return state["route"]


def retrieve(state: AtlasState) -> dict:
    """Plain `def`, on purpose. `retrieve` does network I/O (it hits the
    knowledge base); under `ainvoke` the runtime offloads a sync node like
    this one to a worker thread, so a blocking call inside it never touches
    the event loop. See "Making it correct under load" - marking this
    `async def` while still calling `search_kb` directly is the mistake
    behind the chapter's opening incident (every concurrent run stalls
    behind one blocking request).

    Chapter 6 adds the bounded-retry counter and the failure path: a
    `KnowledgeBaseUnavailable` is caught and recorded in `error` rather than
    left to crash the run, and every successful call increments
    `retrieve_attempts` so `route_after_retrieve` can cap the retry loop."""
    try:
        hits = search_kb(state["messages"])
    except KnowledgeBaseUnavailable as exc:
        return {"error": str(exc)}  # record the failure - do not pretend it worked
    attempts = state.get("retrieve_attempts", 0) + 1
    return {"retrieved": hits, "retrieve_attempts": attempts}


async def retrieve_async(state: AtlasState) -> dict:
    """The correct shape if `retrieve` genuinely needed `async` (say, to
    `await` several I/O calls): offload the blocking call explicitly with
    `asyncio.to_thread` instead of calling it directly on the event loop.
    Not wired into `builder` below - Atlas's `retrieve` has no other await
    work yet, so the plain `def` version above is the right default."""
    hits = await asyncio.to_thread(search_kb, state["messages"])
    return {"retrieved": hits}


def route_after_retrieve(
    state: AtlasState,
) -> Literal["answer", "retrieve", "escalate"]:
    if state.get("error"):
        return "escalate"  # tool failure -> human, not a fake answer
    if state["retrieved"]:
        return "answer"  # got results -> answer
    if state["retrieve_attempts"] >= MAX_RETRIEVE_ATTEMPTS:
        return "escalate"  # gave up -> human, gracefully
    return "retrieve"  # bounded retry


def answer(state: AtlasState) -> dict:
    reply = compose_answer(state["messages"], state["retrieved"])
    return {"messages": [reply]}


def escalate(state: AtlasState) -> dict:
    """The graceful exit the retry loop and the model's off-menu routes both
    fall back to. Same shape as the Chapter 3 whiteboard's escalate node
    (atlas/graph_sketch.py) - Chapter 6 is what finally wires it in."""
    return {"ticket": {"status": "escalated"}}


builder = StateGraph(AtlasState)
builder.add_node("triage", triage)
builder.add_node(
    "retrieve",
    retrieve,
    # First look at durable execution (full treatment: Chapter 10). Safe here
    # because retrieve is a read; do not add a retry_policy to a
    # side-effecting node without the discipline Chapter 10 covers.
    retry_policy=RetryPolicy(max_attempts=3, retry_on=(ConnectionError,)),
)
builder.add_node("answer", answer)
builder.add_node("escalate", escalate)

builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", route_from_triage)  # the branch
builder.add_conditional_edges("retrieve", route_after_retrieve)  # the cycle + its exit
builder.add_edge("answer", END)
builder.add_edge("escalate", END)

graph = builder.compile()
