"""Atlas as a whiteboard state machine - node SHAPES, not yet wired into a graph.

See Chapter 3, "Thinking in Graphs" - "Whiteboarding Atlas". Every node here obeys
the state-transition discipline: read state, return a delta, decide nothing about
what runs next. Routing is a separate, pure function that lives on the edge, not
inside a node body. Chapter 4 wires this into a real, running StateGraph
(atlas/graph.py); Chapter 6 turns route_from_triage into a conditional edge and
adds the escalate branch and the bounded retrieve retry.

The helpers imported below (classify, search_kb, compose_answer) were stubs in
atlas/helpers.py - importing them here let these node shapes read as real
functions instead of pseudocode, even though calling them raised
NotImplementedError.

Chapter 7, "Tools, Models, MCP, and create_agent", fills `classify` in for
real (a validated TriageResult, in atlas/triage.py) - imported from there
now. `search_kb` and `compose_answer` are the adapters in atlas/helpers.py
onto that same chapter's knowledge-base tool; this whiteboard sketch calls
whatever that module currently provides, same as it always has.
"""

from typing import Literal, TypedDict

from atlas.helpers import compose_answer, search_kb
from atlas.triage import classify


class AtlasState(TypedDict):
    """First sketch of the shared state. The full typed schema with reducers is
    built in Chapters 4-5; here we only need its shape to reason about nodes."""

    messages: list       # the conversation so far
    ticket: dict | None  # the active support ticket
    retrieved: list      # knowledge-base hits (the retrieval scratchpad)
    route: str           # the triage decision


def triage(state: AtlasState) -> dict:
    """Read the conversation, decide the route. Returns a delta only."""
    decision = classify(state["messages"])   # "answer" | "retrieve" | "escalate"
    return {"route": decision}


def retrieve(state: AtlasState) -> dict:
    return {"retrieved": search_kb(state["messages"])}


def answer(state: AtlasState) -> dict:
    reply = compose_answer(state["messages"], state["retrieved"])
    return {"messages": [reply]}


def escalate(state: AtlasState) -> dict:
    return {"ticket": {"status": "escalated"}}


def route_from_triage(
    state: AtlasState,
) -> Literal["answer", "retrieve", "escalate"]:
    """Control flow on the edge: read state, return the next node's name.
    It decides nothing the node bodies should decide. Wired in Chapter 6."""
    return state["route"]
