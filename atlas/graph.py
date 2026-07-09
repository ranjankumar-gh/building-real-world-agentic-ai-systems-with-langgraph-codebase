"""The Chapter 3 whiteboard, wired into a real, running StateGraph.

See Chapter 4, "StateGraph, Nodes, and Edges". This module supersedes
atlas/graph_sketch.py's role as Atlas's *executable* graph - graph_sketch.py
stays in the repo unchanged as the Chapter 3 node-SHAPE artifact (no
StateGraph, no compile()); this is the transcription of that whiteboard into
something `compile()` validates and the runtime actually runs.

The topology here is deliberately linear: START -> triage -> retrieve ->
answer -> END. Chapter 3's triage branch and escalate path exist as
`route_after_triage` in graph_sketch.py but are not wired yet - Chapter 6
turns the straight triage->retrieve edge into a conditional one. Chapter 5
moved the state schema out to atlas/state.py and gave every channel its own
deliberately chosen reducer - `AtlasState` is imported from there now, not
defined inline.

The node bodies call the same stubs as graph_sketch.py (atlas.helpers:
classify/search_kb/compose_answer), which raise NotImplementedError until
Chapter 7 fills them in for real.
"""

import asyncio

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from atlas.helpers import classify, compose_answer, search_kb
from atlas.state import AtlasState


def triage(state: AtlasState) -> dict:
    return {"route": classify(state["messages"])}


def retrieve(state: AtlasState) -> dict:
    """Plain `def`, on purpose. `retrieve` does network I/O (it hits the
    knowledge base); under `ainvoke` the runtime offloads a sync node like
    this one to a worker thread, so a blocking call inside it never touches
    the event loop. See "Making it correct under load" - marking this
    `async def` while still calling `search_kb` directly is the mistake
    behind the chapter's opening incident (every concurrent run stalls
    behind one blocking request)."""
    return {"retrieved": search_kb(state["messages"])}


async def retrieve_async(state: AtlasState) -> dict:
    """The correct shape if `retrieve` genuinely needed `async` (say, to
    `await` several I/O calls): offload the blocking call explicitly with
    `asyncio.to_thread` instead of calling it directly on the event loop.
    Not wired into `builder` below - Atlas's `retrieve` has no other await
    work yet, so the plain `def` version above is the right default."""
    hits = await asyncio.to_thread(search_kb, state["messages"])
    return {"retrieved": hits}


def answer(state: AtlasState) -> dict:
    reply = compose_answer(state["messages"], state["retrieved"])
    return {"messages": [reply]}


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

builder.add_edge(START, "triage")
builder.add_edge("triage", "retrieve")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)

graph = builder.compile()
