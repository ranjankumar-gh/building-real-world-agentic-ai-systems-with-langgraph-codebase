"""Chapter 17, "Mounting the resolve agent": the adapter that lets Atlas's
topology call a compiled create_agent graph as one node.

The agent speaks a messages-centric state; AtlasState also carries
`retrieved`, `ticket`, `route` and the Chapter 6 loop guard. There are no
shared keys beyond `messages`, so the compiled agent cannot be added with
`add_node` directly - it needs the same input/output mapping Chapter 17's
`research` node already uses for the map-reduce subgraph.

Verified against the pinned build (langgraph==1.2.6, langchain==1.3.0):
`create_agent` returns a `CompiledStateGraph` typed over `AgentState`, whose
`messages` channel merges with `add_messages` - the returned list is always
the input messages followed by what the agent added, so slicing by the
input length below is safe, not an assumption.

Kept out of atlas/graph.py on purpose: importing the agent constructs a
chat model, and the default graph is model-free so the offline suite and a
reader with no API key both still get a working answer path.

`build_resolved_graph`, below, is the production assembly: the one call a
reader runs to get Atlas's topology with the full `RESOLVE_MIDDLEWARE` stack
mounted in the answering position, via `make_resolve_node`. Its import of
`atlas.agent` is inside the function, not at module level, for the same
reason `atlas/graph.py`'s `answer` stays the default node - constructing
`resolve_agent` builds a chat model, and hoisting that import to the top of
this module would make every importer of `atlas.resolve` (including the
offline tests above) pay that cost too."""

from collections.abc import Callable

from langchain_core.messages import AnyMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel

from atlas.graph import build_graph
from atlas.state import AtlasState


def make_resolve_node(agent: CompiledStateGraph) -> Callable[[AtlasState], dict]:
    """Wrap a compiled agent graph as an Atlas node.

    Returns only the messages the agent added. Returning the agent's whole
    message list would re-send the conversation through `add_messages` on
    every pass, and returning its state wholesale would re-append every
    retrieved document through `dedup_by_id`."""

    def resolve(state: AtlasState) -> dict:
        before = len(state["messages"])
        out = agent.invoke({"messages": state["messages"]})
        new: list[AnyMessage] = out["messages"][before:]
        return {"messages": new}

    return resolve


def build_resolved_graph() -> Pregel:
    """Atlas's topology with the full middleware stack mounted in the
    answering position. This is the production assembly: the seam exists so
    the default graph can stay model-free, and this is the call that fills
    it. Chapter 17, "Mounting the resolve agent"."""
    # Imported here, not at module level - see the module docstring. Every
    # offline test that only needs make_resolve_node must not pay the cost
    # of atlas.agent constructing a chat model at import time.
    from atlas.agent import resolve_agent

    return build_graph(resolve_node=make_resolve_node(resolve_agent))
