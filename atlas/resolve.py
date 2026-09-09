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
reader with no API key both still get a working answer path."""

from collections.abc import Callable

from langchain_core.messages import AnyMessage
from langgraph.graph.state import CompiledStateGraph

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
