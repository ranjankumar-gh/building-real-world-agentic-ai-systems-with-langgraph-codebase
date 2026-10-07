"""Chapter 22, "Deployment and Scaling" - the graphs `langgraph.json` serves.

See "Packaging Atlas for the Agent Server". Two rules shape this module.

The server serves the production assembly. `atlas/graph.py`'s module-level
`graph` is the model-free default (its `answer` is a deterministic stub), so
serving it would run none of the resolve agent's middleware. `resolve` below
mounts the same agent `atlas/resolve.py`'s `build_resolved_graph` does, with
the whole `RESOLVE_MIDDLEWARE` stack, Chapter 23's gates included.

The server owns persistence. The Agent Server supplies the checkpointer and
the store (Postgres under `langgraph up`, in memory under `langgraph dev`),
and `langgraph dev` refuses to load a graph compiled with its own. So both
graphs here compile with neither. `build_resolved_graph()` stays the
in-process entry, compiled with `InMemorySaver`/`InMemoryStore`, for tests
and local scripts.

One more difference from the in-process entry: the served resolve node
builds the `AtlasContext` Chapter 23's gates read from the identity the
server proved (`runtime.server_info.user`, filled by `atlas/auth.py`) and
from the ticket - never from a `context` the HTTP caller sent. With no
`auth` entry in `langgraph.json`, every caller is anonymous and the role
gate refuses every tool call (Chapter 23).

`monitor` is a one-node graph around Chapter 21's `run_quality_monitor`, so
`atlas/deploy/schedule.py`'s cron has a graph to run: input
`{"sample_rate": float}`."""

from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel
from langgraph.runtime import Runtime

from atlas.auth import context_for
from atlas.graph import _make_builder, triage
from atlas.resolve import _agent_turn, mount_resolve_agent, reference_text
from atlas.state import AtlasState


def make_served_resolve_node(
    agent: CompiledStateGraph,
) -> Callable[[AtlasState, Runtime], dict]:
    """`make_resolve_node`, with the context built from the proved identity."""

    def resolve(state: AtlasState, runtime: Runtime) -> dict:
        user = runtime.server_info.user if runtime.server_info else None
        ticket = state.get("ticket") or {}
        sent = {m.id for m in state["messages"]}
        out = agent.invoke(
            {"messages": state["messages"], "reference": reference_text(state)},
            context=context_for(user, ticket.get("customer_id")),  # <1>
        )
        return {"messages": _agent_turn(out["messages"], sent)}

    return resolve


def build_served_graph(agent: CompiledStateGraph | None = None) -> Pregel:
    """Atlas's topology with the mounted agent, for the Agent Server."""
    node = make_served_resolve_node(agent or mount_resolve_agent())
    return _make_builder(triage, resolve_node=node).compile()  # <2>


class MonitorInput(TypedDict):
    sample_rate: float


def run_monitor(state: MonitorInput) -> dict:
    """Chapter 21's monitor, as the one node of a graph a cron can run."""
    from atlas.monitor import run_quality_monitor  # LangSmith client, on demand

    run_quality_monitor(state["sample_rate"])
    return {}


def build_monitor_graph() -> Pregel:
    builder = StateGraph(MonitorInput)
    builder.add_node("run_monitor", run_monitor)
    builder.add_edge(START, "run_monitor")
    builder.add_edge("run_monitor", END)
    return builder.compile()


resolve = build_served_graph()
monitor = build_monitor_graph()

# 1. An explicit `context=` replaces whatever the parent run carried, so a
#    `context` the HTTP caller sent never reaches the gates. `context_for`
#    reads the role off the identity `@auth.authenticate` proved; with no
#    proved role it returns "anonymous", which no role permission grants.
# 2. `.compile()` with no checkpointer and no store: the server supplies
#    both, and the nested agent inherits the store, so the gates write to
#    the server's store.
