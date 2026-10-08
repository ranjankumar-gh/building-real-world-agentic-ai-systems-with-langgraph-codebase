"""Chapter 18, "Deep Agents: The Production Harness" - Atlas's research arm
rebuilt on `create_deep_agent` instead of the Chapter 16 hand-rolled
supervisor or the Chapter 17 `Send`-based map-reduce subgraph.

See "Building the Deep Research Agent". `source_lookup` reuses
`search_source`/`SourceUnavailable` from `atlas/research.py` rather than
inventing a new backend - only the shape around them changes: a Deep Agent
sub-agent tool instead of a `research_worker` reduce-side node. The
partial-failure discipline is identical to Chapter 17's: a dead source
becomes an error STRING the sub-agent can read and report on, not a raised
exception that kills its turn.

`source_researcher` is a `SubAgent` declaration - name, description,
system_prompt, tools - not a hand-wired `Command`-returning tool the way
Chapter 16's `make_handoff` built one. The harness generates the handoff
tool (`task`), the isolated context, and the result aggregation.

`research_namespace` scopes research artifacts the same way Chapter 13's
`profile_ns` scopes profile facts. It is written against the ACTUAL
`deepagents==0.6.x` `NamespaceFactory` contract, not the chapter's earlier
draft: `StoreBackend` calls a namespace factory with its own `Runtime`
instance, never the run's `config` dict (verified against
`deepagents.backends.store.NamespaceFactory` and `langgraph.runtime.
Runtime`'s own docstring, which says plainly that `Runtime` does not carry
`config` - use `get_config()` from `langgraph.config` instead, or inject
`config` as a node/tool parameter). A `config["configurable"][...]`-style
function passed as `namespace=` raises `TypeError: '_NamespaceRuntimeCompat'
object is not subscriptable` the moment `StoreBackend` calls it - confirmed
against the installed 0.6.3 package, not assumed from the docs. This module
therefore takes the `runtime` argument `StoreBackend` actually passes (named
for what it is, even though the factory here does not otherwise use it) and
reads `customer_id` out of `get_config()`, the accessor `StoreBackend`'s own
legacy path already uses internally for the same purpose.

`research_namespace` also applies Chapter 13's `SAFE_ID` check before it
builds the tuple. `StoreBackend`'s `ls`/`glob`/`grep` list files with
`store.search(namespace)`, which takes a namespace PREFIX, and PostgresStore
matches a prefix as `prefix LIKE '<dot-joined namespace>%'`. deepagents'
own validator allows "_" and "." in a namespace component, and "_" is a
LIKE wildcard, so an unchecked "cust_42" would list "cust-42"'s research
files. `SAFE_ID` refuses "_", "." and "%", the same rule `profile_ns`
enforces for profile facts.

`checkpointer`/`store` are the same Chapter 9 / Chapter 13 dev defaults
`atlas/graph.py` and `atlas/memory.py` already establish - `InMemorySaver`
and `build_dev_store()` - so this module needs no new infrastructure to
import or test against; a production deploy swaps them the same way Chapter
9's `run_durable` and Chapter 13's `build_prod_store` already do.

Chapter 19, "Streaming", adds one line to `source_lookup` below - a
`get_stream_writer()` progress emission - so that name, like
`search_source`/`SourceUnavailable`, is load-bearing beyond this chapter.
`get_stream_writer()` requires an active runnable context (a real graph or
agent run); calling `source_lookup.func(...)` directly, with no run
underneath it, now raises `RuntimeError` - see
tests/test_deep_research.py, which exercises the tool from inside a real
(tiny) compiled graph instead, the same discipline Chapter 18's
`research_namespace` tests already established for `get_config()`.

Chapter 20, "Observability and Debugging with LangSmith", makes no code
change here: `source_researcher["name"]` was already required by the
`create_deep_agent` harness (it is not optional on a `SubAgent`), so unlike
`atlas/agent.py`'s `resolve_agent` or `atlas/research.py`'s `supervisor` -
which needed a `name=` ADDED - this value was already doing tracing work
without anyone deciding it should. See "Naming the fleet: attribution
across the supervisor topology": every ephemeral sub-agent the `task` tool
spawns shows up in a trace tree under `source_researcher`, distinguishing
"three sub-agents spawned" from "the source_researcher ran three times."

Chapter 23, "Security, Privacy, and Governance", applies here too, through
Appendix E: source text is untrusted content, and three models read it.
`InjectionGuard` (atlas/security.py) screens every tool result each of them
sees - the main agent (`source_lookup`, and the `/findings/` files it reads
back), `source_researcher`, and the harness's general-purpose sub-agent,
which inherits the main agent's tools. deepagents builds that sub-agent's
middleware stack WITHOUT the caller's `middleware=`, so it is re-declared
under its own name with the guard added; the harness then uses that spec in
place of its default. Chapter 18's printed listing predates the guard.
"""

from deepagents import create_deep_agent
from deepagents.backends.store import StoreBackend
from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config, get_stream_writer
from langgraph.runtime import Runtime

from atlas.memory import SAFE_ID, build_dev_store
from atlas.research import SourceUnavailable, search_source
from atlas.security import InjectionGuard


@tool
def source_lookup(source: str) -> str:
    """Look up findings for a single research source."""
    # Chapter 19, "Streaming": the only line this chapter adds. get_stream_writer()
    # reports progress on the "custom" channel from INSIDE the tool's own execution -
    # something stream_mode="updates" cannot see, because it only reports what a node
    # returns, not what it does mid-run. Requires an active runnable context (a real
    # graph/agent run); called outside one it raises RuntimeError - see
    # tests/test_deep_research.py.
    writer = get_stream_writer()
    writer({"progress": f"researching {source}"})
    try:
        return search_source(source)
    except SourceUnavailable as exc:
        return f"error: {exc}"  # same partial-failure discipline as Ch17


source_researcher = {
    "name": "source_researcher",
    "description": "Investigates one research source and writes a finding to disk.",
    "system_prompt": (
        "Investigate the assigned source using source_lookup. Write your finding to "
        "/findings/<source>.md via write_file. Investigate only the assigned source."
    ),
    "tools": [source_lookup],
    "middleware": [InjectionGuard()],  # Chapter 23: source text is untrusted
}

# The harness's own general-purpose sub-agent, with the same guard.
general_purpose = {**GENERAL_PURPOSE_SUBAGENT, "middleware": [InjectionGuard()]}


def research_namespace(runtime: Runtime) -> tuple[str, str, str]:
    """Scope research artifacts the same way Chapter 13 scoped profile facts.

    `StoreBackend` calls this with its own `Runtime`, not the run's `config`
    dict - `Runtime` deliberately does not carry `config` (see the module
    docstring), so `customer_id` is read back out of the active
    `RunnableConfig` via `get_config()` instead of a `runtime[...]` lookup.
    An id that could widen a prefix match is refused (see `SAFE_ID`).
    """
    customer_id = get_config()["configurable"]["customer_id"]
    if not SAFE_ID.fullmatch(customer_id):
        raise ValueError(f"unsafe customer id: {customer_id!r}")
    return ("customer", customer_id, "research")


# Chapter 9 / Chapter 13 dev defaults; production swaps both.
checkpointer = InMemorySaver()
store = build_dev_store()

deep_research_agent = create_deep_agent(
    model="claude-sonnet-4-6",
    tools=[source_lookup],
    system_prompt=(
        "You are Atlas's deep research agent. Break the request into sources, delegate "
        "each to source_researcher via the task tool, track progress with write_todos, "
        "and compose a final report from the files under /findings/."
    ),
    subagents=[source_researcher, general_purpose],
    middleware=[InjectionGuard()],
    backend=StoreBackend(store=store, namespace=research_namespace),
    checkpointer=checkpointer,
    store=store,
)
