"""Chapter 13, "Short-Term vs Long-Term Memory" - atlas/memory.py, a
BaseStore-backed long-term memory that persists customer facts across
conversations, independent of any `thread_id`.

See "Building long-term memory". The checkpointer (Chapter 9) is
short-term: it holds one conversation's state, scoped to `thread_id`, and no
other conversation can reach it. The store is long-term: a `BaseStore` that
persists JSON under a namespace tuple and a key, independent of any thread,
configured beside the checkpointer (`compile(store=...)`) rather than
replacing it.

`atlas/graph.py`'s `recall`/`remember` are the store's node-facing side, and
they are wired into Atlas: `recall` runs before `triage`, `remember` after
`answer`. Both reach the store through `runtime.store`. This module holds
the store's own shape: the namespace convention that makes it a privacy
boundary (`profile_ns`), search over one customer's profile
(`relevant_memories`, which `recall` calls), and the dev/prod backend swap
(`build_dev_store`/`build_prod_store`), which mirrors Chapter 9's
checkpointer swap: same interface, different backend, and `setup()` run
once per release as a migration (`scripts/setup_store.py`), never on every
process start.

Namespace matching is the privacy boundary, and the backends differ.
InMemoryStore matches a namespace prefix label by label. PostgresStore
matches it as text: `search(("customer", "12"))` becomes `prefix LIKE
'customer.12%'`, which also matches customer 123, and `_` or `%` inside an
id are LIKE wildcards. So `profile_ns` refuses ids outside letters, digits,
and hyphens, `relevant_memories` searches the complete three-label
namespace, never the bare `("customer", customer_id)` prefix, and it keeps
only items whose namespace is exactly that one. See
tests/test_memory.py's 12-vs-123 tests.

Chapter 14, "Advanced Memory: Extraction, Compaction, and LangMem", turns
the store into a memory that learns. `CustomerFact`/`Extraction` and
`extractor` are the structured-output extraction step (Chapter 7's
`response_format` pattern, same shape as `atlas/triage.py`'s
`triage_agent`) - candidate facts, never trusted until checked and
reconciled. `reflect` drops any candidate whose `source_turn` does not
point at a real customer turn of the conversation, then hands the rest to
`compact`, which reconciles them by `fact.key`: an identical value is
skipped, anything else overwrites, so the profile holds one current value
per key. `compact` writes into the same `profile_ns` namespace `remember`
writes, so `recall` surfaces extracted facts on the customer's next thread
with no change. `reflect` runs AFTER the response returns, never on the
hot path: atlas/run.py's `run_and_reflect` submits it to a one-worker pool.

`build_langmem_pipeline` is the "build vs. adopt" section's drop-in:
`create_memory_store_manager` plus `ReflectionExecutor`. LangMem stores
memories in its own format (`{"kind", "content"}` under random ids) and
cannot read the facts `compact` writes, so it gets its own namespace,
`("customer", "{customer_id}", "langmem")`, filled from the `configurable`
dict passed to `submit` (`submit_langmem_reflection`). LangMem is a `0.0.x`
package (pinned exactly in `pyproject.toml`), isolated behind these two
functions so a breaking release touches this module, not every call site.
Constructing `ReflectionExecutor` starts a live, non-daemon background
worker thread immediately (not on first `.submit()`) - callers (and every
test that builds one) must call `.shutdown()` when done, or the process
never exits.
"""

import re
import typing
from collections.abc import Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AnyMessage
from langgraph.store.base import BaseStore, IndexConfig, Item
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

if typing.TYPE_CHECKING:  # annotations only: langmem stays a lazy import
    from langmem.knowledge.extraction import MemoryStoreManager
    from langmem.reflection import LocalReflectionExecutor

DB_URI = "postgresql://atlas:atlas@localhost:5432/atlas"


SAFE_ID = re.compile(r"[A-Za-z0-9-]+")  # no ".", "%", or "_" in a label


def profile_ns(customer_id: str) -> tuple[str, ...]:
    """The namespace for one customer's long-term profile. Scoping by
    customer_id is the privacy boundary, so an id that could widen a
    match is refused, not stored."""
    if not SAFE_ID.fullmatch(customer_id):
        raise ValueError(f"unsafe customer id: {customer_id!r}")
    return ("customer", customer_id, "profile")


def relevant_memories(
    store: BaseStore, customer_id: str, question: str, limit: int = 5
) -> list[Item]:
    """The profile entries most relevant to the question, from this
    customer's namespace and no other."""
    ns = profile_ns(customer_id)
    hits = store.search(
        ns,                      # the full namespace, never ("customer", id)
        query=question,          # ranked by similarity when an index is configured
        limit=limit * 4,         # over-fetch: the filter below may drop some
    )
    own = [item for item in hits if item.namespace == ns]   # exact match only
    return own[:limit]           # the profile's own cap on what enters the prompt


def build_dev_store() -> InMemoryStore:
    """Development: a dict in RAM, gone on restart, no infrastructure.
    The dev/test default - exercises the exact `BaseStore` interface every
    node uses, with no external service."""
    return InMemoryStore()


@contextmanager
def build_prod_store(db_uri: str = DB_URI) -> Iterator[BaseStore]:
    """Production: durable + semantic search.

    `IndexConfig` is what turns `search`'s `query` from a no-op into
    semantic recall: `embed` is the embedding function, `dims` must match
    that model's output dimension (1536 for text-embedding-3-small), and
    `fields` selects which parts of each memory to embed. The vector column
    is created `dims` wide, so a mismatched model fails loudly at the first
    write; the silent failure is a different model with the same width.

    No `setup()` here: the tables, the vector index, and the pgvector
    extension are created once per release by `scripts/setup_store.py`, as
    Chapter 9's `scripts/setup_checkpointer.py` does for the checkpointer.
    Requires a live Postgres instance and embedding-provider credentials -
    the external-service exception (see tests/test_memory.py)."""
    from langchain.embeddings import init_embeddings
    from langgraph.store.postgres import PostgresStore

    embeddings = init_embeddings("openai:text-embedding-3-small")
    with PostgresStore.from_conn_string(
        db_uri,
        index=IndexConfig(embed=embeddings, dims=1536, fields=["$"]),
    ) as store:
        yield store      # compile the graph with store=store


class CustomerFact(BaseModel):
    """One durable, atomic fact about a customer, tied to its source.
    Chapter 14: the extraction step's output shape - a candidate, never
    trusted until `compact` reconciles it against the store."""

    key: str = Field(description="Stable slug, e.g. 'contact_preference'.")
    value: str = Field(description="The fact, stated plainly.")
    kind: Literal["preference", "account", "issue"]
    source_turn: int = Field(description="The [n] number of its message.")


class Extraction(BaseModel):
    """Container so the schema is a single model, not a bare list.
    A bare-list response_format is provider-dependent."""

    facts: list[CustomerFact] = Field(default_factory=list)


# Extraction decides what to remember; it does not act, so it has no tools.
extractor = create_agent(
    model="claude-sonnet-4-6",
    tools=[],
    response_format=ProviderStrategy(Extraction),
    system_prompt=(
        "Extract only facts the customer explicitly stated. Do not infer or "
        "guess. If nothing durable was said, return no facts. Each message "
        "starts with its number in brackets; set source_turn to that number."
    ),
)


def compact(
    store: BaseStore, customer_id: str, candidates: list[CustomerFact]
) -> None:
    """Reconcile candidates against stored facts: one current value per key,
    not an append-only log."""
    ns = profile_ns(customer_id)        # the profile recall already reads
    for fact in candidates:
        existing = store.get(ns, fact.key)
        if existing and existing.value["value"] == fact.value:
            continue                                   # duplicate - skip
        store.put(ns, fact.key, fact.model_dump())     # insert or overwrite


def reflect(store: BaseStore, customer_id: str, messages: list[AnyMessage]) -> None:
    """The full reflection pass - extract, check, compact. Runs AFTER the
    response, scheduled off the hot path."""
    numbered = [  # the model cites the [n] it sees, not a position it counts
        m.model_copy(update={"content": f"[{i}] {m.text}"})
        for i, m in enumerate(messages)
    ]
    result = extractor.invoke({"messages": numbered})
    sourced = [
        fact
        for fact in result["structured_response"].facts
        if 0 <= fact.source_turn < len(messages)
        and messages[fact.source_turn].type == "human"   # a real customer turn
    ]
    compact(store, customer_id, sourced)


def build_langmem_pipeline(
    store: BaseStore,
) -> tuple["MemoryStoreManager", "LocalReflectionExecutor"]:
    """LangMem's manager and executor, behind one seam.

    The "build vs. adopt" section's drop-in for the hand-rolled
    extractor/compact/reflect pipeline above (Exercise 3). Returns
    `(manager, reflection)`; call `reflection.shutdown()` when done -
    constructing `ReflectionExecutor` starts a live worker thread
    immediately, and that thread keeps the process alive until shut down.
    LangMem writes its own format, so it gets its own namespace."""
    from langmem import ReflectionExecutor, create_memory_store_manager

    manager = create_memory_store_manager(
        "claude-sonnet-4-6",
        namespace=("customer", "{customer_id}", "langmem"),
        store=store,
    )
    reflection = ReflectionExecutor(manager, store=store)
    return manager, reflection


def submit_langmem_reflection(
    reflection: "LocalReflectionExecutor",
    customer_id: str,
    thread_id: str,
    result: dict,
) -> Future:
    """Defer LangMem's memory work for one customer's conversation.

    `{customer_id}` in the manager's namespace is a template filled from
    this `configurable` dict; without a config, `submit` outside a graph run
    raises ValueError. `thread_id` is LangMem's pending-work key: a later
    submit for the same thread replaces a pending one, and with no
    `thread_id` every submit shares one key, so customer C-2's submit
    cancels customer C-1's pending reflection (and two submits on the same
    clock reading raise TypeError comparing the queued tasks). Log the
    returned Future: a failed reflection raises nowhere else."""
    if not SAFE_ID.fullmatch(customer_id):  # LangMem's namespace takes it verbatim
        raise ValueError(f"unsafe customer id: {customer_id!r}")
    configurable = {"customer_id": customer_id, "thread_id": thread_id}
    return reflection.submit(
        {"messages": result["messages"]},
        config={"configurable": configurable},
        after_seconds=0,
    )
