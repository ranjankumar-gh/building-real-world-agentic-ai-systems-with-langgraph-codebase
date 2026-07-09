"""Chapter 13, "Short-Term vs Long-Term Memory" - atlas/memory.py, a
BaseStore-backed long-term memory that persists customer facts across
conversations, independent of any `thread_id`.

See "Building long-term memory". The checkpointer (Chapter 9) is
short-term: it holds one conversation's state, scoped to `thread_id`, and is
meant to be forgotten when the thread ends. The store is long-term: a
`BaseStore` that persists JSON under a namespace tuple and a key,
independent of any thread, configured beside the checkpointer
(`compile(store=...)` / `create_agent(..., store=...)`) rather than
replacing it.

`atlas/graph.py`'s `remember`/`recall` are the store's node-facing side -
they reach it through `runtime.store`, the same handle the graph gives every
node. This module holds the store's own shape: the namespace convention
that makes it a privacy boundary (`profile_ns`), semantic search over it
(`relevant_memories`), and the dev/prod backend swap (`build_dev_store`/
`build_prod_store`), which mirrors Chapter 9's checkpointer swap exactly -
same interface, different backend, `.setup()` as a migration.

Chapter 14, "Advanced Memory: Extraction, Compaction, and LangMem", turns
the store into a memory that learns. `CustomerFact`/`Extraction` and
`extractor` are the structured-output extraction step (Chapter 7's
`response_format` pattern, same shape as `atlas/triage.py`'s
`triage_agent`) - candidate facts, never trusted until reconciled.
`compact` is the reconciliation step: keyed by `fact.key`, it updates,
skips duplicates, or overwrites, so the store holds one current value per
key instead of an append-only log. `reflect` is the full pass - extract
then compact - meant to run AFTER the response returns (a background task,
a queue, or LangMem's `ReflectionExecutor`), never on the hot path. Note
the different namespace: `compact`/`reflect` write to `("customer", cid,
"facts")`, not `profile_ns`'s `"profile"` - this chapter's reconciled-fact
store is deliberately separate from Chapter 13's simple profile store, not
a replacement for it.

`build_langmem_pipeline` is the "build vs. adopt" section's drop-in:
`create_memory_store_manager` plus `ReflectionExecutor` do the same
extract-then-compact-off-the-hot-path work as `reflect` above, as a
LangMem primitive instead of hand-rolled code. LangMem is a `0.0.x`
package (pinned exactly in `pyproject.toml`, per the chapter's caution
about pre-1.0 dependencies in the memory layer) - isolated behind this one
function so a breaking release touches this module, not every call site.
Constructing `ReflectionExecutor` starts a live, non-daemon background
worker thread immediately (not on first `.submit()`) - callers (and every
test that builds one) must call `.shutdown()` when done, or the process
never exits.
"""

from contextlib import contextmanager
from typing import Iterator, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langgraph.store.base import BaseStore, IndexConfig
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

DB_URI = "postgresql://atlas:atlas@localhost:5432/atlas"


def profile_ns(customer_id: str) -> tuple[str, ...]:
    """The namespace for one customer's long-term profile. Scoping by
    customer_id is the privacy boundary - a search in one customer's
    namespace cannot return another's memory."""
    return ("customer", customer_id, "profile")


def relevant_memories(store: BaseStore, customer_id: str, question: str) -> list:
    """Semantic recall: the memories most relevant to the question, not just
    an exact key match."""
    return store.search(
        ("customer", customer_id),
        query=question,  # ranked by similarity when an index is configured
        limit=5,  # cap to the retrieved slice - this is Chapter 12's budget
    )


def build_dev_store() -> InMemoryStore:
    """Development: a dict in RAM, gone on restart, no infrastructure. The
    dev/test default - exercises the exact `BaseStore` interface every node
    uses, with no external service."""
    return InMemoryStore()


@contextmanager
def build_prod_store(db_uri: str = DB_URI) -> Iterator[BaseStore]:
    """Production: durable + semantic search. `IndexConfig` is what turns
    `search`'s `query` from a no-op into semantic recall: `embed` is the
    embedding function, `dims` must match that model's output dimension
    (1536 for text-embedding-3-small), and `fields` selects which parts of
    each memory to embed. Get `dims` wrong and writes and queries land in
    different vector spaces; recall silently returns nothing useful.

    Requires a live Postgres instance and embedding-provider credentials -
    the external-service exception (see tests/test_memory.py), not part of
    the seeded, mockable default path."""
    from langchain.embeddings import init_embeddings
    from langgraph.store.postgres import PostgresStore

    embeddings = init_embeddings("openai:text-embedding-3-small")
    with PostgresStore.from_conn_string(
        db_uri,
        index=IndexConfig(embed=embeddings, dims=1536, fields=["$"]),
    ) as store:
        store.setup()  # create tables + the vector index (a migration step)
        yield store


class CustomerFact(BaseModel):
    """One durable, atomic fact about a customer, tied to its source.
    Chapter 14: the extraction step's output shape - a candidate, never
    trusted until `compact` reconciles it against the store."""

    key: str = Field(description="Stable slug, e.g. 'contact_preference'.")
    value: str = Field(description="The fact, stated plainly.")
    kind: Literal["preference", "account", "issue"]
    source_turn: int = Field(description="Index of the message it came from.")


class Extraction(BaseModel):
    """Container so the schema is a single model, not a bare list - a bare
    list response_format is provider-dependent."""

    facts: list[CustomerFact] = Field(default_factory=list)


EXTRACTION_PROMPT = (
    "Extract only facts the customer explicitly stated. Do not infer or "
    "guess. If nothing durable was said, return no facts."
)

extractor = create_agent(
    model="claude-sonnet-4-6",
    tools=[],  # extraction decides what to remember; it does not act
    response_format=ProviderStrategy(Extraction),
    system_prompt=EXTRACTION_PROMPT,
)


def compact(store: BaseStore, customer_id: str, candidates: list[CustomerFact]) -> None:
    """Reconcile candidates against stored facts: one current value per key,
    not an append-only log."""
    ns = ("customer", customer_id, "facts")
    for fact in candidates:
        existing = store.get(ns, fact.key)
        if existing and existing.value["value"] == fact.value:
            continue  # duplicate - skip
        store.put(ns, fact.key, fact.model_dump())  # insert or overwrite


def reflect(store: BaseStore, customer_id: str, messages: list) -> None:
    """The full reflection pass - extract then compact. Runs AFTER the
    response, scheduled off the hot path (a background task, a queue, or
    LangMem's ReflectionExecutor - see `build_langmem_pipeline` below)."""
    result = extractor.invoke({"messages": messages})
    compact(store, customer_id, result["structured_response"].facts)


def build_langmem_pipeline(store: BaseStore):
    """The "build vs. adopt" section's LangMem drop-in for the hand-rolled
    extractor/compact/reflect pipeline above (Exercise 3: swap it in, then
    write the build-vs-adopt decision note). `create_memory_store_manager`
    runs extraction and reconciliation against a `BaseStore` in one call;
    `ReflectionExecutor` is the background-reflection move as a first-class
    primitive - its `.submit(...)` schedules the memory work and returns
    immediately, so the caller's response never waits on it.

    Requires the `langmem` package (pinned to an exact 0.0.x version in
    pyproject.toml - see the module docstring's caution). Returns
    `(manager, reflection)`; call `reflection.shutdown()` when done with
    it - constructing `ReflectionExecutor` starts a live worker thread
    immediately, and that thread keeps the process alive until shut down."""
    from langmem import ReflectionExecutor, create_memory_store_manager

    manager = create_memory_store_manager(
        "claude-sonnet-4-6",
        namespace=("customer", "{customer_id}", "facts"),
        store=store,
    )
    reflection = ReflectionExecutor(manager, store=store)
    return manager, reflection
