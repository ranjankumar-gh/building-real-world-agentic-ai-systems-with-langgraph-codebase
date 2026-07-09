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
"""

from contextlib import contextmanager
from typing import Iterator

from langgraph.store.base import BaseStore, IndexConfig
from langgraph.store.memory import InMemoryStore

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
