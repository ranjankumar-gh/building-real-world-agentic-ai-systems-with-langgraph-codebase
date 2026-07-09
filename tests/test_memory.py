"""Chapter 13, "Short-Term vs Long-Term Memory" - atlas/memory.py's
BaseStore-backed long-term memory: the namespace convention that makes it a
privacy boundary (`profile_ns`), semantic search over it
(`relevant_memories`), and the dev/prod backend swap (`build_dev_store` /
`build_prod_store`).

`build_prod_store` needs a live Postgres instance and a real embedding
provider (`init_embeddings("openai:text-embedding-3-small")`, which itself
requires the `langchain-openai` package and API credentials) - the
external-service exception, same shape as tests/test_run.py's
`ATLAS_POSTGRES_TEST_DSN` guard. It is skip-guarded below and not required
for the rest of the suite to pass."""

import os

import pytest
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from atlas.memory import build_dev_store, build_prod_store, profile_ns, relevant_memories

requires_postgres = pytest.mark.skipif(
    not os.environ.get("ATLAS_POSTGRES_TEST_DSN"),
    reason="requires a live Postgres connection (ATLAS_POSTGRES_TEST_DSN)",
)
requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="requires a live embedding provider (OPENAI_API_KEY)",
)


def test_profile_ns_scopes_by_customer_id():
    """The namespace is the privacy boundary: two different customers get
    two different, non-overlapping namespaces."""
    assert profile_ns("cust-1") == ("customer", "cust-1", "profile")
    assert profile_ns("cust-1") != profile_ns("cust-2")


def test_build_dev_store_is_a_working_in_memory_store():
    store = build_dev_store()
    assert isinstance(store, InMemoryStore)

    store.put(profile_ns("cust-1"), "plan", {"tier": "pro"})
    item = store.get(profile_ns("cust-1"), "plan")

    assert item.value == {"tier": "pro"}


def test_namespaces_isolate_two_customers_from_each_other():
    """Exercise 2: store facts for two different customers, then confirm a
    search in one customer's namespace never returns the other's."""
    store = build_dev_store()
    store.put(profile_ns("cust-a"), "plan", {"tier": "enterprise"})
    store.put(profile_ns("cust-b"), "plan", {"tier": "free"})

    results_a = store.search(("customer", "cust-a"))
    results_b = store.search(("customer", "cust-b"))

    assert [item.value for item in results_a] == [{"tier": "enterprise"}]
    assert [item.value for item in results_b] == [{"tier": "free"}]


def test_relevant_memories_searches_within_the_customer_namespace_and_caps_at_limit():
    store = build_dev_store()
    for i in range(7):
        store.put(profile_ns("cust-1"), f"note-{i}", {"text": f"fact {i}"})
    # A different customer's memory must never leak into the search below.
    store.put(profile_ns("cust-2"), "note-0", {"text": "someone else's fact"})

    results = relevant_memories(store, "cust-1", "what does this customer prefer?")

    assert len(results) == 5  # capped to the retrieved slice (limit=5)
    assert all(item.namespace[1] == "cust-1" for item in results)


def test_a_node_reaches_the_store_through_runtime_store():
    """`runtime.store` is the same handle every node receives - the pattern
    atlas/graph.py's `remember`/`recall` use."""
    store = build_dev_store()
    runtime = Runtime(store=store)

    runtime.store.put(profile_ns("cust-1"), "plan", {"tier": "enterprise"})
    item = runtime.store.get(profile_ns("cust-1"), "plan")

    assert item.value == {"tier": "enterprise"}


@requires_postgres
@requires_openai
def test_build_prod_store_sets_up_a_real_postgres_backed_semantic_store():
    """Skipped by default - see `requires_postgres`/`requires_openai` above."""
    dsn = os.environ["ATLAS_POSTGRES_TEST_DSN"]

    with build_prod_store(dsn) as store:
        store.put(profile_ns("cust-prod"), "plan", {"tier": "enterprise"})
        item = store.get(profile_ns("cust-prod"), "plan")
        assert item.value == {"tier": "enterprise"}
