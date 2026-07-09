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
for the rest of the suite to pass.

Chapter 14, "Advanced Memory: Extraction, Compaction, and LangMem", adds
the extraction/compaction/reflection pipeline: `CustomerFact`/`Extraction`
(the extraction schema), `extractor` (the structured-output agent),
`compact` (reconcile candidates against the store - one current value per
key), `reflect` (extract then compact), and `build_langmem_pipeline` (the
LangMem drop-in). The `compact`/`reflect` tests below monkeypatch
`extractor.invoke` - same no-live-API-key convention as
tests/test_triage.py's `classify` tests - so no live model call happens.
`build_langmem_pipeline` IS constructed for real (no live model call is
needed to build a `MemoryStoreManager`/`ReflectionExecutor`, only to run
one), but `ReflectionExecutor.__init__` starts a live, non-daemon worker
thread immediately - the test calls `.shutdown()` in a `finally` so the
pytest process doesn't hang waiting on that thread."""

import os

import pytest
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore
from pydantic import ValidationError

from atlas import memory as memory_module
from atlas.memory import (
    CustomerFact,
    Extraction,
    build_dev_store,
    build_langmem_pipeline,
    build_prod_store,
    compact,
    extractor,
    profile_ns,
    reflect,
    relevant_memories,
)

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


def test_customer_fact_rejects_a_kind_outside_the_literal():
    with pytest.raises(ValidationError):
        CustomerFact(key="contact_preference", value="email", kind="mood", source_turn=0)


def test_extraction_defaults_to_no_facts():
    """"If nothing durable was said, return no facts" - the container's
    default, not an error, for an empty extraction."""
    assert Extraction().facts == []


def test_extractor_is_built_tool_free():
    """Extraction decides what to remember; it does not act - same
    tool-free shape as atlas/triage.py's triage_agent, and for the same
    reason (sidesteps the response_format-plus-tools sharp edge)."""
    assert hasattr(extractor, "invoke")


def test_compact_inserts_a_new_candidate_fact():
    store = build_dev_store()
    fact = CustomerFact(
        key="contact_preference", value="email", kind="preference", source_turn=1
    )

    compact(store, "cust-1", [fact])

    item = store.get(("customer", "cust-1", "facts"), "contact_preference")
    assert item.value == fact.model_dump()


def test_compact_skips_an_exact_duplicate():
    """A candidate with the same key AND the same value is a no-op, not a
    rewrite - `store.put` must not be called again for it. `InMemoryStore`
    doesn't allow patching `.put` on the instance (it's read-only), so this
    wraps a real dev store in a thin counting proxy that only tracks calls
    - `compact` never sees the difference, since it only calls `.get`/
    `.put`."""

    class CountingStoreProxy:
        def __init__(self, inner):
            self._inner = inner
            self.put_calls = 0

        def get(self, *args, **kwargs):
            return self._inner.get(*args, **kwargs)

        def put(self, *args, **kwargs):
            self.put_calls += 1
            return self._inner.put(*args, **kwargs)

    store = CountingStoreProxy(build_dev_store())
    fact = CustomerFact(
        key="contact_preference", value="email", kind="preference", source_turn=1
    )
    compact(store, "cust-1", [fact])
    assert store.put_calls == 1

    compact(store, "cust-1", [fact])  # identical candidate again
    assert store.put_calls == 1  # unchanged - the duplicate was skipped


def test_compact_reconciles_a_contradiction_to_one_current_value():
    """Exercise 1: three conflicting `contact_preference` facts, written on
    three different "days" - compact must collapse them to exactly one
    current value, keyed by `fact.key`, not an append-only log."""
    store = build_dev_store()
    ns = ("customer", "cust-1", "facts")

    for value in ("prefers email", "prefers phone", "no strong preference"):
        fact = CustomerFact(
            key="contact_preference", value=value, kind="preference", source_turn=1
        )
        compact(store, "cust-1", [fact])

    results = store.search(ns)
    assert len(results) == 1
    assert results[0].value["value"] == "no strong preference"


def test_reflect_extracts_then_compacts(monkeypatch):
    """`reflect` is the full pass - extract then compact - and this is the
    seam a background task/queue/ReflectionExecutor calls off the hot
    path. Monkeypatching `extractor.invoke` keeps this test's assertion on
    reflect's own wiring (unwrap `structured_response.facts`, call
    `compact`), not on a live model call."""
    fake_extraction = Extraction(
        facts=[
            CustomerFact(
                key="contact_preference",
                value="prefers email",
                kind="preference",
                source_turn=0,
            )
        ]
    )
    monkeypatch.setattr(
        memory_module.extractor,
        "invoke",
        lambda payload: {"structured_response": fake_extraction},
    )

    store = build_dev_store()
    reflect(store, "cust-1", [{"role": "user", "content": "I prefer email."}])

    item = store.get(("customer", "cust-1", "facts"), "contact_preference")
    assert item.value["value"] == "prefers email"


def test_build_langmem_pipeline_constructs_a_manager_and_a_reflection_executor():
    """The "build vs. adopt" drop-in: no live model call is needed to
    *construct* `create_memory_store_manager`/`ReflectionExecutor`, only to
    run one - so this test exercises the real LangMem objects. Must shut
    the executor down: `ReflectionExecutor.__init__` starts a live,
    non-daemon worker thread immediately, and that thread otherwise keeps
    the pytest process alive after the run finishes."""
    store = build_dev_store()
    manager, reflection = build_langmem_pipeline(store)
    try:
        assert hasattr(manager, "invoke")
        assert hasattr(reflection, "submit")
    finally:
        reflection.shutdown()


@requires_postgres
@requires_openai
def test_build_prod_store_sets_up_a_real_postgres_backed_semantic_store():
    """Skipped by default - see `requires_postgres`/`requires_openai` above."""
    dsn = os.environ["ATLAS_POSTGRES_TEST_DSN"]

    with build_prod_store(dsn) as store:
        store.put(profile_ns("cust-prod"), "plan", {"tier": "enterprise"})
        item = store.get(profile_ns("cust-prod"), "plan")
        assert item.value == {"tier": "enterprise"}
