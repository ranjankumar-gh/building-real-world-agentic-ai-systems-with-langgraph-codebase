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
pytest process doesn't hang waiting on that thread.

The 12-vs-123 tests reproduce the namespace-prefix leak PostgresStore's
text matching causes. No Postgres is needed for the default suite: one test
builds the real search SQL with the pinned `PostgresStore` (conn=None, no
query is ever executed) and evaluates its LIKE pattern; another runs
`relevant_memories` against `LikePrefixStore`, an InMemoryStore wrapper
whose `search` applies that same pattern the way Postgres does. A third,
skip-guarded on `ATLAS_POSTGRES_TEST_DSN`, runs both against a live
PostgresStore."""

import os
import re
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.config import get_config, get_store
from langgraph.runtime import Runtime
from langgraph.store.base import SearchOp
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.base import PostgresStore, _namespace_to_text
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
    submit_langmem_reflection,
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
    store.put(profile_ns("cust-a"), "plan", {"value": "enterprise"})
    store.put(profile_ns("cust-b"), "plan", {"value": "free"})

    results_a = relevant_memories(store, "cust-a", "what plan?")
    results_b = relevant_memories(store, "cust-b", "what plan?")

    assert [item.value for item in results_a] == [{"value": "enterprise"}]
    assert [item.value for item in results_b] == [{"value": "free"}]


# --- 12 vs 123: PostgresStore matches a namespace prefix as text ----------


def _like(pattern: str, text: str) -> bool:
    """SQL LIKE, as Postgres evaluates it: `%` is any run, `_` any one
    character, everything else literal (case-sensitive)."""
    rx = "".join(
        ".*" if ch == "%" else "." if ch == "_" else re.escape(ch) for ch in pattern
    )
    return re.fullmatch(rx, text, flags=re.DOTALL) is not None


def _postgres_like_pattern(namespace_prefix: tuple[str, ...]) -> str:
    """The exact LIKE parameter the pinned PostgresStore builds for a
    search on this prefix. Built offline: conn=None, nothing is executed."""
    store = PostgresStore(conn=None)
    op = SearchOp(namespace_prefix=namespace_prefix, filter=None, limit=5, offset=0)
    [(sql, params)], _ = store._prepare_batch_search_queries([(0, op)])
    assert "store.prefix LIKE %s" in sql
    return params[0]


class LikePrefixStore:
    """InMemoryStore with PostgresStore's prefix semantics: `search` keeps
    every item whose dotted namespace text matches the prefix's LIKE
    pattern, built by the pinned `_namespace_to_text`."""

    def __init__(self) -> None:
        self.inner = InMemoryStore()

    def put(self, namespace: tuple[str, ...], key: str, value: dict) -> None:
        self.inner.put(namespace, key, value)

    def search(self, namespace_prefix: tuple[str, ...], *, query=None, limit=10):
        pattern = f"{_namespace_to_text(namespace_prefix)}%"
        every = self.inner.search((), limit=10_000)
        hits = [i for i in every if _like(pattern, _namespace_to_text(i.namespace))]
        return hits[:limit]


def test_postgres_prefix_search_sql_matches_customer_123_for_customer_12():
    """The leak, reproduced against the pinned query construction: the old
    two-label prefix ("customer", "12") becomes LIKE 'customer.12%', which
    customer 123's profile text satisfies. The full profile namespace does
    not, and an id carrying a LIKE wildcard would widen the match again -
    which is why profile_ns refuses it."""
    old = _postgres_like_pattern(("customer", "12"))
    new = _postgres_like_pattern(profile_ns("12"))

    assert old == "customer.12%"
    assert _like(old, "customer.123.profile")  # the leak
    assert new == "customer.12.profile%"
    assert not _like(new, "customer.123.profile")  # closed
    assert _like(new, "customer.12.profile")
    assert _like("customer.1_.profile%", "customer.12.profile")  # why "_" is refused


def test_old_prefix_search_leaks_customer_123_and_relevant_memories_does_not():
    """Same leak, end to end through a store with Postgres's matching. The
    pre-fix search, `store.search(("customer", customer_id), ...)`, returns
    customer 123's memory to customer 12; `relevant_memories` returns only
    customer 12's. InMemoryStore matches label by label, so it hides the
    leak - the reason the dev store cannot be the only test."""
    store = LikePrefixStore()
    store.put(profile_ns("12"), "last_issue", {"value": "sync fails"})
    store.put(profile_ns("123"), "last_issue", {"value": "card 4242 declined"})

    old_logic = store.search(("customer", "12"), query="any", limit=5)
    leaked = {item.namespace[1] for item in old_logic}
    assert leaked == {"12", "123"}  # reproduced

    fixed = relevant_memories(store, "12", "any")
    assert [(i.namespace[1], i.value["value"]) for i in fixed] == [("12", "sync fails")]

    dev = InMemoryStore()
    dev.put(profile_ns("12"), "last_issue", {"value": "sync fails"})
    dev.put(profile_ns("123"), "last_issue", {"value": "card 4242 declined"})
    assert {i.namespace[1] for i in dev.search(("customer", "12"))} == {"12"}


@pytest.mark.parametrize("bad", ["1_", "12%", "a.b", "", "x y", "c@example"])
def test_profile_ns_refuses_ids_that_could_widen_a_match(bad):
    with pytest.raises(ValueError):
        profile_ns(bad)


def test_relevant_memories_keeps_only_the_exact_namespace():
    """Belt and braces: even a store that returned a sub-namespace or a
    neighbor would not get it past the exact-match filter."""
    store = LikePrefixStore()
    store.put(profile_ns("12"), "a", {"value": "mine"})
    store.put((*profile_ns("12"), "archive"), "b", {"value": "sub-namespace"})

    assert [i.value["value"] for i in relevant_memories(store, "12", "q")] == ["mine"]


def test_relevant_memories_over_fetches_so_sub_namespace_hits_cannot_starve_it():
    """Five sub-namespace items ahead of the two real entries would fill a
    plain limit=5 page and leave nothing after the exact-namespace filter;
    over-fetching, then filtering, then slicing keeps the real entries."""
    store = LikePrefixStore()
    for i in range(5):
        store.put((*profile_ns("12"), "archive"), f"old-{i}", {"value": "archived"})
    store.put(profile_ns("12"), "last_issue", {"value": "sync fails"})
    store.put(profile_ns("12"), "contact_preference", {"value": "email"})

    plain = [i for i in store.search(profile_ns("12"), limit=5)
             if i.namespace == profile_ns("12")]
    assert plain == []  # what a filter-after-limit would have returned

    found = relevant_memories(store, "12", "q")
    assert sorted(i.key for i in found) == ["contact_preference", "last_issue"]


@requires_postgres
def test_live_postgres_prefix_leak_and_the_fix():
    """Skipped by default. On a live PostgresStore (no index needed), the
    two-label prefix returns customer 123's row for customer 12, and
    `relevant_memories` does not. Unique ids per run, cleaned up after."""
    dsn = os.environ["ATLAS_POSTGRES_TEST_DSN"]
    run = uuid.uuid4().hex[:8]
    a, b = f"12{run}", f"12{run}3"  # b extends a's text, as 123 extends 12
    with PostgresStore.from_conn_string(dsn) as store:
        store.setup()  # test database only; production runs scripts/setup_store.py
        store.put(profile_ns(a), "k", {"value": "mine"})
        store.put(profile_ns(b), "k", {"value": "theirs"})
        try:
            old = {i.namespace[1] for i in store.search(("customer", a))}
            assert old == {a, b}
            assert [i.namespace[1] for i in relevant_memories(store, a, "q")] == [a]
        finally:
            store.delete(profile_ns(a), "k")
            store.delete(profile_ns(b), "k")


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

    item = store.get(profile_ns("cust-1"), "contact_preference")
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
    ns = profile_ns("cust-1")  # the same profile recall reads

    for value in ("prefers email", "prefers phone", "no strong preference"):
        fact = CustomerFact(
            key="contact_preference", value=value, kind="preference", source_turn=1
        )
        compact(store, "cust-1", [fact])

    results = store.search(ns)
    assert len(results) == 1
    assert results[0].value["value"] == "no strong preference"


def _fake_extraction(monkeypatch, *facts: CustomerFact) -> None:
    monkeypatch.setattr(
        memory_module.extractor,
        "invoke",
        lambda payload: {"structured_response": Extraction(facts=list(facts))},
    )


def test_reflect_extracts_then_compacts(monkeypatch):
    """`reflect` is the full pass - extract, check, compact - and this is the
    seam a background task/queue calls off the hot path. Monkeypatching
    `extractor.invoke` keeps the assertion on reflect's own wiring, not on
    a live model call."""
    _fake_extraction(
        monkeypatch,
        CustomerFact(
            key="contact_preference", value="prefers email",
            kind="preference", source_turn=0,
        ),
    )

    store = build_dev_store()
    reflect(store, "cust-1", [HumanMessage("I prefer email.")])

    item = store.get(profile_ns("cust-1"), "contact_preference")
    assert item.value["value"] == "prefers email"


def test_reflect_drops_a_fact_whose_source_turn_is_not_a_real_customer_turn(
    monkeypatch,
):
    """Gate decision 4: the hallucinated "legacy plan". A fact citing the
    assistant's message, or an index past the end of the conversation, is
    dropped before compaction; the fact citing the customer's turn is kept."""
    _fake_extraction(
        monkeypatch,
        CustomerFact(
            key="contact_preference", value="prefers email",
            kind="preference", source_turn=0,
        ),
        CustomerFact(key="plan", value="legacy plan", kind="account", source_turn=1),
        CustomerFact(key="region", value="EU", kind="account", source_turn=7),
        CustomerFact(key="tier", value="gold", kind="account", source_turn=-1),
    )
    messages = [
        HumanMessage("Please email me rather than calling."),
        AIMessage("Noted. You're on the legacy plan, so email works."),
    ]

    store = build_dev_store()
    reflect(store, "cust-1", messages)

    kept = {item.key for item in store.search(profile_ns("cust-1"))}
    assert kept == {"contact_preference"}


def test_reflected_facts_reach_the_next_thread_through_recall(monkeypatch):
    """compact writes into the profile namespace, so Chapter 13's `recall`
    loads an extracted fact on the customer's next thread unchanged."""
    from atlas.graph import recall

    _fake_extraction(
        monkeypatch,
        CustomerFact(
            key="contact_preference", value="prefers email",
            kind="preference", source_turn=0,
        ),
    )
    store = build_dev_store()
    reflect(store, "cust-1", [HumanMessage("Email me, please.")])

    state = {
        "messages": [HumanMessage("New question")],
        "ticket": {"id": "T-2", "customer_id": "cust-1"},
    }
    delta = recall(state, Runtime(store=store))

    assert delta == {"customer_profile": {"contact_preference": "prefers email"}}


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
        # LangMem gets its own namespace, filled from `configurable`:
        config = {"configurable": {"customer_id": "C-1"}}
        assert manager.namespace(config) == ("customer", "C-1", "langmem")
    finally:
        reflection.shutdown()


def test_langmem_submit_without_a_config_fails_outside_a_graph_run():
    """The listing's old call, `reflection.submit({...}, after_seconds=0)`,
    raises before any model call: there is no configurable context to read
    `customer_id` from."""
    _manager, reflection = build_langmem_pipeline(build_dev_store())
    try:
        with pytest.raises(ValueError, match="configurable context"):
            reflection.submit({"messages": []}, after_seconds=0)
    finally:
        reflection.shutdown()


def test_submit_langmem_reflection_refuses_an_unsafe_customer_id():
    """LangMem fills `{customer_id}` into its namespace verbatim, so an id
    with a LIKE wildcard is refused before submit, as profile_ns refuses it."""
    _manager, reflection = build_langmem_pipeline(build_dev_store())
    try:
        with pytest.raises(ValueError, match="unsafe customer id"):
            submit_langmem_reflection(reflection, "1_", {"messages": []})
    finally:
        reflection.shutdown()


def test_submit_langmem_reflection_carries_the_customer_into_the_namespace():
    """`submit_langmem_reflection` passes `customer_id` in `configurable`;
    inside the executor's worker that config is what LangMem's namespace
    template reads. A stand-in reflector records the namespace the real
    manager would write to and the store it would write into - no model
    call."""
    from langmem import ReflectionExecutor

    store = build_dev_store()
    manager, real_executor = build_langmem_pipeline(store)
    real_executor.shutdown()

    class RecordingReflector:
        namespace = manager.namespace  # ReflectionExecutor requires one

        def invoke(self, payload: dict) -> tuple:
            return manager.namespace(get_config()), get_store() is store

    reflection = ReflectionExecutor(RecordingReflector(), store=store)
    try:
        future = submit_langmem_reflection(
            reflection, "C-1", {"messages": [HumanMessage("Email me.")]}
        )
        assert future.result(timeout=10) == (("customer", "C-1", "langmem"), True)
    finally:
        reflection.shutdown()


@requires_postgres
@requires_openai
def test_build_prod_store_sets_up_a_real_postgres_backed_semantic_store():
    """Skipped by default - see `requires_postgres`/`requires_openai` above."""
    dsn = os.environ["ATLAS_POSTGRES_TEST_DSN"]

    from scripts.setup_store import create_store_tables

    create_store_tables(dsn)  # the deploy step, run once - not build_prod_store
    with build_prod_store(dsn) as store:
        store.put(profile_ns("cust-prod"), "plan", {"value": "enterprise"})
        item = store.get(profile_ns("cust-prod"), "plan")
        assert item.value == {"value": "enterprise"}
