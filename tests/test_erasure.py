"""Chapter 23 - atlas/erasure.py.

See "Retention, and the erasure request the audit log cannot honour". The
interesting assertions here are the ones about what erasure does NOT do: it
must not touch another tenant, and it must not silently drop the audit record
that Chapter 1 made a success criterion.
"""

import re

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.base import _namespace_to_text

from atlas.erasure import audit_ns, customer_ns_prefix, erase_customer


def _seeded() -> InMemoryStore:
    store = InMemoryStore()
    store.put(("customer", "C-1", "profile"), "plan", {"tier": "enterprise"})
    store.put(("customer", "C-1", "memories"), "m1", {"fact": "prefers email"})
    store.put(("customer", "C-1", "budget", "2026-08"), "spent", {"tokens": 12})
    store.put(audit_ns("C-1"), "call-1", {"tool": "issue_refund", "args": {"amount": 49}})
    # a second tenant that must survive untouched
    store.put(("customer", "C-2", "profile"), "plan", {"tier": "free"})
    return store


def test_erasure_removes_every_namespace_under_the_customer():
    store = _seeded()

    report = erase_customer(store, "C-1", thread_ids=[])

    assert store.get(("customer", "C-1", "profile"), "plan") is None
    assert store.get(("customer", "C-1", "memories"), "m1") is None
    assert store.get(("customer", "C-1", "budget", "2026-08"), "spent") is None
    assert report.items_deleted == 3


def test_erasure_does_not_touch_another_tenant():
    store = _seeded()

    erase_customer(store, "C-1", thread_ids=[])

    assert store.get(("customer", "C-2", "profile"), "plan") is not None
    assert store.list_namespaces(prefix=customer_ns_prefix("C-2"))


def test_the_audit_record_is_retained_and_reported_not_silently_kept():
    """The honest half. A subject-erasure request against a system with a
    compliance audit log cannot be answered with an unqualified 'done'."""
    store = _seeded()

    report = erase_customer(store, "C-1", thread_ids=[])

    assert store.get(audit_ns("C-1"), "call-1") is not None
    assert report.retained == [(audit_ns("C-1"), "call-1")]
    assert report.complete is False


def test_erasure_can_be_told_to_take_the_audit_record_too():
    """The override exists because the legal basis for retention is the
    caller's decision, not this module's."""
    store = _seeded()

    report = erase_customer(store, "C-1", thread_ids=[], retain_audit=False)

    assert store.get(audit_ns("C-1"), "call-1") is None
    assert report.retained == []
    assert report.complete is True


def test_erasure_deletes_the_customers_checkpoints_when_given_a_checkpointer():
    store = _seeded()
    checkpointer = InMemorySaver()
    config = {"configurable": {"thread_id": "t-1", "checkpoint_ns": ""}}
    checkpointer.put(config, empty_checkpoint(), {}, {})
    assert checkpointer.get(config) is not None

    report = erase_customer(
        store, "C-1", thread_ids=["t-1"], checkpointer=checkpointer
    )

    assert checkpointer.get(config) is None
    assert report.threads_deleted == ["t-1"]


def test_erasing_an_unknown_customer_is_a_no_op_rather_than_an_error():
    store = _seeded()

    report = erase_customer(store, "C-nobody", thread_ids=[])

    assert report.items_deleted == 0
    assert report.complete is True


# --- 12 vs 123: PostgresStore matches namespace prefixes as text ----------

def _like(pattern: str, text: str) -> bool:
    rx = "".join(
        ".*" if c == "%" else "." if c == "_" else re.escape(c) for c in pattern
    )
    return re.fullmatch(rx, text, flags=re.DOTALL) is not None


class LikePrefixStore:
    """InMemoryStore with PostgresStore's prefix semantics for the three
    calls erase_customer makes: `prefix LIKE '<dotted prefix>%'`, the
    pattern the pinned PostgresStore builds for both list_namespaces
    (postgres/base.py:598-600) and search (:445-447)."""

    def __init__(self) -> None:
        self.inner = InMemoryStore()

    def put(self, namespace, key, value) -> None:
        self.inner.put(namespace, key, value)

    def get(self, namespace, key):
        return self.inner.get(namespace, key)

    def delete(self, namespace, key) -> None:
        self.inner.delete(namespace, key)

    def _matches(self, prefix, namespace) -> bool:
        return _like(f"{_namespace_to_text(prefix)}%", _namespace_to_text(namespace))

    def list_namespaces(self, *, prefix, limit=100):
        every = self.inner.list_namespaces(limit=10_000)
        return [ns for ns in every if self._matches(prefix, ns)][:limit]

    def search(self, namespace_prefix, *, limit=10, **_):
        every = self.inner.search((), limit=10_000)
        hits = [i for i in every if self._matches(namespace_prefix, i.namespace)]
        return hits[:limit]


def _twelve_and_123() -> LikePrefixStore:
    store = LikePrefixStore()
    store.put(("customer", "12", "profile"), "plan", {"tier": "pro"})
    store.put(audit_ns("12"), "call-1", {"tool": "issue_refund"})
    store.put(("customer", "123", "profile"), "plan", {"tier": "enterprise"})
    store.put(audit_ns("123"), "call-9", {"tool": "issue_refund"})
    return store


def test_the_old_prefix_loop_deletes_customer_123_when_erasing_12():
    """The pre-fix loop, verbatim, on a store with Postgres's matching:
    erasing customer 12 deletes customer 123's profile too."""
    store = _twelve_and_123()

    for namespace in store.list_namespaces(prefix=customer_ns_prefix("12")):
        for item in store.search(namespace):
            store.delete(namespace, item.key)

    assert store.get(("customer", "123", "profile"), "plan") is None  # data loss


def test_erasing_12_leaves_customer_123_untouched_on_postgres_matching():
    store = _twelve_and_123()

    report = erase_customer(store, "12", thread_ids=[], retain_audit=False)

    assert store.get(("customer", "12", "profile"), "plan") is None
    assert store.get(audit_ns("12"), "call-1") is None
    assert store.get(("customer", "123", "profile"), "plan") is not None
    assert store.get(audit_ns("123"), "call-9") is not None
    assert report.namespaces_deleted == [("customer", "12", "profile")]
    assert report.items_deleted == 2


def test_retained_audit_reports_only_this_customers_records():
    report = erase_customer(_twelve_and_123(), "12", thread_ids=[])

    assert report.retained == [(audit_ns("12"), "call-1")]


def test_erase_customer_refuses_an_unsafe_id():
    """"1_" would match customer 12's namespaces as a LIKE pattern; it is
    refused before anything is listed or deleted."""
    import pytest

    store = _twelve_and_123()

    with pytest.raises(ValueError, match="unsafe customer id"):
        erase_customer(store, "1_", thread_ids=[], retain_audit=False)
    assert store.get(("customer", "12", "profile"), "plan") is not None
