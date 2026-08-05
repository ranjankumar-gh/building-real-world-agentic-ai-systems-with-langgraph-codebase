"""Chapter 23 - atlas/erasure.py.

See "Retention, and the erasure request the audit log cannot honour". The
interesting assertions here are the ones about what erasure does NOT do: it
must not touch another tenant, and it must not silently drop the audit record
that Chapter 1 made a success criterion.
"""

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

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
