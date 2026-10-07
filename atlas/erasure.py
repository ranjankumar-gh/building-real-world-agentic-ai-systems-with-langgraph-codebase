"""Chapter 23, "Security, Privacy, Cost, and Governance" - the deletion path.

See "Retention, and the erasure request the audit log cannot honour".
Everything Atlas persists it persists forever: checkpoints accumulate per
superstep per thread (Ch9), the store holds customer profiles and learned
facts (Ch13-14), and Ch23's audit log deliberately keeps raw tool arguments
so a refund stays traceable to an approval. None of it had a delete path.

`erase_customer` is that path, and it is deliberately partial. It removes
what Atlas is free to remove and reports what it kept, because the honest
answer to a subject-erasure request on a system with a compliance audit log
is not "done" - it is "here is what was deleted, here is what was retained,
and here is the basis for retaining it." A function that silently deleted the
audit record would trade one compliance problem for a worse one.

What legal basis lets you retain an audit record is a question for the people
who own that decision at your company, not for this book. What this module
insists on is that the retention be deliberate and visible rather than an
accident of nobody having written a deletion path.

Every namespace and item is checked against the exact customer before it is
deleted or reported. PostgresStore matches a namespace prefix as SQL text:
`list_namespaces(prefix=("customer", "12"))` becomes `prefix LIKE
'customer.12%'`, which also lists customer 123's namespaces, and a search on
`("audit", "12")` also returns `("audit", "123")` items. Without the checks,
erasing customer 12 would delete customer 123's data (Chapter 13 closed the
same leak for search). InMemoryStore matches whole labels and hides it; see
tests/test_erasure.py's 12-vs-123 tests.
"""

from dataclasses import dataclass, field

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore, Item

from atlas.memory import SAFE_ID

# `list_namespaces` and `search` both paginate, and both default to a page
# far smaller than a real tenant's data. An erasure that quietly stops at the
# first page is worse than one that fails loudly.
_PAGE = 1000


@dataclass
class ErasureReport:
    """What an erasure actually did, and what it deliberately did not."""

    customer_id: str
    namespaces_deleted: list[tuple[str, ...]] = field(default_factory=list)
    items_deleted: int = 0
    threads_deleted: list[str] = field(default_factory=list)
    retained: list[tuple[tuple[str, ...], str]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        """True only when nothing was retained.

        Deliberately not the same as "the function ran". A caller answering a
        regulator needs to know the difference between erased and mostly
        erased, and the default answer here is `False`.
        """
        return not self.retained


def customer_ns_prefix(customer_id: str) -> tuple[str, ...]:
    return ("customer", customer_id)


def is_customers(namespace: tuple[str, ...], customer_id: str) -> bool:
    """The exact customer segment, not a text prefix of it."""
    return len(namespace) > 1 and namespace[1] == customer_id


def audit_ns(customer_id: str) -> tuple[str, ...]:
    """Ch23's audit namespace. Named here so the erasure path can exclude it
    on purpose rather than miss it by accident."""
    return ("audit", customer_id)


def erase_customer(
    store: BaseStore,
    customer_id: str,
    thread_ids: list[str],
    checkpointer: BaseCheckpointSaver | None = None,
    retain_audit: bool = True,
) -> ErasureReport:
    """Delete a customer's durable data, and report what was retained.

    `thread_ids` is passed in rather than discovered: a checkpointer indexes
    by thread, not by customer, so the mapping from customer to threads lives
    in whatever system issued the thread ids. Pretending otherwise would hide
    the one piece of work a real deployment has to do for itself.
    """
    if not SAFE_ID.fullmatch(customer_id):   # a "%" or "_" would widen the match
        raise ValueError(f"unsafe customer id: {customer_id!r}")
    report = ErasureReport(customer_id=customer_id)

    for namespace in store.list_namespaces(
        prefix=customer_ns_prefix(customer_id), limit=_PAGE
    ):
        if not is_customers(namespace, customer_id):
            continue  # "customer.123..." also matches the prefix of "12"
        for item in store.search(namespace, limit=_PAGE):
            if item.namespace != namespace:
                continue  # listed on its own; delete each item once
            store.delete(namespace, item.key)
            report.items_deleted += 1
        report.namespaces_deleted.append(namespace)

    if checkpointer is not None:
        for thread_id in thread_ids:
            checkpointer.delete_thread(thread_id)
            report.threads_deleted.append(thread_id)

    if retain_audit:
        # NOT deleted, and said out loud. The audit log is the record that a
        # refund was approved by a named human - the artifact Ch1 made a
        # success criterion and Ch23 built precisely so it would outlive
        # trace retention.
        for item in _own_audit(store, customer_id):
            report.retained.append((audit_ns(customer_id), item.key))
    else:
        for item in _own_audit(store, customer_id):
            store.delete(audit_ns(customer_id), item.key)
            report.items_deleted += 1

    return report


def _own_audit(store: BaseStore, customer_id: str) -> list[Item]:
    """This customer's audit items only: ("audit", "12") is also a text
    prefix of ("audit", "123")."""
    ns = audit_ns(customer_id)
    return [i for i in store.search(ns, limit=_PAGE) if i.namespace == ns]
