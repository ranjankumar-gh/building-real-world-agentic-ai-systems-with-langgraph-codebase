"""Chapter 10, "Durable Execution, Long-Running Workflows, and State
Migration" - Atlas's first crossing of the checkpoint membrane.

See "Making the refund idempotent". A checkpointer (Chapter 9) gives
at-least-once execution: a crash between a completed side effect and the
checkpoint that records it causes a resume to re-run the step. This module
isolates the side-effecting code - the refund - so it can be made safe on
its own terms: a stable idempotency key derived from durable state
(`thread_id` + `ticket_id`, never random, never time-based) and an idempotent
operation that dedupes at the backend, atomically with the charge, the way a
real payment provider's `Idempotency-Key` header works.

`atlas/graph.py`'s `refund` node imports `idempotency_key` and
`charge_refund` from here and is the only caller.
"""


def idempotency_key(thread_id: str, ticket_id: str) -> str:
    """A stable key for one logical refund. Identical across retries and
    resumes - never random, never time-based - so repeated attempts at the
    same refund collapse to the same key."""
    return f"refund:{thread_id}:{ticket_id}"


# Seeded backend (companion repo). _LEDGER is the PROVIDER's dedup store:
# the same key returns the original result instead of charging again.
_REFUNDS = {"T-1001": {"status": "pending", "amount": 49.0}}
_LEDGER: dict[str, str] = {}


class RefundError(RuntimeError):
    """The payment backend rejected or failed the refund."""


def charge_refund(key: str, ticket_id: str) -> str:
    """Idempotent at the provider: a repeated key returns the first result
    and does not charge again."""
    if key in _LEDGER:
        return _LEDGER[key]  # already charged - return prior result
    record = _REFUNDS[ticket_id]
    record["status"] = "refunded"
    result = f"Refund of ${record['amount']:.2f} issued for {ticket_id}."
    _LEDGER[key] = result  # dedup recorded atomically with the charge
    return result
