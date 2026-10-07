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
`charge_refund` from here and is the only caller. It passes the amount from
state - the approved amount, after any edit at Chapter 11's approval gate -
and the backend refuses an amount above the original charge with
`RefundRefused`. That refusal is deterministic - asking again gets the same
answer - so it is a `ValueError`, outside the refund node's
`retry_on=(RefundError,)`: `refund_failed` runs after one attempt, with no
retries.

Chapter 20, "Observability and Debugging with LangSmith", adds `@traceable`
to `idempotency_key` - see "Turning tracing on, by environment". A plain
Python function called from inside an already-traced node is otherwise
invisible in a trace: `create_agent`/`StateGraph` runs self-instrument once
tracing is on, but a helper function they call is not itself a LangChain
runnable, so it disappears into the parent span unless `@traceable` gives
it one of its own. Decorating it needs no live LangSmith connection -
`@traceable` only submits a run when `LANGSMITH_TRACING` is actually
"true"; see `tests/test_effects.py`.
"""

from langsmith import traceable


@traceable(run_type="tool", name="idempotency_key")
def idempotency_key(thread_id: str, ticket_id: str) -> str:
    """A stable key for one logical refund. Identical across retries and
    resumes - never random, never time-based - so repeated attempts at the
    same refund collapse to the same key. Now visible in a trace (Chapter
    20) - without @traceable it ran invisibly inside whatever node called
    it."""
    return f"refund:{thread_id}:{ticket_id}"


# Seeded backend (companion repo). _LEDGER is the PROVIDER's dedup store:
# the same key returns the original result instead of charging again.
_REFUNDS = {"T-1001": {"status": "pending", "amount": 49.0}}
_LEDGER: dict[str, str] = {}


class RefundError(RuntimeError):
    """The payment backend rejected or failed the refund."""


class RefundRefused(ValueError):
    """The amount can never succeed, so retrying it is wasted work."""


def charge_refund(key: str, ticket_id: str, amount: float) -> str:
    """Idempotent at the provider: a repeated key returns the first result
    and does not charge again. Refuses more than the original charge."""
    if key in _LEDGER:
        return _LEDGER[key]  # already charged - return prior result
    record = _REFUNDS[ticket_id]
    if amount > record["amount"]:  # the provider's cap: never more than paid
        raise RefundRefused(f"{amount:.2f} exceeds the original {record['amount']:.2f}")
    record["status"] = "refunded"
    result = f"Refund of ${amount:.2f} issued for {ticket_id}."
    _LEDGER[key] = result  # dedup recorded atomically with the charge
    return result
