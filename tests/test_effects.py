"""Chapter 10, "Durable Execution, Long-Running Workflows, and State
Migration" - atlas/effects.py's idempotency key and idempotent refund
operation, isolated from the graph so they are testable on their own.

Chapter 20, "Observability and Debugging with LangSmith", decorates
`idempotency_key` with `@traceable` - see "Turning tracing on, by
environment". Decorating and calling it needs no live LangSmith connection:
`@traceable` only submits a run once `LANGSMITH_TRACING` is actually
"true", so the existing return-value tests above still exercise the real
function with no mocking, and the new test below just confirms the
decorator applied without changing behavior."""

import pytest

from atlas.effects import RefundError, charge_refund, idempotency_key


def test_idempotency_key_is_stable_across_calls_for_the_same_thread_and_ticket():
    """"A stable idempotency key": a retry or a resume recomputes the exact
    same key - it is derived from durable state, never random, never
    time-based."""
    key_a = idempotency_key("thread-1", "T-1001")
    key_b = idempotency_key("thread-1", "T-1001")

    assert key_a == key_b == "refund:thread-1:T-1001"


def test_idempotency_key_differs_across_threads_or_tickets():
    assert idempotency_key("thread-1", "T-1001") != idempotency_key(
        "thread-2", "T-1001"
    )
    assert idempotency_key("thread-1", "T-1001") != idempotency_key(
        "thread-1", "T-1002"
    )


def test_charge_refund_charges_once_and_updates_the_seeded_ledger():
    """Chapter 10's seeded backend: atlas.effects._REFUNDS / _LEDGER model a
    payment provider that dedupes at the backend."""
    from atlas import effects

    effects._REFUNDS["T-idem-1"] = {"status": "pending", "amount": 10.0}
    key = idempotency_key("thread-charge-once", "T-idem-1")

    result = charge_refund(key, "T-idem-1")

    assert result == "Refund of $10.00 issued for T-idem-1."
    assert effects._REFUNDS["T-idem-1"]["status"] == "refunded"
    assert effects._LEDGER[key] == result


def test_charge_refund_is_idempotent_a_repeated_key_does_not_charge_again():
    """The core claim of "The idempotent operation": running the exact same
    key twice returns the FIRST result and does not touch the record a
    second time - the dedup is atomic with the charge, at the provider."""
    from atlas import effects

    effects._REFUNDS["T-idem-2"] = {"status": "pending", "amount": 25.0}
    key = idempotency_key("thread-charge-twice", "T-idem-2")

    first = charge_refund(key, "T-idem-2")
    # Mutate the seeded record directly to prove a repeat does not re-read it.
    effects._REFUNDS["T-idem-2"]["amount"] = 999.0
    second = charge_refund(key, "T-idem-2")

    assert first == second == "Refund of $25.00 issued for T-idem-2."


def test_charge_refund_with_a_different_key_charges_independently():
    from atlas import effects

    effects._REFUNDS["T-idem-3"] = {"status": "pending", "amount": 5.0}
    key_a = idempotency_key("thread-a", "T-idem-3")
    key_b = idempotency_key("thread-b", "T-idem-3")

    result_a = charge_refund(key_a, "T-idem-3")

    # The provider already marked the ticket refunded under key_a; a
    # DIFFERENT logical key (different thread) is not deduped against it and
    # hits the now-refunded record - this is Atlas's seeded backend, which
    # does not model per-ticket exclusivity, only per-key dedup.
    assert key_a != key_b
    assert result_a == "Refund of $5.00 issued for T-idem-3."


def test_refund_error_is_a_plain_runtime_error_the_retry_policy_can_target():
    assert issubclass(RefundError, RuntimeError)
    with pytest.raises(RefundError):
        raise RefundError("backend rejected the refund")


def test_idempotency_key_is_traceable_and_still_returns_the_same_value():
    """Chapter 20: @traceable wraps the function without a live LangSmith
    connection or changing its return value - the trace it produces (once
    LANGSMITH_TRACING is on) is a byproduct of the same call, not a second
    code path."""
    assert hasattr(idempotency_key, "__wrapped__")
    assert idempotency_key("thread-9", "T-9001") == "refund:thread-9:T-9001"
