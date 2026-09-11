"""Chapter 4, "A first look at retries" - what `retry_on` costs you when you
name it.

The book's first retry listing writes `retry_on=(ConnectionError,)`, which
reads like a helpful specialisation and is actually a narrowing. These tests
pin what the default predicate does, so the difference is a fact rather than
an opinion. No network, no API key: the exceptions are constructed directly.
"""

import httpx
import pytest
from langgraph.types import RetryPolicy, default_retry_on

from atlas.graph import _make_builder


def _rate_limit_error() -> Exception:
    """A real provider 429, built without touching the network."""
    import anthropic

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.RateLimitError(
        message="rate limited",
        response=httpx.Response(429, request=request),
        body=None,
    )


def test_a_provider_rate_limit_is_not_a_connection_error() -> None:
    """The whole reason the narrowing bites. A 429 arrives as
    anthropic.RateLimitError, which shares no ancestry with ConnectionError,
    so `retry_on=(ConnectionError,)` silently excludes the single most common
    transient failure in production."""
    error = _rate_limit_error()

    assert isinstance(error, Exception)
    assert not isinstance(error, ConnectionError)
    assert error.status_code == 429


def test_the_default_predicate_retries_a_rate_limit() -> None:
    """Omit retry_on and LangGraph uses default_retry_on, which does retry
    it. Naming retry_on=(ConnectionError,) throws this away."""
    assert default_retry_on(_rate_limit_error()) is True


def test_the_narrowed_policy_would_not_have() -> None:
    """The same error against the book's original narrowing, stated as an
    assertion rather than left implicit."""
    narrowed = (ConnectionError,)

    assert not isinstance(_rate_limit_error(), narrowed)


@pytest.mark.parametrize(
    "exc",
    [ValueError("bad arg"), TypeError("bad type"), KeyError("missing"),
     ImportError("no module"), RuntimeError("logic")],
)
def test_the_default_predicate_declines_to_retry_programming_errors(
    exc: Exception,
) -> None:
    """The default is not "retry everything". It refuses the error classes a
    retry cannot fix, which is why omitting retry_on is a considered choice
    and not laziness."""
    assert default_retry_on(exc) is False


@pytest.mark.parametrize("status,expected", [(500, True), (503, True), (404, False)])
def test_the_default_predicate_retries_server_errors_but_not_client_errors(
    status: int, expected: bool
) -> None:
    request = httpx.Request("GET", "https://example.invalid/")
    error = httpx.HTTPStatusError(
        "boom", request=request, response=httpx.Response(status, request=request)
    )

    assert default_retry_on(error) is expected


def test_triage_carries_a_retry_policy_that_keeps_the_default_predicate() -> None:
    """Ch4 says a retry_policy is safe for a read, and triage is a read: it
    makes the first live model call in the graph and changes nothing. It
    previously carried no policy at all.

    The assertion is deliberately about retry_on being the DEFAULT, not
    merely about a policy existing. A policy narrowed to ConnectionError
    would satisfy "has a retry_policy" and still drop every 429."""
    builder = _make_builder(lambda state: {"route": "answer"})

    policy = builder.nodes["triage"].retry_policy
    assert policy is not None, "triage should carry a retry policy"

    # RetryPolicy is a NamedTuple, so it IS a tuple - check for the policy
    # type before treating a sequence as a list of policies, or you index
    # into the policy and compare against its initial_interval.
    first = policy if isinstance(policy, RetryPolicy) else policy[0]
    assert isinstance(first, RetryPolicy)
    assert first.max_attempts == 3
    assert first.retry_on is default_retry_on
    assert default_retry_on(_rate_limit_error()) is True


# --- Chapter 17, "Bound the fan-out": max_concurrency bounds one invocation,
# --- so the ceiling that survives a fleet goes on the model instead.


def test_a_rate_limiter_wires_through_init_chat_model() -> None:
    """The claim Ch17 makes, asserted rather than assumed: the limiter
    reaches the model instance, so every call through it is throttled
    whichever graph or run makes the call.

    No network and no key are used - constructing the model does not call
    the provider."""
    from langchain.chat_models import init_chat_model
    from langchain_core.rate_limiters import InMemoryRateLimiter

    limiter = InMemoryRateLimiter(requests_per_second=5)
    model = init_chat_model(
        "claude-sonnet-4-6", rate_limiter=limiter, api_key="unused-offline"
    )

    assert model.rate_limiter is limiter


def test_the_rate_limiter_actually_throttles() -> None:
    """A limiter that wires through and does nothing would pass the test
    above. This one measures: three acquisitions against a five-per-second
    bucket of one cannot complete instantly."""
    import time

    from langchain_core.rate_limiters import InMemoryRateLimiter

    limiter = InMemoryRateLimiter(
        requests_per_second=5, check_every_n_seconds=0.01, max_bucket_size=1
    )

    start = time.monotonic()
    for _ in range(3):
        limiter.acquire(blocking=True)
    elapsed = time.monotonic() - start

    # Two waits of roughly 0.2s after the first token; allow slack for a
    # loaded machine, but anything near zero means it is not limiting.
    assert elapsed > 0.25
