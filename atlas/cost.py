"""Chapter 23, "Security, Privacy, Cost, and Governance" - a cumulative
per-tenant cost ceiling.

See "A cumulative cost ceiling". Chapter 12's `ContextBudget`
(`atlas/context.py`) bounds what a *single* model call sees - a per-turn
allocation, enforced on the hot path. It says nothing about the tenth call,
or the thousandth, from the same customer in the same month. A cumulative
cap needs to persist across calls, which means it belongs in the same
`BaseStore` Chapter 13 already wired up (`atlas/memory.py`'s `profile_ns`
convention), under its own namespace: `budget_ns` below.

`degrade` is Atlas's own choice, not a framework default: this module ships
the simplest safe one - a clear refusal explaining the tenant is over its
monthly allocation, returned as a real `ModelResponse` so `wrap_model_call`
never has to call the underlying model once the cap is hit. A production
deployment might instead retry against a cheaper model. Either beats the
third option - silently continuing to spend past a cap nobody is watching,
which is what "no per-tenant ceiling" actually means in practice.

The counter lives in the store the graph was compiled with (`graph_store`),
so an operator reading or resetting it sees the same rows the guard
writes."""

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.store.base import BaseStore, Item

from atlas.containment import arevoke, revoke
from atlas.security import graph_store

MONTHLY_TOKEN_CAP = 2_000_000


def current_period(now: datetime | None = None) -> str:
    """The billing window this call belongs to, as `YYYY-MM`.

    Injectable so a test can roll the clock without waiting for a month."""
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


def budget_ns(customer_id: str, period: str) -> tuple:
    """One namespace per tenant per period.

    The period is part of the namespace, so the counter starts at zero on
    the first of the month because nothing is there yet - not because a
    reset job remembered to run. Without it, MONTHLY_TOKEN_CAP is a lifetime
    cap that permanently degrades the first tenant to reach it."""
    return ("customer", customer_id, "budget", period)


def degrade(request: ModelRequest) -> ModelResponse:
    """The over-cap response: a clear refusal, not a silent overspend. See
    the module docstring - swap this for a cheaper-model retry if that
    fits your product's degradation policy better."""
    return ModelResponse(
        result=[
            AIMessage(
                "This account has reached its monthly usage allocation. "
                "Contact support to raise the limit before continuing."
            )
        ]
    )


class TenantBudgetGuard(AgentMiddleware):
    """A CUMULATIVE, per-tenant ceiling - Ch12's budget bounds one call;
    this bounds the sum of all of them, this month."""

    # SOFT ceiling, deliberately. The get/put below is a read-modify-write and
    # `BaseStore` has no compare-and-swap (Ch13's own warning), so concurrent
    # calls for one tenant lose increments silently - and that is exactly the
    # runaway-spend case the cap exists for. Good enough as a cost guardrail;
    # not good enough for a cap you have to defend to a customer. For that,
    # put the counter somewhere increments are atomic (Redis INCR, or a row
    # updated in a transaction). See Ch23's callout.

    def __init__(
        self, store: BaseStore | None = None, revoke_on_breach: bool = False
    ) -> None:
        self.store = store  # only for an agent run with no store of its own
        # Off by default, deliberately. The chapter's own listing above
        # argues a soft ceiling and a graceful refusal, and a printed
        # listing that quietly revoked would contradict the paragraph
        # explaining it. A deployment that wants the harder behavior asks
        # for it - see "Stopping a run that is already doing damage".
        self.revoke_on_breach = revoke_on_breach

    def _spend(self, request: ModelRequest, item: Item | None) -> tuple[bool, int]:
        """(over the cap?, the new running total) for this call."""
        spent = item.value["tokens"] if item else 0
        if spent >= MONTHLY_TOKEN_CAP:
            return True, spent
        return False, spent + count_tokens_approximately(request.messages)  # <3>

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        customer_id: str = request.runtime.context.customer_id
        store = graph_store(request.runtime, self.store)
        ns = budget_ns(customer_id, current_period())
        over, total = self._spend(request, store.get(ns, "spent"))
        if over:
            if self.revoke_on_breach:  # <1>
                revoke(store, customer_id, reason="monthly token cap exceeded")
            return degrade(request)  # <2>
        store.put(ns, "spent", {"tokens": total})
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        customer_id: str = request.runtime.context.customer_id
        store = graph_store(request.runtime, self.store)
        ns = budget_ns(customer_id, current_period())
        over, total = self._spend(request, await store.aget(ns, "spent"))
        if over:
            if self.revoke_on_breach:
                await arevoke(store, customer_id, reason="monthly token cap exceeded")
            return degrade(request)
        await store.aput(ns, "spent", {"tokens": total})
        return await handler(request)


# 1. Climbing a rung: refusing each call one at a time leaves the run
#    alive and trying. Revoking moves the tenant out of reach of every
#    later call, in every thread, until a human puts them back. The two
#    are not alternatives - the refusal below still happens, so this turn
#    ends the same graceful way whether or not the authority was taken.
# 2. `degrade` is Atlas's own choice, not a framework default: route to a
#    cheaper model, or return a clear refusal explaining the tenant is over
#    its monthly allocation. Either beats the third option - silently
#    continuing to spend past a cap nobody is watching, which is what "no
#    per-tenant ceiling" actually means in practice.
# 3. Counted from the *request*, before the call: an estimate of the input
#    messages only, with the same approximate-and-headroom discipline
#    Chapter 12 established. It leaves out the system message and every
#    output token. The response carries exact usage in
#    `AIMessage.usage_metadata`; adding it after `handler` returns is the
#    natural next step (the chapter's Exercise 3).
