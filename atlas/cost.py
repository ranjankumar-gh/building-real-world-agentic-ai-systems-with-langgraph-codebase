"""Chapter 23, "Security, Privacy, Cost, and Governance" - a hard,
cumulative per-tenant cost ceiling.

See "A hard, cumulative cost ceiling". Chapter 12's `ContextBudget`
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
which is what "no per-tenant ceiling" actually means in practice."""

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.messages import AIMessage
from langgraph.store.base import BaseStore

MONTHLY_TOKEN_CAP = 2_000_000


def budget_ns(customer_id: str) -> tuple:
    return ("customer", customer_id, "budget")


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

    def __init__(self, store: BaseStore) -> None:
        self.store = store

    def wrap_model_call(self, request, handler):
        customer_id: str = request.runtime.context.customer_id
        ns = budget_ns(customer_id)
        item = self.store.get(ns, "monthly_tokens")
        spent = item.value["tokens"] if item else 0
        if spent >= MONTHLY_TOKEN_CAP:
            return degrade(request)  # <1>
        estimated = count_tokens_approximately(request.messages)  # <2>
        self.store.put(ns, "monthly_tokens", {"tokens": spent + estimated})
        return handler(request)


# 1. `degrade` is Atlas's own choice, not a framework default: route to a
#    cheaper model, or return a clear refusal explaining the tenant is over
#    its monthly allocation. Either beats the third option - silently
#    continuing to spend past a cap nobody is watching, which is what "no
#    per-tenant ceiling" actually means in practice.
# 2. Counted from the *request*, before the call, the same
#    approximate-and-headroom discipline Chapter 12 already established -
#    not an exact post-call usage figure this book hasn't verified an API
#    for. A cap that undercounts slightly by design is safer than one built
#    on an assumption.
