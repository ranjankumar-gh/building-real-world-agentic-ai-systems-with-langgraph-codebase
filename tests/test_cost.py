"""Chapter 23, "Security, Privacy, Cost, and Governance" - atlas/cost.py.

See "A cumulative cost ceiling". `TenantBudgetGuard.wrap_model_call`
needs no live model call to test - it only reads/writes the cumulative
token count in a `BaseStore` and decides whether to call `handler` at all,
so an `InMemoryStore` (the same dev/test default `atlas/memory.py` already
uses) plus a dummy `ModelRequest` (the `test_context.py` convention) is
enough to exercise it directly."""

import asyncio

from datetime import datetime, timezone

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from atlas.cost import (
    MONTHLY_TOKEN_CAP,
    TenantBudgetGuard,
    budget_ns,
    current_period,
    degrade,
)
from atlas.containment import is_revoked
from atlas.security import AtlasContext


def _request(customer_id: str, messages: list) -> ModelRequest:
    return ModelRequest(
        model=object(),  # stand-in for BaseChatModel; never invoked once over cap
        messages=messages,
        system_message=SystemMessage("You are Atlas."),
        runtime=Runtime(context=AtlasContext(role="support_agent", customer_id=customer_id)),
    )


def test_budget_ns_scopes_by_customer_and_by_period():
    assert budget_ns("C-1", "2026-08") == ("customer", "C-1", "budget", "2026-08")
    assert budget_ns("C-1", "2026-08") != budget_ns("C-2", "2026-08")


def test_budget_ns_rolls_so_the_cap_is_monthly_not_lifetime():
    """The defect this guards against: with no period in the namespace,
    MONTHLY_TOKEN_CAP is a lifetime cap, and the first tenant to reach it is
    refused service permanently with no reset path."""
    assert budget_ns("C-1", "2026-08") != budget_ns("C-1", "2026-09")


def test_current_period_formats_the_billing_window_as_year_month():
    assert current_period(datetime(2026, 8, 5, tzinfo=timezone.utc)) == "2026-08"


def test_a_tenant_capped_last_month_starts_the_new_month_clean():
    """The counter reads zero on the first of the month because the new
    period's namespace is empty - not because a reset job remembered to run."""
    store = InMemoryStore()
    store.put(budget_ns("C-1", "2026-08"), "spent", {"tokens": MONTHLY_TOKEN_CAP})

    assert store.get(budget_ns("C-1", "2026-09"), "spent") is None


def test_degrade_returns_a_model_response_without_calling_the_model():
    result = degrade(_request("C-1", [HumanMessage("hi")]))

    assert isinstance(result, ModelResponse)
    assert isinstance(result.result[0], AIMessage)
    assert "monthly usage allocation" in result.result[0].content


def test_guard_lets_a_fresh_tenant_through_and_records_spend():
    store = InMemoryStore()
    guard = TenantBudgetGuard(store)
    request = _request("C-1", [HumanMessage("what is the refund window")])

    result = guard.wrap_model_call(request, lambda r: "handled")

    assert result == "handled"
    item = store.get(budget_ns("C-1", current_period()), "spent")
    assert item.value["tokens"] > 0


def test_guard_accumulates_spend_across_multiple_calls():
    store = InMemoryStore()
    guard = TenantBudgetGuard(store)

    for _ in range(3):
        guard.wrap_model_call(
            _request("C-1", [HumanMessage("what is the refund window")]), lambda r: "handled"
        )

    item = store.get(budget_ns("C-1", current_period()), "spent")
    assert item.value["tokens"] > 0
    # three calls costs strictly more than one
    single_store = InMemoryStore()
    TenantBudgetGuard(single_store).wrap_model_call(
        _request("C-1", [HumanMessage("what is the refund window")]), lambda r: "handled"
    )
    single = single_store.get(budget_ns("C-1", current_period()), "spent").value["tokens"]
    assert item.value["tokens"] == single * 3


def test_guard_degrades_once_the_cumulative_cap_is_hit_without_calling_the_model():
    store = InMemoryStore()
    store.put(budget_ns("C-1", current_period()), "spent", {"tokens": MONTHLY_TOKEN_CAP})
    guard = TenantBudgetGuard(store)
    called = []

    def handler(_request):
        called.append(True)
        return "should not run"

    result = guard.wrap_model_call(
        _request("C-1", [HumanMessage("one more question")]), handler
    )

    assert isinstance(result, ModelResponse)
    assert called == []  # the model was never called once over the cap


def test_guard_keeps_tenants_isolated_in_separate_namespaces():
    store = InMemoryStore()
    guard = TenantBudgetGuard(store)
    store.put(budget_ns("C-over", current_period()), "spent", {"tokens": MONTHLY_TOKEN_CAP})

    # C-fresh has never spent anything, so it must sail through even though
    # C-over is capped in the same store.
    result = guard.wrap_model_call(
        _request("C-fresh", [HumanMessage("hello")]), lambda r: "handled"
    )
    assert result == "handled"


def test_the_cap_only_degrades_unless_revocation_is_asked_for():
    """The default stays a soft refusal, because the chapter's printed
    listing argues one. Revocation is opt-in."""
    store = InMemoryStore()
    guard = TenantBudgetGuard(store)
    store.put(
        budget_ns("customer-42", current_period()),
        "spent",
        {"tokens": MONTHLY_TOKEN_CAP + 1},
    )

    guard.wrap_model_call(
        _request("customer-42", [HumanMessage("hello")]), handler=lambda r: None
    )

    assert is_revoked(store, "customer-42") is False


def test_breaching_the_cap_revokes_rather_than_only_degrading():
    """Chapter 23: a ceiling that refuses each call is not a switch that
    stops the run. Past the cap, take the authority away."""
    store = InMemoryStore()
    guard = TenantBudgetGuard(store, revoke_on_breach=True)
    store.put(
        budget_ns("customer-42", current_period()),
        "spent",
        {"tokens": MONTHLY_TOKEN_CAP + 1},
    )

    guard.wrap_model_call(
        _request("customer-42", [HumanMessage("hello")]), handler=lambda r: None
    )

    assert is_revoked(store, "customer-42") is True


def test_the_breach_records_the_cap_as_the_reason():
    """A revocation with no reason is an outage nobody can explain. The
    reason string is what an operator reads first."""
    store = InMemoryStore()
    guard = TenantBudgetGuard(store, revoke_on_breach=True)
    store.put(
        budget_ns("customer-42", current_period()),
        "spent",
        {"tokens": MONTHLY_TOKEN_CAP + 1},
    )

    guard.wrap_model_call(
        _request("customer-42", [HumanMessage("hello")]), handler=lambda r: None
    )

    item = store.get(("containment", "customer-42"), "revocation")
    assert "cap" in item.value["reason"]


# --- The async twin and the graph's store ----------------------------------


def _request_on(store: InMemoryStore, customer_id: str, messages: list) -> ModelRequest:
    request = _request(customer_id, messages)
    return request.override(
        runtime=Runtime(context=request.runtime.context, store=store)
    )


def test_the_async_twin_charges_the_graphs_store_the_same_amount():
    sync_store, async_store = InMemoryStore(), InMemoryStore()
    messages = [HumanMessage("what is the refund window")]

    async def handler(_request):
        return "handled"

    TenantBudgetGuard().wrap_model_call(
        _request_on(sync_store, "C-1", messages), lambda r: "handled"
    )
    result = asyncio.run(
        TenantBudgetGuard().awrap_model_call(
            _request_on(async_store, "C-1", messages), handler
        )
    )

    ns = budget_ns("C-1", current_period())
    assert result == "handled"
    assert async_store.get(ns, "spent").value == sync_store.get(ns, "spent").value


def test_the_async_twin_degrades_and_revokes_over_the_cap():
    store = InMemoryStore()
    ns = budget_ns("C-1", current_period())
    store.put(ns, "spent", {"tokens": MONTHLY_TOKEN_CAP})

    async def handler(_request):
        raise AssertionError("the model must not be called over the cap")

    result = asyncio.run(
        TenantBudgetGuard(revoke_on_breach=True).awrap_model_call(
            _request_on(store, "C-1", [HumanMessage("hi")]), handler
        )
    )

    assert isinstance(result, ModelResponse)
    assert is_revoked(store, "C-1") is True
