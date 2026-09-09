"""Chapter 23: the containment ladder's second rung - revoking authority
that has already been granted."""

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from atlas.containment import RevocationGate, is_revoked, revoke
from atlas.security import AtlasContext


def _request(customer_id: str, messages: list) -> ModelRequest:
    return ModelRequest(
        model=object(),  # stand-in for BaseChatModel; never invoked once revoked
        messages=messages,
        system_message=SystemMessage("You are Atlas."),
        runtime=Runtime(
            context=AtlasContext(role="support_agent", customer_id=customer_id)
        ),
    )


def test_a_fresh_subject_is_not_revoked() -> None:
    store = InMemoryStore()
    assert is_revoked(store, "customer-42") is False


def test_revoke_is_visible_to_every_later_check() -> None:
    store = InMemoryStore()
    revoke(store, "customer-42", reason="cumulative spend over cap")
    assert is_revoked(store, "customer-42") is True


def test_revoke_records_the_reason_and_the_time() -> None:
    store = InMemoryStore()
    revoke(store, "customer-42", reason="operator halt")
    item = store.get(("containment", "customer-42"), "revocation")
    assert item.value["reason"] == "operator halt"
    assert item.value["revoked_at"].endswith("+00:00")


def test_the_gate_refuses_every_call_after_revocation() -> None:
    store = InMemoryStore()
    gate = RevocationGate(store)
    request = _request("customer-42", [HumanMessage("hello")])

    revoke(store, "customer-42", reason="operator halt")

    with pytest.raises(RuntimeError, match="revoked"):
        gate.wrap_model_call(request, handler=lambda r: "should not reach here")


def test_the_gate_lets_an_unrevoked_subject_through() -> None:
    store = InMemoryStore()
    gate = RevocationGate(store)
    request = _request("customer-42", [HumanMessage("hello")])

    result = gate.wrap_model_call(request, handler=lambda r: "handled")

    assert result == "handled"


def test_revocation_is_scoped_to_its_own_subject() -> None:
    store = InMemoryStore()
    revoke(store, "customer-over", reason="operator halt")

    assert is_revoked(store, "customer-fresh") is False
