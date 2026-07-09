"""Chapter 1: atlas/breaks.py - the same three failure modes, reproduced deterministically."""

import pytest
from langchain_core.messages import AIMessage

from atlas import breaks, naive


@pytest.fixture(autouse=True)
def restore_naive_state():
    """atlas.naive holds mutable module-level state (TOOLS_BY_NAME, refund_calls,
    _REFUNDS) that breaks.py deliberately mutates to demonstrate each failure -
    snapshot and restore it so these tests don't bleed into each other or into
    atlas.naive's own tests."""
    original_tools = dict(naive.TOOLS_BY_NAME)
    original_calls = list(naive.refund_calls)
    original_refunds = {k: dict(v) for k, v in naive._REFUNDS.items()}
    yield
    naive.TOOLS_BY_NAME.clear()
    naive.TOOLS_BY_NAME.update(original_tools)
    naive.refund_calls.clear()
    naive.refund_calls.extend(original_calls)
    naive._REFUNDS.clear()
    naive._REFUNDS.update(original_refunds)


def test_stuck_model_never_terminates_within_the_watchdog_limit():
    stuck = breaks.ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "issue_refund", "args": {"ticket_id": "T-1001"},
             "id": "call_1", "type": "tool_call"},
        ]),
    ])

    result = breaks.run_with_watchdog("refund my order", stuck, limit=5)

    assert result == "NEVER TERMINATED: still looping at step 5"


def test_second_turn_has_no_memory_of_the_first_because_state_is_local():
    two_turns = breaks.ScriptedModel([
        AIMessage(content="Sure - what is your order number?"),
        AIMessage(content="I don't have any record of a previous request. "
                          "Could you tell me what you need?"),
    ])

    first_turn = naive.run("I'd like a refund.", two_turns)
    # ... imagine a deploy restarts the process here ...
    second_turn = naive.run("It's T-1001.", two_turns)

    assert first_turn == "Sure - what is your order number?"
    assert "any record of a previous request" in second_turn


def test_swallowed_tool_error_produces_a_confident_wrong_answer():
    naive.refund_calls.clear()
    naive.TOOLS_BY_NAME["issue_refund"] = breaks.issue_refund_broken

    confirm = breaks.ScriptedModel([
        AIMessage(content="", tool_calls=[
            {"name": "issue_refund", "args": {"ticket_id": "T-1001"},
             "id": "c1", "type": "tool_call"}]),
        AIMessage(content="All set - your refund has been issued. Anything else?"),
    ])

    result = breaks.run_swallowing("Refund T-1001 please", confirm)

    assert result == "All set - your refund has been issued. Anything else?"
    assert "T-1001" not in naive.refund_calls
