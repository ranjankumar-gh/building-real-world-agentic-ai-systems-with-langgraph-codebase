"""Chapter 21, "Evaluation and Testing" - atlas/evals.py.

See "Building a path-coverage dataset" and "Four evaluators, four different
questions". `routing_correct`, `handoffs_within_bound`, and `tool_call_correct`
(via `agentevals`) are all deterministic and need no live model or LangSmith
connection - they are tested for real below. `answer_quality` (via
`openevals`) needs a live Anthropic call to actually score anything, and
`build_regression_dataset`/`run_regression_suite` need a live LangSmith
connection - both are skip-guarded, matching the `requires_postgres`/
`requires_openai`/`requires_anthropic`/`requires_langsmith` pattern already
used throughout this repo (see `tests/test_tracing.py`, `tests/test_memory.py`,
`tests/test_run_research.py`) for the external-service exception."""

import os

import pytest

from atlas import evals as evals_module
from atlas.evals import (
    ALL_EVALUATORS,
    REGRESSION_EXAMPLES,
    answer_quality,
    build_regression_dataset,
    handoffs_within_bound,
    routing_correct,
    run_atlas,
    run_regression_suite,
    tool_call_correct,
)

requires_langsmith = pytest.mark.skipif(
    not os.environ.get("LANGSMITH_API_KEY"),
    reason="requires a live LangSmith connection",
)
requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires a live Anthropic model call through create_llm_as_judge",
)


# --- Path coverage: the dataset shape --------------------------------------


def test_regression_examples_cover_one_example_per_known_route():
    """Path coverage, not question variety: one example per Chapter 6/11/16
    route, not many differently-worded questions on the same route."""
    routes = {ex["outputs"].get("route") for ex in REGRESSION_EXAMPLES}
    assert routes == {"answer", "retrieve", "refund", None}  # None: the research example

    targets = {ex["inputs"]["target"] for ex in REGRESSION_EXAMPLES}
    assert targets == {"resolve", "research"}


def test_only_the_refund_example_carries_a_tool_call_trajectory_reference():
    with_messages = [ex for ex in REGRESSION_EXAMPLES if "messages" in ex["outputs"]]
    assert len(with_messages) == 1
    assert with_messages[0]["outputs"]["route"] == "refund"


def test_only_the_research_example_carries_a_handoff_bound():
    with_bound = [ex for ex in REGRESSION_EXAMPLES if "max_handoffs" in ex["outputs"]]
    assert len(with_bound) == 1
    assert with_bound[0]["inputs"]["target"] == "research"


# --- Deterministic evaluators: no model, no LangSmith call -----------------


def test_routing_correct_matches_the_reference_route():
    assert routing_correct({}, {"route": "answer"}, {"route": "answer"}) is True


def test_routing_correct_flags_a_mismatched_route():
    assert routing_correct({}, {"route": "retrieve"}, {"route": "answer"}) is False


def test_routing_correct_is_a_no_op_on_research_examples_with_no_route_reference():
    assert routing_correct({}, {"handoffs": 5}, {"max_handoffs": 2}) is True


def test_handoffs_within_bound_passes_when_the_run_stays_under_the_cap():
    assert handoffs_within_bound({}, {"handoffs": 2}, {"max_handoffs": 2}) is True


def test_handoffs_within_bound_catches_the_opening_hooks_regression():
    """The one evaluator in the suite that would have caught the extra
    redundant research-handoff hop: the final answer and specialist were
    both correct, only the hop count was too high."""
    assert handoffs_within_bound({}, {"handoffs": 3}, {"max_handoffs": 2}) is False


def test_handoffs_within_bound_is_a_no_op_on_main_graph_examples():
    assert handoffs_within_bound({}, {"route": "answer"}, {"route": "answer"}) is True


def test_tool_call_correct_is_a_no_op_when_no_messages_reference_exists():
    assert tool_call_correct({"route": "answer"}, {"route": "answer"}) is True


def test_tool_call_correct_passes_when_the_expected_tools_are_called_in_order():
    """Real agentevals call - a pure structural comparison, no model call."""
    reference = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "lookup_ticket", "arguments": "{}"}},
            ]},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "set_ticket_status", "arguments": "{}"}},
            ]},
        ]
    }
    outputs = {"messages": reference["messages"]}

    result = tool_call_correct(outputs, reference)

    assert result["score"] is True


def test_tool_call_correct_fails_when_lookup_ticket_is_skipped():
    reference = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "lookup_ticket", "arguments": "{}"}},
            ]},
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "set_ticket_status", "arguments": "{}"}},
            ]},
        ]
    }
    outputs = {
        "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "set_ticket_status", "arguments": "{}"}},
            ]},
        ]
    }

    result = tool_call_correct(outputs, reference)

    assert result["score"] is False


def test_answer_quality_is_constructed_locally_and_is_callable():
    """Constructing the judge (create_llm_as_judge) is a local operation -
    only calling it needs a live Anthropic connection."""
    assert callable(answer_quality)


def test_all_evaluators_lists_all_four_evaluators_in_the_documented_order():
    assert ALL_EVALUATORS == [
        routing_correct, handoffs_within_bound, tool_call_correct, answer_quality
    ]


# --- run_atlas: dispatch, not the graphs' own correctness -------------------


def test_run_atlas_dispatches_resolve_targets_to_the_main_graph(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        evals_module.graph, "invoke", lambda inputs, config: seen.update(
            inputs=inputs, config=config
        ) or {"route": "answer"}
    )

    result = run_atlas({"message": "hi", "target": "resolve"})

    assert result == {"route": "answer"}
    assert seen["inputs"] == {"messages": [{"role": "user", "content": "hi"}]}
    assert "thread_id" in seen["config"]["configurable"]


def test_run_atlas_dispatches_research_targets_to_the_research_graph(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        evals_module.research_graph, "invoke", lambda inputs, config: seen.update(
            inputs=inputs, config=config
        ) or {"handoffs": 1}
    )

    result = run_atlas({"message": "compare SLAs", "target": "research"})

    assert result == {"handoffs": 1}
    assert seen["inputs"] == {
        "messages": [{"role": "user", "content": "compare SLAs"}]
    }


def test_run_atlas_uses_a_fresh_thread_id_on_every_call(monkeypatch):
    """Chapter 9: graph is checkpointer-backed - a fixed thread_id would let
    one example's state (or a suspended approval_gate interrupt) leak into
    the next one."""
    configs = []
    monkeypatch.setattr(
        evals_module.graph, "invoke",
        lambda inputs, config: configs.append(config) or {}
    )

    run_atlas({"message": "one", "target": "resolve"})
    run_atlas({"message": "two", "target": "resolve"})

    thread_ids = [c["configurable"]["thread_id"] for c in configs]
    assert thread_ids[0] != thread_ids[1]


# --- External-service exception: LangSmith -------------------------------


@requires_langsmith
def test_build_regression_dataset_creates_the_frozen_dataset_in_langsmith():
    """Skipped by default - see `requires_langsmith` above. Real teams run
    this once, not in CI."""
    build_regression_dataset("atlas-regression-test-run")


@requires_langsmith
@requires_anthropic
def test_run_regression_suite_gates_against_the_frozen_dataset():
    """Skipped by default - needs both a live LangSmith dataset and a live
    Anthropic connection for the answer_quality judge."""
    run_regression_suite()
