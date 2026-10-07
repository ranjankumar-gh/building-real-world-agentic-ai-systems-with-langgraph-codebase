"""Chapter 21, "Evaluation and Testing" - atlas/evals.py.

See "Building a path-coverage dataset" and "Four evaluators, four different
questions". `routing_correct`, `handoffs_within_bound`, and `tool_call_correct`
(via `agentevals`) are deterministic and need no live model or LangSmith
connection - they are tested for real below. `run_atlas` is driven through
the real graphs it dispatches to - Chapter 16's `supervisor_graph` and
Chapter 17's resolved graph - with scripted models in place of the live
ones, so the path each example covers actually runs: the handoff counter
counts, the resolve agent's write pauses and resumes, the refund pauses at
`approval_gate` and resumes. `answer_quality`'s two prompts are exercised
with a scripted judge model; calling the real judge needs a live Anthropic
connection, and `build_regression_dataset`/`run_regression_suite` need a
live LangSmith one - both are skip-guarded, matching the `requires_*`
pattern used throughout this repo.

Chapter 27, "Capstone", adds `checkins_sent_only_if_approved` and
`run_sla_watch` - both deterministic, no live connection needed, exercised
for real below via a genuine suspend/resume cycle through
`atlas.sla_watch.build_sla_watch_graph()`."""

import copy
import os
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langsmith.evaluation import EvaluationResult
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

import atlas.agent as agent_module
import atlas.effects as effects_module
import atlas.tools as tools_module
from atlas import evals as evals_module
from atlas import graph as graph_module
from atlas import monitor as monitor_module
from atlas import research as research_module
from atlas.evals import (
    ALL_EVALUATORS,
    ONLINE_CORRECTNESS_PROMPT,
    REGRESSION_EXAMPLES,
    answer_quality,
    build_regression_dataset,
    checkins_sent_only_if_approved,
    gate,
    handoffs_within_bound,
    routing_correct,
    run_atlas,
    run_regression_suite,
    run_sla_watch,
    tool_call_correct,
)
from atlas.research import SupervisorState, build_supervisor_graph, make_handoff
from atlas.resolve import build_resolved_graph

requires_langsmith = pytest.mark.skipif(
    not os.environ.get("LANGSMITH_API_KEY"),
    reason="requires a live LangSmith connection",
)
requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires a live Anthropic model call through create_llm_as_judge",
)


def _example(**match: Any) -> dict:
    """The one REGRESSION_EXAMPLES entry whose inputs carry these values."""
    (found,) = [
        ex for ex in REGRESSION_EXAMPLES
        if all(ex["inputs"].get(k) == v for k, v in match.items())
    ]
    return found


def _trajectory(*names: str, args: str = "{}") -> list[dict]:
    return [
        {"role": "assistant", "tool_calls": [
            {"function": {"name": name, "arguments": args}},
        ]}
        for name in names
    ]


class _ScriptedChatModel(BaseChatModel):
    """A chat model that replays a script - one reply per call - and binds
    tools by ignoring them. Stands in for the live model inside the real
    graphs, so no test below makes a model call."""

    script: list[AIMessage]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedChatModel":
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


class _ScriptedJudge(BaseChatModel):
    """A judge model: `create_llm_as_judge` asks for structured output, so
    this returns a fixed verdict and records the prompt it was shown."""

    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted-judge"

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None,
                  **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        def verdict(messages: list[dict]) -> dict:
            self.seen.append(messages)
            return {"reasoning": "scripted", "score": True}

        return RunnableLambda(verdict)


def _call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


# --- Path coverage: the dataset shape --------------------------------------


def test_regression_examples_cover_one_example_per_known_route():
    """Path coverage, not question variety: one example per Chapter 6/11/16
    route, not many differently-worded questions on the same route."""
    routes = [ex["outputs"].get("route") for ex in REGRESSION_EXAMPLES]
    assert sorted(r for r in routes if r) == ["answer", "refund", "retrieve"]

    targets = {ex["inputs"]["target"] for ex in REGRESSION_EXAMPLES}
    assert targets == {"resolve", "research", "sla_watch"}


def test_regression_examples_cover_both_sla_watch_paths():
    """Chapter 27: one example per SLA Watch path - approve sends, reject
    doesn't - the same path-coverage logic as every other route."""
    sla_examples = [ex for ex in REGRESSION_EXAMPLES if ex["inputs"]["target"] == "sla_watch"]
    decisions = {ex["inputs"]["decision"] for ex in sla_examples}
    assert decisions == {"approve", "reject"}
    sent = {ex["inputs"]["decision"]: ex["outputs"]["sla_watch_sent"] for ex in sla_examples}
    assert sent["approve"] == ["T-2001"]
    assert sent["reject"] == []


def test_only_the_answer_route_example_carries_a_tool_call_trajectory():
    """The trajectory belongs to the resolve agent, which runs on the
    `answer` route; the refund route never reaches it."""
    with_messages = [ex for ex in REGRESSION_EXAMPLES if "messages" in ex["outputs"]]
    assert len(with_messages) == 1
    assert with_messages[0]["outputs"]["route"] == "answer"


def test_the_refund_example_carries_the_ticket_and_a_decision_with_by():
    refund = _example(target="resolve", message="I need a refund for T-1001: $49.")
    assert refund["inputs"]["ticket"] == {
        "id": "T-1001", "amount": 49.0, "customer_id": "C-1"
    }
    assert refund["inputs"]["resume"]["type"] == "approve"
    assert refund["inputs"]["resume"]["by"]
    assert "messages" not in refund["outputs"]


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
    assert handoffs_within_bound({}, {"handoffs": 3}, {"max_handoffs": 2}) is False


def test_handoffs_within_bound_is_a_no_op_on_support_graph_examples():
    assert handoffs_within_bound({}, {"route": "answer"}, {"route": "answer"}) is True


def test_tool_call_correct_is_a_no_op_when_no_messages_reference_exists():
    assert tool_call_correct({"route": "answer"}, {"route": "answer"}) is True


def test_tool_call_correct_ignores_real_arguments_against_the_empty_reference():
    """Real runs carry real arguments; the reference's `{}` names the tools
    only. The default "exact" args mode would fail every real run."""
    reference = {"messages": _trajectory("lookup_ticket", "set_ticket_status")}
    outputs = {"messages": _trajectory(
        "lookup_ticket", "set_ticket_status", args='{"ticket_id": "T-1001"}'
    )}

    assert tool_call_correct(outputs, reference)["score"] is True


def test_tool_call_correct_allows_an_extra_call():
    reference = {"messages": _trajectory("lookup_ticket", "set_ticket_status")}
    outputs = {"messages": _trajectory(
        "lookup_ticket", "search_kb", "set_ticket_status"
    )}

    assert tool_call_correct(outputs, reference)["score"] is True


def test_tool_call_correct_fails_when_lookup_ticket_is_skipped():
    reference = {"messages": _trajectory("lookup_ticket", "set_ticket_status")}
    outputs = {"messages": _trajectory("set_ticket_status")}

    assert tool_call_correct(outputs, reference)["score"] is False


def test_superset_mode_does_not_check_order():
    """What the chapter says about "superset": a run that writes before it
    looks still passes. Pinned here so a library change shows up."""
    reference = {"messages": _trajectory("lookup_ticket", "set_ticket_status")}
    outputs = {"messages": _trajectory("set_ticket_status", "lookup_ticket")}

    assert tool_call_correct(outputs, reference)["score"] is True


def test_all_evaluators_lists_all_five_evaluators_in_the_documented_order():
    assert ALL_EVALUATORS == [
        routing_correct,
        handoffs_within_bound,
        tool_call_correct,
        answer_quality,
        checkins_sent_only_if_approved,
    ]


# --- answer_quality: two prompts, one scripted judge ------------------------


def _scripted_judges(monkeypatch) -> _ScriptedJudge:
    judge = _ScriptedJudge(seen=[])
    monkeypatch.setattr(evals_module, "reference_judge", create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT, feedback_key="correctness", judge=judge,
    ))
    monkeypatch.setattr(evals_module, "online_judge", create_llm_as_judge(
        prompt=ONLINE_CORRECTNESS_PROMPT, feedback_key="online-correctness",
        judge=judge,
    ))
    return judge


def test_the_correctness_prompt_cannot_run_without_a_reference():
    """Why the online monitor needs its own prompt: CORRECTNESS_PROMPT
    formats `{reference_outputs}`, and a production trace has none."""
    judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT, judge=_ScriptedJudge(seen=[])
    )
    with pytest.raises(KeyError, match="reference_outputs"):
        judge(inputs={"message": "hi"}, outputs={"answer": "hello"})


def test_answer_quality_is_reference_free_on_a_production_shaped_run(monkeypatch):
    """The monitor's call shape - inputs and outputs only - scores instead
    of raising, through the online prompt."""
    judge = _scripted_judges(monkeypatch)

    result = answer_quality(
        inputs={"messages": [{"role": "user", "content": "Where is order 4471?"}]},
        outputs={"messages": [{"role": "assistant", "content": "It ships today."}]},
    )

    assert result["key"] == "online-correctness"
    assert result["score"] is True
    prompt = judge.seen[0][0]["content"]
    assert "There is no reference answer." in prompt
    assert "Where is order 4471?" in prompt


def test_the_monitor_runs_end_to_end_with_the_real_answer_quality(monkeypatch):
    """atlas/monitor.py, unchanged, over a production-shaped run: the real
    answer_quality (scripted judge model) scores it and the feedback lands."""
    _scripted_judges(monkeypatch)
    run = SimpleNamespace(
        id=0,
        inputs={"messages": [{"role": "user", "content": "Close T-1001."}]},
        outputs={"messages": [{"role": "assistant", "content": "Closed."}]},
    )
    monkeypatch.setattr(monitor_module.Client, "list_runs", lambda self, **kw: [run])
    feedback = []
    monkeypatch.setattr(
        monitor_module.Client, "create_feedback",
        lambda self, run_id, key, score: feedback.append((run_id, key, score)),
    )

    monitor_module.run_quality_monitor(sample_rate=0.05)

    assert feedback == [(0, "online-correctness", True)]


def test_answer_quality_grades_the_final_answer_against_the_reference(monkeypatch):
    judge = _scripted_judges(monkeypatch)

    result = answer_quality(
        inputs={"message": "T-1001 is fixed. Please close it."},
        outputs={"messages": [AIMessage("Done: T-1001 is resolved.")]},
        reference_outputs={"route": "answer", "answer": "T-1001 is now resolved."},
    )

    assert result["key"] == "correctness"
    prompt = judge.seen[0][0]["content"]
    assert "Done: T-1001 is resolved." in prompt
    assert "T-1001 is now resolved." in prompt


def test_answer_quality_is_a_no_op_without_a_reference_answer(monkeypatch):
    judge = _scripted_judges(monkeypatch)

    assert answer_quality({}, {"route": "retrieve"}, {"route": "retrieve"}) is True
    assert judge.seen == []


# --- Chapter 27: checkins_sent_only_if_approved -----------------------------


def test_checkins_sent_only_if_approved_is_a_no_op_when_no_reference_exists():
    assert checkins_sent_only_if_approved({"route": "answer"}, {"route": "answer"}) is True


def test_checkins_sent_only_if_approved_passes_when_the_approved_ticket_sent():
    assert checkins_sent_only_if_approved(
        {"sla_watch_sent": ["T-2001"]}, {"sla_watch_sent": ["T-2001"]}
    ) is True


def test_checkins_sent_only_if_approved_fails_when_a_rejected_draft_still_sent():
    assert checkins_sent_only_if_approved(
        {"sla_watch_sent": ["T-2001"]}, {"sla_watch_sent": []}
    ) is False


# --- Chapter 27: run_sla_watch, driven through a real interrupt/resume -----


def test_run_sla_watch_sends_the_check_in_when_approved():
    result = run_sla_watch({"target": "sla_watch", "decision": "approve"})
    assert result == {"sla_watch_sent": ["T-2001"]}


def test_run_sla_watch_sends_nothing_when_rejected():
    result = run_sla_watch({"target": "sla_watch", "decision": "reject"})
    assert result == {"sla_watch_sent": []}


def test_run_atlas_dispatches_sla_watch_targets_without_a_message_field():
    """SLA Watch runs on a schedule, not a customer turn - its dataset
    examples carry no `message` key, so run_atlas must branch to
    run_sla_watch BEFORE it ever reads inputs["message"]."""
    result = run_atlas({"target": "sla_watch", "decision": "approve"})
    assert result == {"sla_watch_sent": ["T-2001"]}


# --- run_atlas: research examples on the real supervisor graph -------------


def _delegate(specialist: str, call_id: str) -> AIMessage:
    return AIMessage("", tool_calls=[_call(
        f"delegate_to_{specialist}", {"task": f"look up {call_id}"}, call_id
    )])


def _scripted_supervisor(monkeypatch, script: list[AIMessage]) -> None:
    """Chapter 16's real build_supervisor_graph around a scripted
    coordinator; the specialists' own model calls are faked."""

    class _FakeSpecialist:
        def invoke(self, input_: dict) -> dict:
            return {"messages": [AIMessage("Vendor SLAs: 99.9% uptime.")]}

    coordinator = create_agent(
        model=_ScriptedChatModel(script=script),
        tools=[
            make_handoff("web_research", "Delegate a web-search sub-task."),
            make_handoff("doc_research", "Delegate an internal-docs sub-task."),
        ],
        system_prompt="coordinate",
        state_schema=SupervisorState,
    )
    monkeypatch.setattr(research_module, "create_agent", lambda **_: _FakeSpecialist())
    monkeypatch.setattr(
        evals_module, "supervisor_graph", build_supervisor_graph(coordinator)
    )


def test_research_examples_target_chapter_16s_supervisor_graph():
    assert evals_module.supervisor_graph is research_module.supervisor_graph


def test_the_research_example_runs_and_passes_the_bound_on_a_two_hop_run(
    monkeypatch,
):
    _scripted_supervisor(monkeypatch, [
        _delegate("web_research", "c1"),
        _delegate("doc_research", "c2"),
        AIMessage("Our SLA matches two of three competitors."),
    ])
    example = _example(target="research")

    outputs = run_atlas(example["inputs"])

    assert outputs["handoffs"] == 2
    assert handoffs_within_bound(example["inputs"], outputs, example["outputs"]) is True


def test_the_research_example_catches_the_opening_hooks_redundant_hop(monkeypatch):
    """The opening hook, reproduced: web, doc, then web again before the
    answer. The answer is fine; the third handoff fails the bound."""
    _scripted_supervisor(monkeypatch, [
        _delegate("web_research", "c1"),
        _delegate("doc_research", "c2"),
        _delegate("web_research", "c3"),
        AIMessage("Our SLA matches two of three competitors."),
    ])
    example = _example(target="research")

    outputs = run_atlas(example["inputs"])

    assert outputs["handoffs"] == 3
    assert outputs["messages"][-1].content.startswith("Our SLA matches")
    verdict = handoffs_within_bound(example["inputs"], outputs, example["outputs"])
    assert verdict is False


# --- run_atlas: support examples on the real resolved graph ----------------


def _scripted_resolved(monkeypatch, route: str, script: list[AIMessage]) -> None:
    """Chapter 17's real build_resolved_graph - the whole RESOLVE_MIDDLEWARE
    stack - with triage forced to `route` and the resolve agent's model
    scripted. The seeded ticket and refund backends are swapped for copies,
    so a test that closes T-1001 or refunds it leaves the seed as it was
    for every other test module."""
    monkeypatch.setattr(tools_module, "_TICKETS", copy.deepcopy(tools_module._TICKETS))
    monkeypatch.setattr(
        effects_module, "_REFUNDS", copy.deepcopy(effects_module._REFUNDS)
    )
    monkeypatch.setattr(effects_module, "_LEDGER", {})
    monkeypatch.setattr(
        graph_module, "classify", lambda messages: SimpleNamespace(route=route)
    )
    monkeypatch.setattr(agent_module, "model", _ScriptedChatModel(script=script))
    monkeypatch.setattr(evals_module, "resolved", build_resolved_graph())


def test_the_answer_example_resumes_the_write_and_passes_the_trajectory(
    monkeypatch,
):
    """The resolve agent looks the ticket up, proposes the write, pauses at
    HumanInTheLoopMiddleware, and run_atlas resumes it with the example's
    own decision - so the trajectory and the route are both scored."""
    _scripted_resolved(monkeypatch, "answer", [
        AIMessage("", tool_calls=[
            _call("lookup_ticket", {"ticket_id": "T-1001"}, "c1")
        ]),
        AIMessage("", tool_calls=[_call(
            "set_ticket_status", {"ticket_id": "T-1001", "status": "resolved"}, "c2"
        )]),
        AIMessage("Ticket T-1001 is now resolved."),
    ])
    example = _example(target="resolve", message="T-1001 is fixed. Please close it.")

    outputs = run_atlas(example["inputs"])

    assert "__interrupt__" not in outputs
    assert routing_correct(example["inputs"], outputs, example["outputs"]) is True
    assert tool_call_correct(outputs, example["outputs"])["score"] is True


def test_the_refund_example_resumes_the_approval_gate_and_refunds(monkeypatch):
    """The path the refund actually takes: triage -> approval_gate (pause)
    -> resumed with a decision carrying `by` -> refund. No agent runs."""
    _scripted_resolved(monkeypatch, "refund", [])
    example = _example(target="resolve", message="I need a refund for T-1001: $49.")

    outputs = run_atlas(example["inputs"])

    assert "__interrupt__" not in outputs
    assert outputs["refund_done"] is True
    assert outputs["approval"]["by"] == "eval@ci"
    assert routing_correct(example["inputs"], outputs, example["outputs"]) is True


def test_run_atlas_uses_a_fresh_thread_id_on_every_call(monkeypatch):
    """Chapter 9: the resolved graph is checkpointer-backed - a fixed
    thread_id would let one example's state (or a suspended approval_gate
    interrupt) leak into the next one."""
    configs = []
    monkeypatch.setattr(
        evals_module.resolved, "invoke",
        lambda inputs, config, **kw: configs.append(config) or {},
    )

    run_atlas({"message": "one", "target": "resolve"})
    run_atlas({"message": "two", "target": "resolve"})

    thread_ids = [c["configurable"]["thread_id"] for c in configs]
    assert thread_ids[0] != thread_ids[1]


# --- The merge gate: evaluate() records, gate() decides --------------------


def _row(example_id: str, *scores: Any, error: str | None = None) -> dict:
    return {
        "run": SimpleNamespace(error=error),
        "example": SimpleNamespace(id=example_id),
        "evaluation_results": {"results": [
            EvaluationResult(key=f"e{i}", score=s) for i, s in enumerate(scores)
        ]},
    }


def test_gate_passes_when_every_evaluator_passes():
    assert gate([_row("a", True, True), _row("b", True, 1.0)]) == 0


def test_gate_fails_on_any_false_score():
    assert gate([_row("a", True, True), _row("b", True, False)]) == 1


def test_gate_fails_on_a_numeric_zero_score():
    """A numeric evaluator scores a failure 0 (or 0.0), not False."""
    assert gate([_row("a", True, 0)]) == 1
    assert gate([_row("b", 0.0)]) == 1
    assert gate([_row("c", None, 0.5)]) == 0


def test_gate_fails_on_a_run_that_raised():
    """A run that raised hands its evaluators empty outputs, which a no-op
    evaluator would pass: the KeyError that hid the research bug."""
    assert gate([_row("a", True, error="KeyError: 'sources'")]) == 1


# --- External-service exception: LangSmith -------------------------------


@requires_langsmith
def test_build_regression_dataset_creates_the_frozen_dataset_in_langsmith():
    """Skipped by default - see `requires_langsmith` above. Real teams run
    this once, not in CI.

    It reads the dataset back out of LangSmith and checks that every
    REGRESSION_EXAMPLES entry actually landed. The name is per-run because
    `create_dataset` errors on a collision, by design - the chapter argues a
    frozen dataset should be hard to overwrite.
    """
    import uuid

    name = f"atlas-regression-test-{uuid.uuid4().hex[:12]}"

    build_regression_dataset(name)
    dataset = None
    try:
        dataset = evals_module.client.read_dataset(dataset_name=name)
        assert dataset.name == name

        examples = list(evals_module.client.list_examples(dataset_id=dataset.id))
        assert len(examples) == len(REGRESSION_EXAMPLES)

        expected_routes = sorted(
            e["outputs"]["route"]
            for e in REGRESSION_EXAMPLES
            if "route" in e["outputs"]
        )
        actual_routes = sorted(
            e.outputs["route"] for e in examples if e.outputs and "route" in e.outputs
        )
        assert actual_routes == expected_routes
    finally:
        # Delete by ID, not by name: delete_dataset(dataset_name=...) returned
        # a 500 from the live API when this test was first run against it.
        if dataset is not None:
            evals_module.client.delete_dataset(dataset_id=dataset.id)


@requires_langsmith
@requires_anthropic
def test_run_regression_suite_gates_against_the_frozen_dataset():
    """Skipped by default - needs both a live LangSmith dataset and a live
    Anthropic connection for the judge and the resolve agent.

    NOT YET RUN AGAINST THE LIVE SERVICES. Treat the assertions below as
    unverified until someone runs it, and read the result rather than the
    green tick the first time.
    """
    results = run_regression_suite()

    rows = list(results)
    assert rows, "the suite evaluated no examples, so it gates nothing"
    assert len(rows) == len(REGRESSION_EXAMPLES)
    for row in rows:
        scores = {r.key: r for r in row["evaluation_results"]["results"]}
        assert scores, "a row came back with no evaluator results at all"
    assert gate(rows) == 0
