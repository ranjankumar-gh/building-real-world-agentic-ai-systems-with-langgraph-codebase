"""Chapter 21, "Evaluation and Testing" - Atlas's path-coverage eval suite.

See "Building a path-coverage dataset" and "Four evaluators, four different
questions". `atlas-regression` is a **frozen** LangSmith dataset - one
example per known Atlas route (Chapter 6's `answer`/`retrieve`, Chapter 11's
approved refund, Chapter 16's research handoffs) - not a pile of
differently-worded questions that all happen to land on the same route.
`build_regression_dataset` is the one-time setup script that creates it: a
real, one-shot write against a live LangSmith project, so it runs only from
`python -m atlas.evals --create-dataset`, never on import - importing this
module (as `tests/test_evals.py` and `atlas/monitor.py` both do) must never
itself create a dataset, let alone try to create the same one twice.

The four evaluators below answer four different questions. `routing_correct`
and `handoffs_within_bound` are hand-rolled and deterministic - they read
facts already sitting in Atlas's own returned state (Chapter 6's `route`,
Chapter 16's `handoffs`) and need no model call. Each is a no-op (returns
`True`) on examples whose `reference_outputs` lack the field it checks,
since `route` (support-graph-only) and `handoffs`/`max_handoffs`
(research-supervisor-only) never coexist on the same example. Only
`handoffs_within_bound` would have caught the opening hook's regression: the
final answer and the final specialist were both correct, only the hop count
was wrong.

`tool_call_correct` checks *how* the mounted resolve agent (Chapters 7 and
17) got its answer, not just where it ended up - via `agentevals`, a pre-1.0
langchain-ai package purpose-built for tool-call trajectory comparison
(verified against 0.0.9). It is a structural comparison, no model call; its
"superset" mode does not compare order. `answer_quality` is the only
evaluator that calls a model - an LLM-as-judge via `openevals` (same
langchain-ai family, verified against 0.2.0), calibrated against human
labels before it's trusted (see the chapter's Deep Dive). Offline it grades
against the example's reference answer; called with no reference at all,
the way `atlas/monitor.py` calls it on production runs, it switches to a
reference-free prompt, because `CORRECTNESS_PROMPT` needs
`{reference_outputs}` and a production trace has none.

`run_atlas` is the CI-facing target `evaluate()` calls once per dataset
example. Support examples run on `build_resolved_graph()` (Chapter 17's
production assembly, not the module-level `graph` and its model-free
`answer` stub); research examples run on Chapter 16's `supervisor_graph`,
seeded with `"handoffs": 0`, because that is the graph that counts handoffs
(Chapter 17's map-reduce, `run_research(sources, thread_id)`, has no
handoff counter and is not under test here). An example that pauses - the
refund at `approval_gate`, a status change at the resolve agent's
`HumanInTheLoopMiddleware` - carries its own `resume` value, and `run_atlas`
resumes it, so the run reaches the end of the path the example covers.
`gate` turns `evaluate()`'s recorded scores into the exit status a CI job
blocks a merge on.

Chapter 27, "Capstone", adds `checkins_sent_only_if_approved` - SLA Watch's
own deterministic evaluator, following the same no-op-if-not-applicable
shape as `routing_correct`/`handoffs_within_bound` - plus two new examples
(`target: "sla_watch"`) in the SAME frozen `atlas-regression` dataset, not a
parallel suite. `run_sla_watch` is `run_atlas`'s dispatch target for those
two examples: it drives `atlas/sla_watch.py`'s `build_sla_watch_graph()`
through a REAL suspend-at-`approval_gate`/resume-with-`Command(resume=...)`
cycle - the example's own `decision` field becomes every draft's decision -
and reports which ticket ids actually got a `send_checkin` call, the exact
field `checkins_sent_only_if_approved` checks against the reference."""

from __future__ import annotations

import sys
import uuid
from collections.abc import Iterable

from agentevals.trajectory.match import create_trajectory_match_evaluator
from langgraph.types import Command
from langsmith import Client
from langsmith.evaluation import evaluate
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from atlas.research import supervisor_graph
from atlas.resolve import build_resolved_graph
from atlas.security import AtlasContext
from atlas.sla_watch import build_sla_watch_graph
from atlas.tools import text_of

REGRESSION_DATASET = "atlas-regression"

# One example per known Atlas ROUTE - path coverage, not question variety.
# See "Building a path-coverage dataset".
REGRESSION_EXAMPLES = [
    {
        "inputs": {
            "message": "T-1001 is fixed. Please close it.",
            "target": "resolve",
            "resume": {"decisions": [{"type": "approve"}]},
        },
        "outputs": {
            "route": "answer",
            "answer": "Ticket T-1001 is now resolved.",
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"function": {"name": "lookup_ticket", "arguments": "{}"}},
                ]},
                {"role": "assistant", "tool_calls": [
                    {"function": {"name": "set_ticket_status", "arguments": "{}"}},
                ]},
            ],
        },
    },
    {
        "inputs": {"message": "My order 4471 never arrived.", "target": "resolve"},
        "outputs": {"route": "retrieve"},
    },
    {
        "inputs": {
            "message": "I need a refund for T-1001: $49.",
            "target": "resolve",
            "ticket": {"id": "T-1001", "amount": 49.0, "customer_id": "C-1"},
            "resume": {"type": "approve", "by": "eval@ci"},
        },
        "outputs": {"route": "refund"},
    },
    {
        "inputs": {
            "message": "Compare our SLA to three competitors' public SLAs.",
            "target": "research",
        },
        "outputs": {"max_handoffs": 2},
    },
    # Chapter 27: SLA Watch's two paths - approve sends, reject doesn't. No
    # "message" field - SLA Watch runs on a schedule, not a customer turn.
    {
        "inputs": {"target": "sla_watch", "decision": "approve"},
        "outputs": {"sla_watch_sent": ["T-2001"]},
    },
    {
        "inputs": {"target": "sla_watch", "decision": "reject"},
        "outputs": {"sla_watch_sent": []},
    },
]

client = Client()  # no API key needed to build; its background thread fetches /info


def build_regression_dataset(
    dataset_name: str = REGRESSION_DATASET,
) -> None:
    """One-time setup: create the frozen dataset and seed it."""
    dataset = client.create_dataset(
        dataset_name,
        description="One example per known Atlas route - path coverage, not questions.",
    )
    client.create_examples(dataset_id=dataset.id, examples=REGRESSION_EXAMPLES)


def routing_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Ch6: did Atlas take the expected route? A no-op on research examples,
    which have no `route` key - that field belongs to the support graph."""
    if "route" not in reference_outputs:
        return True
    return outputs.get("route") == reference_outputs["route"]


def handoffs_within_bound(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> bool:
    """Ch16: did the run take no more handoffs than this example needs? A
    no-op on support-graph examples, which have no `max_handoffs` reference.

    This is the ONLY evaluator in the suite that would have caught the
    opening hook's regression - the final answer and even the final
    specialist were both correct; only the hop count was wrong.
    """
    if "max_handoffs" not in reference_outputs:
        return True
    return outputs.get("handoffs", 0) <= reference_outputs["max_handoffs"]


resolve_trajectory = create_trajectory_match_evaluator(
    trajectory_match_mode="superset",
    tool_args_match_mode="ignore",
)


def tool_call_correct(outputs: dict, reference_outputs: dict) -> dict | bool:
    """Ch7: did the resolve agent call at least the expected tools? A no-op
    on examples with no `messages` reference to check against."""
    if "messages" not in reference_outputs:
        return True
    return resolve_trajectory(
        outputs=outputs["messages"],
        reference_outputs=reference_outputs["messages"],
    )


JUDGE_MODEL = "anthropic:claude-sonnet-4-6"

ONLINE_CORRECTNESS_PROMPT = """\
You are grading a customer-support answer. There is no reference answer.
Score it correct only if it answers what the customer asked, contradicts
nothing in the request, and states no amount, date, or policy it could not
have known from the request or a tool result it reports.

<request>
{inputs}
</request>

<answer>
{outputs}
</answer>
"""

reference_judge = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model=JUDGE_MODEL,
)
online_judge = create_llm_as_judge(
    prompt=ONLINE_CORRECTNESS_PROMPT,  # no {reference_outputs}: production has none
    feedback_key="online-correctness",
    model=JUDGE_MODEL,
)


def answer_quality(
    inputs: dict, outputs: dict, reference_outputs: dict | None = None
) -> dict | bool:
    """Is the answer correct? Against the example's reference answer in CI;
    reference-free when called with no reference at all (the online
    monitor). A no-op on examples that carry no reference answer."""
    if reference_outputs is None:
        return online_judge(inputs=inputs, outputs=outputs)
    if "answer" not in reference_outputs:
        return True
    return reference_judge(
        inputs=inputs["message"],
        outputs=text_of(outputs["messages"][-1]),
        reference_outputs=reference_outputs["answer"],
    )


def checkins_sent_only_if_approved(outputs: dict, reference_outputs: dict) -> bool:
    """Chapter 27: SLA Watch's own deterministic check: send_checkin must
    never fire for a rejected draft. A no-op on every other route's
    examples, the same no-op-if-not-applicable shape as `routing_correct`/
    `handoffs_within_bound` above."""
    if "sla_watch_sent" not in reference_outputs:
        return True
    return outputs.get("sla_watch_sent", []) == reference_outputs["sla_watch_sent"]


def run_sla_watch(inputs: dict) -> dict:
    """Chapter 27: drive one SLA Watch dataset example through a REAL
    suspend-at-`approval_gate`/resume-with-`Command(resume=...)` cycle -
    every drafted check-in gets the example's own `decision` - and report
    which ticket ids actually received a `send_checkin` call, the field
    `checkins_sent_only_if_approved` checks. A fresh graph (fresh
    checkpointer AND fresh store) per call, the same example-isolation
    discipline `run_atlas`'s fresh `thread_id` already applies below."""
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    watch_graph = build_sla_watch_graph()
    watch_graph.invoke({}, config)  # suspends at approval_gate
    n_drafts = len(watch_graph.get_state(config).values.get("drafts", []))
    decisions = [{"type": inputs["decision"]}] * n_drafts
    result = watch_graph.invoke(Command(resume=decisions), config)
    sent = [
        draft["ticket_id"]
        for draft, decision in zip(result["drafts"], result["decisions"])
        if decision["type"] != "reject"
    ]
    return {"sla_watch_sent": sent}


resolved = build_resolved_graph()
EVAL_CONTEXT = AtlasContext(role="support_agent", customer_id="C-1")


def run_atlas(inputs: dict) -> dict:
    """The CI-facing target: dispatch to the graph this example targets.
    Resume a paused run with the example's own `resume` value, so the run
    finishes the path it covers."""
    if inputs["target"] == "sla_watch":  # Chapter 27: no customer message
        return run_sla_watch(inputs)
    message = {"role": "user", "content": inputs["message"]}
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    if inputs["target"] == "research":
        return supervisor_graph.invoke({"messages": [message], "handoffs": 0}, config)
    state: dict = {"messages": [message]}
    if "ticket" in inputs:
        state["ticket"] = inputs["ticket"]
    out = resolved.invoke(state, config, context=EVAL_CONTEXT)
    if "__interrupt__" in out and "resume" in inputs:
        out = resolved.invoke(
            Command(resume=inputs["resume"]), config, context=EVAL_CONTEXT
        )
    return out


ALL_EVALUATORS = [
    routing_correct,
    handoffs_within_bound,
    tool_call_correct,
    answer_quality,
    checkins_sent_only_if_approved,
]


def run_regression_suite(dataset_name: str = REGRESSION_DATASET):
    """Run the suite against the FROZEN dataset - distinct from an evolving
    dev-iteration dataset. See "Wiring the suite into CI"."""
    return evaluate(
        run_atlas,
        data=dataset_name,
        evaluators=ALL_EVALUATORS,
        experiment_prefix="ci",
    )


def gate(results: Iterable[dict]) -> int:
    """The merge gate: exit status 1 if any run raised or any evaluator
    scored any example False or 0. `evaluate()` only records scores."""
    failed = [
        row["example"].id
        for row in results
        if row["run"].error
        or any(r.score in (False, 0) for r in row["evaluation_results"]["results"])
    ]
    for example_id in failed:
        print(f"regression: example {example_id} failed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--create-dataset"]:
        build_regression_dataset()
    else:
        raise SystemExit(gate(run_regression_suite()))
