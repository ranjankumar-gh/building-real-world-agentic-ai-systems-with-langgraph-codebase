"""Chapter 21, "Evaluation and Testing" - Atlas's path-coverage eval suite.

See "Building a path-coverage dataset" and "Four evaluators, four different
questions". `atlas-regression` is a **frozen** LangSmith dataset - one
example per known Atlas route (Chapter 6's `answer`/`retrieve`/`escalate`,
Chapter 11's refund approve/reject/edit, Chapter 16's research handoffs) -
not a pile of differently-worded questions that all happen to land on the
same route. `build_regression_dataset` is the one-time setup script that
creates it: a real, one-shot write against a live LangSmith project, so it
is guarded behind `if __name__ == "__main__"` rather than run on import -
importing this module (as `tests/test_evals.py` and `atlas/monitor.py` both
do) must never itself create a dataset, let alone try to create the same
one twice.

The four evaluators below answer four different questions. `routing_correct`
and `handoffs_within_bound` are hand-rolled and deterministic - they read
facts already sitting in Atlas's own returned state (Chapter 6's `route`,
Chapter 16's `handoffs`) and need no model call. Each is a no-op (returns
`True`) on examples whose `reference_outputs` lack the field it checks,
since `route` (main-graph-only) and `handoffs`/`max_handoffs`
(research-graph-only) never coexist on the same example - Atlas is two
graphs, not one (Chapter 15's multi-agent-split decision). Only
`handoffs_within_bound` would have caught the opening hook's regression: the
final answer and the final specialist were both correct, only the hop count
was wrong.

`tool_call_correct` checks *how* `resolve_agent` (Chapter 7) got its answer,
not just where it ended up - via `agentevals`, a pre-1.0 langchain-ai
package purpose-built for tool-call trajectory comparison (verified against
0.0.9 on PyPI; same maturity caveat Chapter 14 gave LangMem). `answer_quality`
is the only evaluator that needs a model at all - an LLM-as-judge via
`openevals` (same langchain-ai family, same caveat; verified against 0.2.0),
calibrated against human labels before it's trusted (see the chapter's Deep
Dive) and requiring a live Anthropic connection to actually score anything.

`run_atlas` is the CI-facing target `evaluate()` calls once per dataset
example: it dispatches to whichever of Atlas's two graphs the example's
`target` field names. `graph` is checkpointer-backed (Chapter 9) - a fresh
`thread_id` per call keeps one example's state (or a suspended
`approval_gate` interrupt from the refund example) from leaking into the
next one."""

from __future__ import annotations

import uuid

from agentevals.trajectory.match import create_trajectory_match_evaluator
from langsmith import Client
from langsmith.evaluation import evaluate
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from atlas.graph import graph
from atlas.research import research_graph

REGRESSION_DATASET = "atlas-regression"

# One example per known Atlas ROUTE - path coverage, not question variety.
# See "Building a path-coverage dataset".
REGRESSION_EXAMPLES = [
    {
        "inputs": {"message": "What's your refund policy?", "target": "resolve"},
        "outputs": {"route": "answer"},
    },
    {
        "inputs": {"message": "My order 4471 never arrived.", "target": "resolve"},
        "outputs": {"route": "retrieve"},
    },
    {
        "inputs": {
            "message": "I want a refund for order 4471, $340.",
            "target": "resolve",
        },
        "outputs": {
            "route": "refund",
            "decision": "approve",
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
        "inputs": {
            "message": "Compare our SLA to three competitors' public SLAs.",
            "target": "research",
        },
        "outputs": {"max_handoffs": 2},
    },
]

client = Client()  # local construction only - no network call, no API key needed


def build_regression_dataset(
    dataset_name: str = REGRESSION_DATASET,
) -> None:
    """One-time setup: create the frozen path-coverage dataset in LangSmith
    and seed it with `REGRESSION_EXAMPLES`. Requires a live LangSmith
    connection - run once (e.g. `python -m atlas.evals`), not on import, and
    not more than once against the same name (`create_dataset` errors on a
    collision - see "Editing a failing regression example is not the same
    as fixing the bug")."""
    dataset = client.create_dataset(
        dataset_name,
        description="One example per known Atlas route - path coverage, not questions.",
    )
    client.create_examples(dataset_id=dataset.id, examples=REGRESSION_EXAMPLES)


def routing_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Ch6: did Atlas take the expected route? A no-op on research examples,
    which have no `route` key - that field belongs to the main graph only."""
    if "route" not in reference_outputs:
        return True
    return outputs.get("route") == reference_outputs["route"]


def handoffs_within_bound(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> bool:
    """Ch16: did the run take no more handoffs than this example needs? A
    no-op on main-graph examples, which have no `max_handoffs` reference.

    This is the ONLY evaluator in the suite that would have caught the
    opening hook's regression - the final answer and even the final
    specialist were both correct; only the hop count was wrong.
    """
    if "max_handoffs" not in reference_outputs:
        return True
    return outputs.get("handoffs", 0) <= reference_outputs["max_handoffs"]


refund_trajectory = create_trajectory_match_evaluator(
    trajectory_match_mode="superset",  # extra tool calls OK; reference calls must all appear, in order
)


def tool_call_correct(outputs: dict, reference_outputs: dict) -> dict | bool:
    """Ch7: did resolve_agent call at least the expected tools, in order? A
    no-op on examples with no `messages` reference to check against."""
    if "messages" not in reference_outputs:
        return True
    return refund_trajectory(
        outputs=outputs["messages"],
        reference_outputs=reference_outputs["messages"],
    )


answer_quality = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model="anthropic:claude-sonnet-4-6",
)


def run_atlas(inputs: dict) -> dict:
    """The CI-facing target: dispatch to the graph this example targets,
    since Atlas is two graphs, not one (Chapter 15's multi-agent boundary).
    invoke() already returns the full final state for whichever graph ran -
    route and messages from the main graph, handoffs and messages from the
    research graph - no separate get_state() call needed for a fresh run."""
    message = {"role": "user", "content": inputs["message"]}
    config = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    if inputs["target"] == "research":
        return research_graph.invoke({"messages": [message]}, config)
    return graph.invoke({"messages": [message]}, config)


ALL_EVALUATORS = [
    routing_correct, handoffs_within_bound, tool_call_correct, answer_quality
]


def run_regression_suite(dataset_name: str = REGRESSION_DATASET):
    """Wire the suite into CI as a merge gate against the FROZEN dataset -
    distinct from an evolving dev-iteration dataset. See "Wiring the suite
    into CI"."""
    return evaluate(
        run_atlas,
        data=dataset_name,
        evaluators=ALL_EVALUATORS,
        experiment_prefix="ci",
    )


if __name__ == "__main__":
    build_regression_dataset()
