"""Chapter 21, "Evaluation and Testing" - the online quality monitor.

See "The online quality monitor: the same judge model, no reference".
CI (`atlas/evals.py`) answers "did this change regress." It says nothing
about whether production quality is holding *today*, against real traffic
no dataset was built to anticipate. `run_quality_monitor` reuses the same
calibrated `answer_quality` judge against a sampled slice of Chapter 20's
tagged production traces, and writes the score back as LangSmith feedback
attached to the run it graded - a live signal, not a merge-time one.

`hash(run.id) % 100 >= sample_rate * 100` keeps the monitor's own cost
bounded - judging every production run with another model call doubles the
LLM spend on every request. `sample_rate` is a knob to tune against
Chapter 23's cost controls, not a constant to leave at whatever felt
reasonable during development."""

from langsmith import Client

from atlas.evals import answer_quality

client = Client()  # no API key needed to build; its background thread fetches /info


def run_quality_monitor(sample_rate: float = 0.05) -> None:
    """Score a sampled slice of tagged production traces (Chapter 20)."""
    runs = client.list_runs(
        project_name="atlas-prod",
        filter='has(tags, "atlas")',
        limit=200,
    )
    for run in runs:
        if hash(run.id) % 100 >= sample_rate * 100:
            continue
        result = answer_quality(
            inputs=run.inputs,
            outputs=run.outputs,
        )
        client.create_feedback(
            run_id=run.id,
            key="online-correctness",
            score=result["score"],
        )
