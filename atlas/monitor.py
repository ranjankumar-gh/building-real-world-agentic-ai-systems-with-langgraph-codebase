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
reasonable during development.

`is_root=True`: Chapter 20's `trace_config` tags are inherited by every span
under the root, so `has(tags, "atlas")` alone also returns the model, tool,
and node spans, and the judge would grade a tool call as if it were an
answer. `error=False` drops runs that raised: they have no answer to grade.

Each tick scores the window that has just closed, not the one still open:
it reads from `now - 2 * window` and skips any run that started after
`now - window`, so a run gets a full schedule period (Chapter 22's cron
fires every 15 minutes) to finish before it is graded, and consecutive
ticks score disjoint slices with no second feedback row on a scored run. A
run still in flight after that has `outputs=None`; it is skipped, because
`online_judge` raises `KeyError: 'outputs'` on it and would abort the tick.
A window, rather than a `since` the caller passes, keeps the monitor
stateless: Chapter 22's cron starts every run on a fresh thread with only
`sample_rate` as input."""

from datetime import datetime, timedelta, timezone

from langsmith import Client

from atlas.evals import answer_quality

client = Client()  # no API key needed to build; its background thread fetches /info


def run_quality_monitor(
    sample_rate: float = 0.05, window: timedelta = timedelta(minutes=15)
) -> None:
    """Score a sampled slice of tagged production traces (Chapter 20)."""
    now = datetime.now(timezone.utc)
    runs = client.list_runs(
        project_name="atlas-prod",
        filter='has(tags, "atlas")',
        is_root=True,
        error=False,
        start_time=now - 2 * window,
    )
    for run in runs:
        if run.start_time > now - window or run.outputs is None:
            continue
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
