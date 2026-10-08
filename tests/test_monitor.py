"""Chapter 21, "Evaluation and Testing" - atlas/monitor.py.

See "The online quality monitor: the same judge model, no reference".
`run_quality_monitor`'s own logic (the hash-based sampling, the
answer_quality/create_feedback wiring) is exercised below by monkeypatching
`Client.list_runs`/`Client.create_feedback` (the LangSmith `Client`'s
attributes are read-only per-instance, so the class methods are patched
instead) and `answer_quality` - none of which touches the network in the
test itself, matching the style already used for `atlas/tracing.py`'s
`hide_outputs`. Actually calling the real `client.list_runs`/
`client.create_feedback` against a live LangSmith project needs a live
connection; nothing in this file does that."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from atlas import monitor as monitor_module
from atlas.monitor import run_quality_monitor


_SETTLED = object()


def _run(run_id, inputs=None, outputs=_SETTLED, age=timedelta(minutes=20)):
    """A stand-in for a LangSmith Run - just enough shape (.id, .inputs,
    .outputs, .start_time) for run_quality_monitor to read, without a live
    connection. The default age puts it in the window that has just closed
    (15-30 minutes ago); `outputs=None` is a run still in flight."""
    return SimpleNamespace(
        id=run_id,
        inputs=inputs or {},
        outputs={} if outputs is _SETTLED else outputs,
        start_time=datetime.now(timezone.utc) - age,
    )


def test_run_quality_monitor_scores_only_the_sampled_slice(monkeypatch):
    """hash(int) == int in CPython, so integer run ids make the sampling
    boundary deterministic to test: at sample_rate=0.05, only ids whose
    value mod 100 is under 5 pass the filter."""
    runs = [_run(run_id=n) for n in range(10)]
    monkeypatch.setattr(monitor_module.Client, "list_runs", lambda self, **kw: runs)
    monkeypatch.setattr(
        monitor_module, "answer_quality", lambda inputs, outputs: {"score": 0.9}
    )
    scored_ids = []
    monkeypatch.setattr(
        monitor_module.Client, "create_feedback",
        lambda self, run_id, key, score: scored_ids.append(run_id),
    )

    run_quality_monitor(sample_rate=0.05)

    assert scored_ids == [n for n in range(10) if hash(n) % 100 < 5]


def test_run_quality_monitor_passes_the_runs_inputs_and_outputs_to_the_judge(
    monkeypatch,
):
    target_run = _run(run_id=0, inputs={"message": "hi"}, outputs={"reply": "hello"})
    monkeypatch.setattr(
        monitor_module.Client, "list_runs", lambda self, **kw: [target_run]
    )
    seen = {}
    monkeypatch.setattr(
        monitor_module, "answer_quality",
        lambda inputs, outputs: seen.update(inputs=inputs, outputs=outputs) or {
            "score": 1.0
        },
    )
    monkeypatch.setattr(
        monitor_module.Client, "create_feedback", lambda self, **kw: None
    )

    run_quality_monitor(sample_rate=1.0)  # every run passes the filter

    assert seen == {"inputs": {"message": "hi"}, "outputs": {"reply": "hello"}}


def test_run_quality_monitor_writes_the_judges_score_back_as_online_feedback(
    monkeypatch,
):
    target_run = _run(run_id=0)
    monkeypatch.setattr(
        monitor_module.Client, "list_runs", lambda self, **kw: [target_run]
    )
    monkeypatch.setattr(
        monitor_module, "answer_quality", lambda inputs, outputs: {"score": 0.75}
    )
    feedback_calls = []
    monkeypatch.setattr(
        monitor_module.Client, "create_feedback",
        lambda self, run_id, key, score: feedback_calls.append((run_id, key, score)),
    )

    run_quality_monitor(sample_rate=1.0)

    assert feedback_calls == [(0, "online-correctness", 0.75)]


def test_run_quality_monitor_queries_the_tagged_atlas_prod_traces(monkeypatch):
    seen_kwargs = {}
    monkeypatch.setattr(
        monitor_module.Client, "list_runs",
        lambda self, **kw: seen_kwargs.update(kw) or [],
    )

    run_quality_monitor()

    assert seen_kwargs["project_name"] == "atlas-prod"
    assert seen_kwargs["filter"] == 'has(tags, "atlas")'


def test_run_quality_monitor_reads_only_root_runs_inside_its_window(monkeypatch):
    """Chapter 20's tags are inherited by every span, so without is_root=True
    the judge would grade tool and model spans as answers; the start_time
    window keeps each scheduled run from re-scoring runs it already scored."""
    seen_kwargs = {}
    monkeypatch.setattr(
        monitor_module.Client, "list_runs",
        lambda self, **kw: seen_kwargs.update(kw) or [],
    )
    before = datetime.now(timezone.utc)

    run_quality_monitor(window=timedelta(minutes=15))

    after = datetime.now(timezone.utc)
    assert seen_kwargs["is_root"] is True
    assert seen_kwargs["error"] is False  # a run that raised has no answer
    start = seen_kwargs["start_time"]
    # two windows back: the closed window is read, the open one is skipped
    assert before - timedelta(minutes=30) <= start <= after - timedelta(minutes=30)
    assert "limit" not in seen_kwargs  # the window bounds the read, not a count


def test_run_quality_monitor_skips_open_window_and_in_flight_runs(monkeypatch):
    """A run still in flight has outputs=None, and online_judge raises
    KeyError: 'outputs' on it; a run in the current, unfinished window belongs to the
    next tick. Both are skipped, the tick does not abort, and the settled run
    in the closed window is still scored."""
    runs = [
        _run(run_id=0, outputs=None),  # closed window, still in flight
        _run(run_id=1, age=timedelta(minutes=5)),  # open window: next tick's
        _run(run_id=2, outputs={"messages": []}),  # closed window, settled
    ]
    monkeypatch.setattr(monitor_module.Client, "list_runs", lambda self, **kw: runs)

    def judge(inputs, outputs):
        if outputs is None:
            raise KeyError("outputs")  # what openevals does with outputs=None
        return {"score": True}

    monkeypatch.setattr(monitor_module, "answer_quality", judge)
    scored_ids = []
    monkeypatch.setattr(
        monitor_module.Client, "create_feedback",
        lambda self, run_id, key, score: scored_ids.append(run_id),
    )

    run_quality_monitor(sample_rate=1.0)

    assert scored_ids == [2]
