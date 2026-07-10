"""Chapter 26, "The Frontier and Future-Proofing" - atlas/deploy/upgrade_check.py.

See "The upgrade-and-currency playbook". `check_candidate_upgrade` installs a
candidate package version via `uv pip install` and runs Chapter 21's
UNCHANGED regression suite against it via a live LangSmith `evaluate()` call
- both are real side effects (mutating the active environment, writing an
experiment to LangSmith), so - matching the external-service exception used
throughout this repo for Postgres/LangSmith/the Agent Server - the tests
below monkeypatch both `subprocess.run` and `evaluate` and check the call
shape and the pass/fail interpretation, never a live install or a live
LangSmith project."""

from types import SimpleNamespace

from atlas.deploy import upgrade_check as upgrade_check_module
from atlas.deploy.upgrade_check import check_candidate_upgrade
from atlas.evals import ALL_EVALUATORS, run_atlas


def _stub_subprocess(monkeypatch, calls):
    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(upgrade_check_module.subprocess, "run", fake_run)


def test_check_candidate_upgrade_installs_the_exact_pinned_candidate(monkeypatch):
    install_calls: list[dict] = []
    _stub_subprocess(monkeypatch, install_calls)
    monkeypatch.setattr(
        upgrade_check_module,
        "evaluate",
        lambda *a, **k: SimpleNamespace(summary_results={"failures": 0}),
    )

    check_candidate_upgrade("langgraph", "1.3.0")

    assert install_calls == [
        {
            "cmd": ["uv", "pip", "install", "langgraph==1.3.0"],
            "kwargs": {"check": True},
        }
    ]


def test_check_candidate_upgrade_points_the_unchanged_regression_suite_at_it(
    monkeypatch,
):
    _stub_subprocess(monkeypatch, [])
    evaluate_calls: list[dict] = []

    def fake_evaluate(target, **kwargs):
        evaluate_calls.append({"target": target, **kwargs})
        return SimpleNamespace(summary_results={"failures": 0})

    monkeypatch.setattr(upgrade_check_module, "evaluate", fake_evaluate)

    check_candidate_upgrade("langgraph", "1.3.0")

    assert len(evaluate_calls) == 1
    call = evaluate_calls[0]
    assert call["target"] is run_atlas
    assert call["data"] == "atlas-regression"
    assert call["evaluators"] == ALL_EVALUATORS
    assert call["experiment_prefix"] == "upgrade-check-langgraph-1.3.0"


def test_check_candidate_upgrade_returns_true_when_the_suite_has_no_failures(
    monkeypatch,
):
    _stub_subprocess(monkeypatch, [])
    monkeypatch.setattr(
        upgrade_check_module,
        "evaluate",
        lambda *a, **k: SimpleNamespace(summary_results={"failures": 0}),
    )

    assert check_candidate_upgrade("langchain", "1.4.0") is True


def test_check_candidate_upgrade_returns_false_when_the_suite_has_failures(
    monkeypatch,
):
    _stub_subprocess(monkeypatch, [])
    monkeypatch.setattr(
        upgrade_check_module,
        "evaluate",
        lambda *a, **k: SimpleNamespace(summary_results={"failures": 2}),
    )

    assert check_candidate_upgrade("langchain", "1.4.0") is False


def test_check_candidate_upgrade_defaults_to_failing_closed_if_the_field_is_missing(
    monkeypatch,
):
    """The chapter's own annotation flags `summary_results`'s exact shape as
    illustrative, not a verified API contract across `evaluate()` versions -
    the `.get("failures", 1)` default means an unrecognized result shape is
    treated as a failure, not a silent pass."""
    _stub_subprocess(monkeypatch, [])
    monkeypatch.setattr(
        upgrade_check_module,
        "evaluate",
        lambda *a, **k: SimpleNamespace(summary_results={}),
    )

    assert check_candidate_upgrade("langchain", "1.4.0") is False

