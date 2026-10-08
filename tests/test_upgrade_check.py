"""Chapter 26, "The Frontier and Future-Proofing" - atlas/deploy/upgrade_check.py.

See "The upgrade-and-currency playbook". `check_candidate_upgrade` runs
Chapter 21's UNCHANGED regression suite in a child process, `uv run
--isolated --with <package>==<version> python -m atlas.evals`, and reads the
verdict from its exit status. The unit tests stub `subprocess.run`: a real
run builds an environment and writes an experiment to LangSmith. They check
the command (a child process, an isolated environment, no install into this
one) and the verdict.

`test_the_overlay_runs_the_candidate_not_the_pin` is opt-in
(ATLAS_UV_OVERLAY_TEST=1): it runs the real `uv run --offline --isolated
--with` overlay against a langgraph version already in uv's cache and checks
the child imports the candidate while this environment keeps the pin.

The pytest `filterwarnings` policy in pyproject.toml is pinned here too:
a framework deprecation fails a test, a beta warning does not, and the one
dated third-party ignore covers only the module it names."""

import importlib.metadata
import os
import subprocess
import sys
import warnings
from types import SimpleNamespace

import pytest
from langchain_core._api.beta_decorator import LangChainBetaWarning
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langgraph.warnings import LangGraphDeprecatedSinceV10, LangGraphDeprecationWarning

from atlas.deploy import upgrade_check as upgrade_check_module
from atlas.deploy.upgrade_check import PROJECT_ROOT, check_candidate_upgrade


def _stub_subprocess(monkeypatch, calls: list[dict], returncode: int = 0) -> None:
    def fake_run(cmd, **kwargs):
        calls.append({"cmd": cmd, "kwargs": kwargs})
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(upgrade_check_module.subprocess, "run", fake_run)


def test_the_suite_runs_in_a_child_with_the_candidate_layered_on_an_isolated_env(
    monkeypatch,
):
    calls: list[dict] = []
    _stub_subprocess(monkeypatch, calls)

    check_candidate_upgrade("langgraph", "1.3.0")

    assert calls == [
        {
            "cmd": [
                "uv", "run", "--isolated",
                "--with", "langgraph==1.3.0",
                "python", "-m", "atlas.evals",
            ],
            "kwargs": {"cwd": PROJECT_ROOT},
        }
    ]


def test_nothing_is_installed_into_the_running_environment(monkeypatch):
    calls: list[dict] = []
    _stub_subprocess(monkeypatch, calls)

    check_candidate_upgrade("langchain", "1.4.0")

    cmd = calls[0]["cmd"]
    assert "pip" not in cmd and "install" not in cmd and "sync" not in cmd


def test_the_project_root_holds_the_lock_the_isolated_env_is_built_from():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "uv.lock").is_file()


def test_a_zero_exit_from_the_gate_means_the_candidate_passes(monkeypatch):
    _stub_subprocess(monkeypatch, [], returncode=0)

    assert check_candidate_upgrade("langchain", "1.4.0") is True


@pytest.mark.parametrize("returncode", [1, 2, -9])
def test_any_other_exit_means_do_not_upgrade(monkeypatch, returncode):
    """1 is gate() finding a regression; anything else (a resolver failure,
    an unreachable LangSmith, a killed process) also fails closed."""
    _stub_subprocess(monkeypatch, [], returncode=returncode)

    assert check_candidate_upgrade("langchain", "1.4.0") is False


def test_importing_the_module_loads_no_framework_code():
    """The check never needs the candidate in this process, so the module
    imports nothing that would pin the old version in memory."""
    code = (
        "import sys, atlas.deploy.upgrade_check; "
        "print(any(m.split('.')[0] in ('langgraph', 'langchain', 'langsmith') "
        "for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "False"


@pytest.mark.skipif(
    not os.environ.get("ATLAS_UV_OVERLAY_TEST"),
    reason="opt-in: builds a real uv environment (ATLAS_UV_OVERLAY_TEST=1)",
)
def test_the_overlay_runs_the_candidate_not_the_pin():
    """Offline, from uv's cache: the child sees the candidate, this process
    keeps the pin. Set ATLAS_UV_OVERLAY_VERSION to a cached version."""
    candidate = os.environ.get("ATLAS_UV_OVERLAY_VERSION", "1.2.8")
    code = "import importlib.metadata as m; print(m.version('langgraph'))"
    out = subprocess.run(
        [
            "uv", "run", "--offline", "--isolated",
            "--with", f"langgraph=={candidate}",
            "python", "-c", code,
        ],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == candidate
    assert importlib.metadata.version("langgraph") == "1.2.6"


# --- The warnings policy (pyproject.toml [tool.pytest.ini_options]) --------


@pytest.mark.parametrize(
    "warning",
    [
        LangGraphDeprecatedSinceV10("deprecated in this test"),
        LangChainDeprecationWarning("deprecated in this test"),
    ],
)
def test_a_framework_deprecation_fails_the_test_run(warning):
    with pytest.raises(type(warning)):
        warnings.warn(warning)


def test_a_beta_warning_does_not_fail_the_test_run():
    assert not issubclass(LangChainBetaWarning, LangChainDeprecationWarning)
    with warnings.catch_warnings(record=True):
        warnings.warn("beta in this test", LangChainBetaWarning)


def test_the_dated_ignore_covers_only_the_module_it_names():
    """The real warning: trustcall imports Send from its pre-1.0 path."""
    send = LangGraphDeprecatedSinceV10("Importing Send from langgraph.constants")
    warnings.warn_explicit(send, None, "_base.py", 46, module="trustcall._base")
    with pytest.raises(LangGraphDeprecationWarning):
        warnings.warn_explicit(send, None, "graph.py", 1, module="atlas.graph")
