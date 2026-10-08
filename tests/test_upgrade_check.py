"""Chapter 26, "The Frontier and Future-Proofing" - atlas/deploy/upgrade_check.py.

See "The upgrade-and-currency playbook". `check_candidate_upgrade` gives the
candidate an environment of its own: a temporary copy of pyproject.toml and
uv.lock, checked with `uv lock --locked`, re-locked with the candidate pinned
(`uv add`), and Chapter 21's UNCHANGED regression suite run there with `uv run
--locked`. The verdict is the exit status. The unit tests stub `subprocess.run`
and check every step: the commands, that each targets the temporary project
and never the repo, that the first failure stops the check and fails it, and
that the package and version are validated before anything runs.

The opt-in tests (ATLAS_UV_CANDIDATE_TEST=1) call `check_candidate_upgrade`
for real, offline from uv's cache (UV_OFFLINE=1): the candidate runs alone in
its environment; a candidate the pins cannot resolve is refused; a stale lock
is refused; the real suite fails closed without LangSmith; and the overlay the
check no longer uses is shown masking a module the candidate does not have.

The pytest `filterwarnings` policy in pyproject.toml is pinned here too:
a framework deprecation fails a test, a beta warning does not, and the one
dated third-party ignore covers only the module it names."""

import importlib.metadata
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core._api.beta_decorator import LangChainBetaWarning
from langchain_core._api.deprecation import LangChainDeprecationWarning
from langgraph.warnings import LangGraphDeprecatedSinceV10, LangGraphDeprecationWarning

from atlas.deploy import upgrade_check as upgrade_check_module
from atlas.deploy.upgrade_check import (
    EVAL_COMMAND,
    PROJECT_ROOT,
    candidate_requirement,
    check_candidate_upgrade,
)


def _stub_subprocess(monkeypatch, calls: list[dict], returncodes=()) -> None:
    """Record each step, and what the temporary project held when it ran."""
    codes = list(returncodes)

    def fake_run(cmd, **kwargs):
        project = Path(cmd[cmd.index("--project") + 1])
        calls.append(
            {
                "cmd": cmd,
                "kwargs": kwargs,
                "project": project,
                "files": sorted(p.name for p in project.iterdir()),
                "lock": (project / "uv.lock").read_bytes(),
            }
        )
        return SimpleNamespace(returncode=codes.pop(0) if codes else 0)

    monkeypatch.setattr(upgrade_check_module.subprocess, "run", fake_run)


def test_the_candidate_is_locked_and_run_in_a_temporary_project_of_its_own(
    monkeypatch,
):
    calls: list[dict] = []
    _stub_subprocess(monkeypatch, calls)

    assert check_candidate_upgrade("langgraph", "1.3.0") is True

    project = str(calls[0]["project"])
    assert [c["cmd"] for c in calls] == [
        ["uv", "lock", "--locked", "--project", project],
        ["uv", "add", "--no-sync", "--project", project, "langgraph==1.3.0"],
        ["uv", "run", "--locked", "--project", project, *EVAL_COMMAND],
    ]
    assert all(c["kwargs"] == {"cwd": PROJECT_ROOT} for c in calls)
    assert Path(project) != PROJECT_ROOT
    assert {"pyproject.toml", "uv.lock"} <= set(calls[0]["files"])
    assert calls[0]["lock"] == (PROJECT_ROOT / "uv.lock").read_bytes()
    assert not Path(project).exists()  # removed afterwards


def test_nothing_is_installed_into_the_running_environment_or_the_repo(monkeypatch):
    calls: list[dict] = []
    _stub_subprocess(monkeypatch, calls)
    lock_before = (PROJECT_ROOT / "uv.lock").read_bytes()

    check_candidate_upgrade("langchain", "1.4.0")

    for call in calls:
        assert "pip" not in call["cmd"] and "sync" not in call["cmd"]
        assert call["project"] != PROJECT_ROOT
    assert (PROJECT_ROOT / "uv.lock").read_bytes() == lock_before


def test_the_project_root_holds_the_files_the_candidate_project_copies():
    assert (PROJECT_ROOT / "pyproject.toml").is_file()
    assert (PROJECT_ROOT / "uv.lock").is_file()


@pytest.mark.parametrize("failing_step", [0, 1, 2])
@pytest.mark.parametrize("returncode", [1, 2, -9])
def test_any_failing_step_means_do_not_upgrade(monkeypatch, failing_step, returncode):
    """A stale lock (step 0), a candidate the pins cannot resolve (step 1),
    gate() finding a regression or anything else going wrong in the run
    (step 2): each fails closed, and nothing after it runs."""
    calls: list[dict] = []
    _stub_subprocess(monkeypatch, calls, [0] * failing_step + [returncode])

    assert check_candidate_upgrade("langchain", "1.4.0") is False
    assert len(calls) == failing_step + 1


@pytest.mark.parametrize(
    "package",
    ["requests", "langgraph ", "-e", "langgraph[all]", "git+https://x/langgraph"],
)
def test_a_package_outside_the_allow_list_is_refused(monkeypatch, package):
    calls: list[dict] = []
    _stub_subprocess(monkeypatch, calls)

    with pytest.raises(ValueError, match="not a framework package"):
        check_candidate_upgrade(package, "1.3.0")
    assert calls == []


@pytest.mark.parametrize(
    "version",
    [
        "",
        "latest",
        "1.3.0; python_version > '3'",  # an environment marker
        "1.3.0,langchain==0.1",  # a second requirement
        "1.3.0 @ https://evil.example/langgraph.whl",  # a direct URL
        "--index-url=https://evil.example",  # an option
        "1.3.0+local",  # a local version: not a published release
        "1.3.0\n",
        ">=1.3",
    ],
)
def test_a_version_that_is_not_pep_440_is_refused(monkeypatch, version):
    calls: list[dict] = []
    _stub_subprocess(monkeypatch, calls)

    with pytest.raises(ValueError, match="not a PEP 440 version"):
        check_candidate_upgrade("langgraph", version)
    assert calls == []


@pytest.mark.parametrize(
    "version", ["1.3.0", "2.0", "1.3.0a1", "1.3.0rc2", "1.3.0.post1", "1.3.0.dev4"]
)
def test_a_pep_440_release_or_prerelease_is_accepted(version):
    assert candidate_requirement("langgraph", version) == f"langgraph=={version}"


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


# --- Opt-in: the real check, offline from uv's cache -------------------------

real_uv = pytest.mark.skipif(
    not os.environ.get("ATLAS_UV_CANDIDATE_TEST"),
    reason="opt-in: builds real uv environments (ATLAS_UV_CANDIDATE_TEST=1)",
)
CANDIDATE = os.environ.get("ATLAS_UV_CANDIDATE_VERSION", "1.2.8")  # a cached wheel
NO_OP = ("python", "-c", "raise SystemExit(0)")


@pytest.fixture
def offline_uv(monkeypatch):
    monkeypatch.setenv("UV_OFFLINE", "1")
    lock_before = (PROJECT_ROOT / "uv.lock").read_bytes()
    yield
    assert (PROJECT_ROOT / "uv.lock").read_bytes() == lock_before
    assert importlib.metadata.version("langgraph") == "1.2.6"


@real_uv
def test_the_candidate_runs_alone_in_its_own_environment(offline_uv):
    """One `langgraph` on the child's path, the candidate's: a module the
    candidate does not ship has no pinned layer to be imported from."""
    probe = "\n".join(
        [
            "import importlib.metadata as m, langgraph, pathlib, sys",
            f"assert m.version('langgraph') == {CANDIDATE!r}",
            "[path] = langgraph.__path__",
            "assert pathlib.Path(path).is_relative_to(sys.prefix)",
            "others = [p for p in sys.path if p",
            "          and pathlib.Path(p, 'langgraph').is_dir()",
            "          and not pathlib.Path(p).is_relative_to(sys.prefix)]",
            "assert others == [], others",
        ]
    )
    command = ("python", "-c", probe)

    assert check_candidate_upgrade("langgraph", CANDIDATE, command=command) is True


@real_uv
def test_a_candidate_the_pins_cannot_resolve_is_refused(offline_uv):
    """langgraph 1.0.8 is cached, but langchain 1.3.0 needs langgraph>=1.2:
    no environment is built, nothing runs, and the verdict is "do not
    upgrade" - not a 1.0.8 layered over 1.2.6."""
    assert check_candidate_upgrade("langgraph", "1.0.8", command=NO_OP) is False


@real_uv
def test_a_stale_lock_is_refused_not_re_resolved(offline_uv, monkeypatch, tmp_path):
    for name in ("pyproject.toml", "uv.lock", ".python-version"):
        if (PROJECT_ROOT / name).is_file():
            shutil.copy2(PROJECT_ROOT / name, tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    text = pyproject.read_text()
    pyproject.write_text(text.replace('"mcp>=1.28.1",', '"mcp>=1.28.1", "six",'))
    monkeypatch.setattr(upgrade_check_module, "PROJECT_ROOT", tmp_path)
    codes: list[int] = []
    real_run = subprocess.run

    def recording_run(cmd, **kwargs):
        proc = real_run(cmd, **kwargs)
        codes.append(proc.returncode)
        return proc

    monkeypatch.setattr(upgrade_check_module.subprocess, "run", recording_run)

    assert check_candidate_upgrade("langgraph", CANDIDATE, command=NO_OP) is False
    assert len(codes) == 1 and codes[0] != 0  # `uv lock --locked` refused
    repo_lock = (PROJECT_ROOT / "uv.lock").read_bytes()
    assert (tmp_path / "uv.lock").read_bytes() == repo_lock


@real_uv
def test_the_real_suite_fails_closed_without_langsmith(offline_uv, monkeypatch):
    """`python -m atlas.evals` in the candidate's environment, with LangSmith
    unreachable: gate() never gets a verdict, so the check says no."""
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("LANGSMITH_API_KEY", "")

    assert check_candidate_upgrade("langgraph", CANDIDATE) is False


@real_uv
def test_the_overlay_this_check_does_not_use_would_mask_a_missing_module():
    """Why there is no `--with`: langgraph 1.0.8 has no `langgraph.callbacks`,
    yet layered over the pinned 1.2.6 environment it still imports one, from
    the pin, because `langgraph` is a namespace package."""
    code = (
        "import importlib.metadata as m, langgraph.callbacks as c; "
        "print(m.version('langgraph'), c.__file__)"
    )
    out = subprocess.run(
        ["uv", "run", "--offline", "--isolated", "--with", "langgraph==1.0.8",
         "python", "-c", code],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
    )
    version, origin = out.stdout.strip().splitlines()[-1].split(" ", 1)
    assert version == "1.0.8"  # the overlay is what the child reports...
    assert "archive-v0" not in origin  # ...but callbacks is the pinned layer's


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
