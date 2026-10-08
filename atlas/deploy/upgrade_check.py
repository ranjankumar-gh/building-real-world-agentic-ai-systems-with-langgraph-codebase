"""Chapter 26, "The Frontier and Future-Proofing" - the upgrade-and-currency
playbook.

See "The upgrade-and-currency playbook". Chapter 2's `scripts/check_versions.py`
asserts the installed environment matches the pin - the has-it-drifted check.
Chapter 21's eval suite catches a regression in Atlas's own code.
`check_candidate_upgrade` points Chapter 21's UNCHANGED regression suite at a
framework-version-bump target instead of an Atlas-code-change target, so the
same machinery that catches Atlas breaking itself also catches a candidate
LangGraph/LangChain release breaking Atlas, before anyone adopts it.

The suite runs in a child process, never in this one. Installing the candidate
and importing it here would not work: a running interpreter keeps the version
it already imported, whatever is on disk afterwards. Nor is the candidate
layered over the pinned environment (`uv run --with`): `langgraph` is a
namespace package split across several distributions, so an overlay imports
the candidate's modules where it has them and the pinned ones where it does
not, and a module the candidate deleted would still import. The candidate gets
an environment of its own instead. A copy of `pyproject.toml` and `uv.lock` in
a temporary directory is checked against itself (`uv lock --locked`: a stale
lock fails rather than being quietly re-resolved), then re-locked with the
candidate pinned (`uv add`), and the suite runs there with `uv run --locked`.
The repo's own `uv.lock` and `.venv` are never written. The child runs
`python -m atlas.evals`, Chapter 21's CI entry point, from the repo root, so
it imports this checkout's `atlas`; its exit status is `gate()`'s verdict: 0
only if no run raised and no evaluator scored an example False or 0. Any other
status, at any step, including a resolver failure or an unreachable
LangSmith, reads as "do not upgrade".

The candidate's name must be one of the framework packages Atlas pins, and its
version a PEP 440 public version: the two strings become a requirement uv
parses, and nothing else may ride along in them (a marker, a URL, an option).

Run it from CI on a schedule, on a throwaway runner, not from the Agent Server.
`tests/test_upgrade_check.py` stubs `subprocess.run` to check every step, and
opt-in tests run the real thing from uv's cache."""

import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # pyproject.toml and uv.lock
PROJECT_FILES = ("pyproject.toml", "uv.lock", ".python-version")
EVAL_COMMAND = ("python", "-m", "atlas.evals")  # Chapter 21's CI entry point
CANDIDATE_PACKAGES = frozenset({
    "deepagents", "langchain", "langchain-anthropic", "langchain-core",
    "langgraph", "langgraph-checkpoint", "langgraph-checkpoint-postgres",
    "langgraph-prebuilt", "langgraph-sdk", "langmem", "langsmith",
})
PEP440_PUBLIC = re.compile(  # N[.N]*[{a|b|rc}N][.postN][.devN], no local part
    r"(?:[1-9][0-9]*!)?(?:0|[1-9][0-9]*)(?:\.(?:0|[1-9][0-9]*))*"
    r"(?:(?:a|b|rc)(?:0|[1-9][0-9]*))?(?:\.post(?:0|[1-9][0-9]*))?"
    r"(?:\.dev(?:0|[1-9][0-9]*))?"
)


def candidate_requirement(package: str, candidate_version: str) -> str:
    """`package==version`, or ValueError if either is not what it claims."""
    if package not in CANDIDATE_PACKAGES:
        raise ValueError(f"not a framework package Atlas pins: {package!r}")
    if not PEP440_PUBLIC.fullmatch(candidate_version):
        raise ValueError(f"not a PEP 440 version: {candidate_version!r}")
    return f"{package}=={candidate_version}"


def check_candidate_upgrade(
    package: str,
    candidate_version: str,
    command: Sequence[str] = EVAL_COMMAND,
) -> bool:
    """Run the SAME regression suite Chapter 21 built for Atlas's own code
    against a candidate version, in an environment resolved for it alone,
    and report whether it still passes - before anyone adopts the upgrade."""
    requirement = candidate_requirement(package, candidate_version)
    with tempfile.TemporaryDirectory(
        prefix="atlas-candidate-", ignore_cleanup_errors=True
    ) as project:
        for name in PROJECT_FILES:
            if (PROJECT_ROOT / name).is_file():
                shutil.copy2(PROJECT_ROOT / name, project)
        steps = [
            ["uv", "lock", "--locked", "--project", project],
            ["uv", "add", "--no-sync", "--project", project, requirement],
            ["uv", "run", "--locked", "--project", project, *command],
        ]
        for step in steps:
            if subprocess.run(step, cwd=PROJECT_ROOT).returncode != 0:
                return False
    return True
