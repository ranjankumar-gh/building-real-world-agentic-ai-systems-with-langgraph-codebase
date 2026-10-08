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
it already imported, whatever is on disk afterwards. `uv run --isolated` builds
a fresh, throwaway project environment from `uv.lock` instead of reusing
`.venv`, and `--with` layers the candidate on top of it in a second ephemeral
environment that takes precedence (uv allows it to conflict with the lock).
Nothing is installed into the developer's environment. The child runs
`python -m atlas.evals`, Chapter 21's CI entry point, whose exit status is
`gate()`'s verdict: 0 only if no run raised and no evaluator scored an example
False or 0. Any other status, including a failure to build the environment or
to reach LangSmith, reads as "do not upgrade".

Run it from CI on a schedule, on a throwaway runner, not from the Agent Server.
`tests/test_upgrade_check.py` stubs `subprocess.run` and checks the command
and the verdict; an opt-in test runs the real overlay from uv's cache."""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # pyproject.toml and uv.lock


def check_candidate_upgrade(package: str, candidate_version: str) -> bool:
    """Run the SAME regression suite Chapter 21 built for Atlas's own code
    against a candidate version, in a throwaway environment, and report
    whether it still passes - before anyone adopts the upgrade."""
    proc = subprocess.run(
        [
            "uv", "run", "--isolated",
            "--with", f"{package}=={candidate_version}",
            "python", "-m", "atlas.evals",  # Chapter 21's CI entry point
        ],
        cwd=PROJECT_ROOT,
    )
    return proc.returncode == 0
