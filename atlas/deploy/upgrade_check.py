"""Chapter 26, "The Frontier and Future-Proofing" - the upgrade-and-currency
playbook.

See "The upgrade-and-currency playbook". Chapter 2's `scripts/check_versions.py`
already asserts the installed environment matches the pin on every CI run - the
has-it-drifted check. Chapter 21's eval suite already catches a regression in
Atlas's own code. `check_candidate_upgrade` is neither of those - it points
Chapter 21's UNCHANGED regression suite at a framework-version-bump target
instead of an Atlas-code-change target, so the same machinery that catches
Atlas breaking itself also catches a candidate LangGraph/LangChain release
breaking Atlas, before anyone adopts it in production.

Run this against an isolated environment - a staging deploy, a throwaway
container - never against the environment production traffic depends on.
`tests/test_upgrade_check.py` never lets `subprocess.run` or `evaluate` touch
a real environment or a live LangSmith project; it monkeypatches both and
checks the call shape and the pass/fail interpretation instead."""

import subprocess

from langsmith.evaluation import evaluate

from atlas.evals import ALL_EVALUATORS, run_atlas  # Chapter 21, unchanged


def check_candidate_upgrade(package: str, candidate_version: str) -> bool:
    """Install a candidate version in an isolated env, run the SAME
    regression suite Chapter 21 built for Atlas's own code, and report
    whether it still passes - before anyone adopts the upgrade."""
    subprocess.run(  # <1>
        ["uv", "pip", "install", f"{package}=={candidate_version}"],
        check=True,
    )
    results = evaluate(
        run_atlas,
        data="atlas-regression",
        evaluators=ALL_EVALUATORS,
        experiment_prefix=f"upgrade-check-{package}-{candidate_version}",
    )
    return results.summary_results.get("failures", 1) == 0  # <2>
