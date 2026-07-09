"""Chapter 2: scripts/check_versions.py - the pinned-environment gate."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_check_versions_passes_against_the_pinned_environment():
    result = subprocess.run(
        [sys.executable, "scripts/check_versions.py"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0
    assert "Environment matches the version matrix." in result.stdout
