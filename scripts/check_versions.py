"""Fail loudly if the installed environment drifts from the book's version matrix.

See Chapter 2, "Pinning the environment". Wire this as the first CI step, before
the test suite - it turns the tested-version matrix into a gate instead of a claim.
"""

import sys
from importlib.metadata import version

EXPECTED = {
    "langgraph": "1.2.6",
    "langchain": "1.3.0",
}


def main() -> int:
    mismatched = [
        f"{pkg}: expected {want}, got {version(pkg)}"
        for pkg, want in EXPECTED.items()
        if version(pkg) != want
    ]
    if mismatched:
        print("Version mismatch:\n  " + "\n  ".join(mismatched))
        return 1
    print("Environment matches the version matrix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
