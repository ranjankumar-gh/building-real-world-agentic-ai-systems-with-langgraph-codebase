"""The pip route in the book's "Using the code" (`pip install -r
requirements.txt`) has to install what `uv sync` installs from `uv.lock`.

requirements.txt is generated, never edited by hand:

    uv export --format requirements-txt --no-hashes --frozen --no-emit-project

This test fails when the lock moves and the export was not re-run: every
package pinned in uv.lock (the project itself aside) must appear in
requirements.txt at the same version, and nothing else may."""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked() -> dict[str, str]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    return {
        _norm(p["name"]): p["version"]
        for p in lock["package"]
        if "virtual" not in p.get("source", {})  # the atlas project itself
    }


def _required() -> dict[str, str]:
    lines = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {
        _norm(m.group(1)): m.group(2)
        for line in lines
        if (m := PIN.match(line.strip()))
    }


def test_requirements_txt_pins_exactly_what_uv_lock_pins():
    assert _required() == _locked()


def test_requirements_txt_carries_every_direct_dependency():
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    direct = {
        _norm(re.split(r"[<>=!~\[ ;]", dep, maxsplit=1)[0])
        for dep in project["project"]["dependencies"]
    }
    assert direct <= set(_required())
