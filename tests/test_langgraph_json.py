"""Chapter 22, "Deployment and Scaling" - langgraph.json.

See "Packaging Atlas for the Agent Server". `langgraph.json` is the one file
the CLI needs to build, run, and deploy Atlas - it maps graph names to
compiled objects. This test does not need a live `langgraph up` server (that
part of the chapter is skip-guarded elsewhere); it only checks the config
file's own shape and that every `graphs` entry actually resolves to a real,
importable, compiled graph - the same kind of check that caught Chapter 21's
research-graph mismatch (the eval suite's `run_atlas` target function
assuming a `messages`/`handoffs` shape that `atlas/research.py`'s actual
compiled `research_graph` - the Chapter 17 Send-based map-reduce pipeline -
does not have)."""

import importlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_config() -> dict:
    return json.loads((REPO_ROOT / "langgraph.json").read_text())


def test_langgraph_json_lives_at_the_repo_root_not_inside_atlas():
    assert (REPO_ROOT / "langgraph.json").is_file()
    assert not (REPO_ROOT / "atlas" / "langgraph.json").exists()


def test_langgraph_json_is_valid_json_with_the_expected_top_level_keys():
    config = _load_config()
    assert config["dependencies"] == ["."]
    assert config["env"] == ".env"
    assert config["python_version"] == "3.12"


def test_langgraph_json_maps_both_of_atlass_graphs():
    config = _load_config()
    assert config["graphs"] == {
        "resolve": "./atlas/graph.py:graph",
        "research": "./atlas/research.py:research_graph",
    }


def test_every_graphs_entry_resolves_to_a_real_compiled_graph():
    """Each value is '<path>:<attr>' - confirm the module imports and the
    named attribute exists and is actually a compiled (invoke-able) graph,
    not a stale or renamed reference."""
    config = _load_config()
    for name, target in config["graphs"].items():
        path, attr = target.split(":")
        module_name = "atlas." + Path(path).stem  # "./atlas/graph.py" -> "atlas.graph"
        module = importlib.import_module(module_name)
        graph_obj = getattr(module, attr, None)
        assert graph_obj is not None, f"{name}: {target} has no such attribute"
        assert hasattr(graph_obj, "invoke"), f"{name}: {target} is not a compiled graph"
