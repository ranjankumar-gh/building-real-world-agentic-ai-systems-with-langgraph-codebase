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


def test_langgraph_json_serves_the_resolved_graph_research_and_the_monitor():
    """`resolve` is the production assembly (the mounted agent and its gates),
    not atlas/graph.py's model-free `graph`; `monitor` is the graph Chapter
    22's cron targets; `sla-watch` is the graph Chapter 27's cron targets."""
    config = _load_config()
    assert config["graphs"] == {
        "resolve": "./atlas/deploy/server.py:resolve",
        "research": "./atlas/research.py:research_graph",
        "monitor": "./atlas/deploy/server.py:monitor",
        "sla-watch": "./atlas/deploy/server.py:sla_watch",
    }


def test_no_served_graph_brings_its_own_checkpointer_or_store():
    """`langgraph dev` refuses a graph compiled with either ("Heads up! Your
    graph ... includes a custom checkpointer"), and under `langgraph up` the
    server's Postgres replaces them anyway."""
    for name, graph_obj in _served_graphs():
        assert graph_obj.checkpointer is None, name
        assert graph_obj.store is None, name


def _served_graphs() -> list[tuple[str, object]]:
    found = []
    for name, target in _load_config()["graphs"].items():
        path, attr = target.split(":")
        # "./atlas/deploy/server.py" -> "atlas.deploy.server"
        module_name = ".".join(Path(path).with_suffix("").parts)
        module = importlib.import_module(module_name)
        found.append((name, getattr(module, attr, None)))
    return found


def test_every_graphs_entry_resolves_to_a_real_compiled_graph():
    """Each value is '<path>:<attr>' - confirm the module imports and the
    named attribute exists and is actually a compiled (invoke-able) graph,
    not a stale or renamed reference."""
    for name, graph_obj in _served_graphs():
        assert graph_obj is not None, f"{name}: no such attribute"
        assert hasattr(graph_obj, "invoke"), f"{name}: not a compiled graph"


def test_the_served_resolve_graph_mounts_the_agent_not_the_stub():
    from atlas.graph import answer

    resolve = dict(_served_graphs())["resolve"]
    assert resolve.builder.nodes["answer"].runnable.func is not answer
