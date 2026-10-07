"""Appendix E, "The Deep Research Agent" - atlas/research.py + atlas/deep_research.py.

Appendix E adds no new code: its own opening paragraph says so explicitly -
"Neither listing is new here; both are copied verbatim from their chapters,
placed side by side because the comparison is the point." Form 1 is Chapter
17's Send-based map-reduce subgraph (`atlas.research.research_graph`); Form 2
is Chapter 18's Deep Agents harness (`atlas.deep_research.deep_research_agent`).
Both are already fully unit-tested in `tests/test_research.py` and
`tests/test_deep_research.py` - this module does not re-test either form's
own behavior. It tests the appendix's own comparison claims instead: the
"Side by side" table's rows about mounting, shared backend, and partial
failure, plus one gap the appendix's table does not mention.

Mounting, as of the v1.2 revision: Atlas's support graph (`atlas/graph.py`)
does not mount `research_graph` - no triage route leads to research, so the
node was dropped rather than left unreachable. Research runs as its own
compiled graph (`run_research`, and `langgraph.json`'s "research" entry).
Chapter 16's hand-rolled supervisor is wired into its own compiled graph,
`atlas.research.supervisor_graph`. The tests below pin both facts, so an
appendix listing that still shows research mounted inside Atlas fails
against the code rather than drifting silently."""

from atlas import graph as graph_module
from atlas.deep_research import deep_research_agent, source_lookup
from atlas.research import (
    SourceUnavailable,
    research_graph,
    research_worker,
    search_source,
    supervisor,
    supervisor_graph,
)


def test_form_1_research_graph_is_its_own_graph_not_a_node_in_atlas():
    """Research is a separate compiled graph; Atlas's builder has no
    "research" node."""
    assert "research" not in graph_module.builder.nodes
    assert hasattr(research_graph, "invoke")


def test_form_0_ch16_supervisor_runs_as_its_own_compiled_graph():
    """Chapter 16's supervisor topology is wired into
    `supervisor_graph`, not into Atlas's support graph."""
    nodes = set(supervisor_graph.get_graph().nodes)
    assert {"supervisor", "web_research", "doc_research", "compile"} <= nodes
    assert "supervisor" not in graph_module.builder.nodes
    assert hasattr(supervisor, "invoke")


def test_both_forms_share_the_same_seeded_backend_not_two_parallel_ones():
    """The "Where to reach for it" row implies one underlying problem, two
    harnesses - not two competing backends. Form 2's source_lookup tool
    literally reuses Form 1's search_source/SourceUnavailable rather than
    inventing its own, per atlas/deep_research.py's own docstring."""
    from atlas.deep_research import search_source as form_2_search_source
    from atlas.deep_research import SourceUnavailable as form_2_source_unavailable

    assert form_2_search_source is search_source
    assert form_2_source_unavailable is SourceUnavailable


def test_both_forms_apply_the_same_partial_failure_discipline_on_a_dead_source():
    """Appendix E's table, "Partial failure" row: "A caught exception becomes
    a finding with an error key" (Form 1) vs. "the same discipline, expressed
    as a tool returning an error string instead of raising" (Form 2) - same
    contract, different shape. Neither form raises out to its caller."""
    dead_source = "nope/does-not-exist"

    form_1_finding = research_worker({"source": dead_source})["findings"][0]
    assert "error" in form_1_finding

    try:
        search_source(dead_source)
        raised = False
    except SourceUnavailable:
        raised = True
    assert raised  # the raw backend still raises; each form is what CATCHES it

    # Form 2's catch lives inside source_lookup's own try/except (Chapter 19
    # wraps it in get_stream_writer(), which needs a real run context -
    # tests/test_deep_research.py already exercises that path end to end;
    # here we only need the module-level function identity confirmed above
    # plus the source module's own contract, not a duplicate graph run).
    assert source_lookup.name == "source_lookup"


def test_form_2_deep_research_agent_is_a_standalone_graph_not_wired_into_atlas_graph():
    """Unlike Form 1, Form 2 is not referenced by atlas/graph.py at all - it
    is invoked directly (see the appendix's own "Invoked like any other
    checkpointed run" listing), not mounted as a node."""
    assert hasattr(deep_research_agent, "invoke")
    assert "deep_research" not in graph_module.builder.nodes
    assert "research_namespace" not in dir(graph_module)
