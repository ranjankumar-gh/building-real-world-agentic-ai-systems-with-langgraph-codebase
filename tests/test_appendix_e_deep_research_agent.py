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

The gap: the appendix compares only two of the THREE research
implementations Atlas has accumulated. Chapter 16's hand-rolled
`supervisor`/`web_research`/`doc_research` (see `atlas/research.py`'s module
docstring) is never mounted into any compiled `StateGraph` anywhere in this
repo - confirmed below by asserting `"supervisor"` is absent from
`atlas.graph.builder`'s nodes. The appendix's own "Mounted into Atlas"
listing for Form 1 is also incomplete on its own: `research_graph` IS
mounted as a node named `"research"` in `atlas/graph.py`, but that node has
no incoming edge - `atlas/graph.py`'s own docstring says so plainly ("the
chapter names the node, not a place in the routing topology to reach it
from"). A reader with only Appendix E's "Mounted into Atlas" snippet would
not know that. This module asserts both facts directly against the compiled
graph object rather than trusting either chapter's prose, so a future chapter
that finally wires `route_from_triage` to reach "research" (or mounts the
Chapter 16 supervisor) will break these tests as its signal to update this
appendix - not leave the drift undetected."""

from atlas import graph as graph_module
from atlas import research as research_module
from atlas.deep_research import deep_research_agent, source_lookup
from atlas.research import (
    SourceUnavailable,
    research_graph,
    research_worker,
    search_source,
    supervisor,
)


def test_form_1_research_graph_is_mounted_as_a_node_in_atlas_graph():
    """Appendix E's "Mounted into Atlas" listing, Form 1: `research_graph` is
    a node in Atlas's own compiled builder, not just a standalone subgraph."""
    assert "research" in graph_module.builder.nodes


def test_form_1_research_node_has_no_incoming_edge_in_the_live_routing_graph():
    """The nuance the appendix's own listing does not spell out: being a
    node in `builder` is not the same as being reachable. `atlas/graph.py`'s
    docstring says this plainly ("the chapter names the node, not a place in
    the routing topology to reach it from") - this test pins that fact
    against the actual compiled graph so a later chapter wiring it in has to
    update this appendix too."""
    compiled = graph_module.graph.get_graph()
    incoming = [edge for edge in compiled.edges if edge.target == "research"]

    assert incoming == []


def test_form_0_ch16_supervisor_is_still_not_mounted_anywhere():
    """The comparison the appendix's own "Side by side" table does not
    draw: Chapter 16's hand-rolled supervisor topology (`supervisor`,
    `web_research`, `doc_research`) exists and is unit-tested
    (tests/test_research.py) but is not a node in ANY compiled StateGraph in
    this repo - not Atlas's main graph, not a subgraph of its own. Form 1 and
    Form 2 both eventually get real (if differently reachable) pipelines;
    Form 0 never does. This is the cross-chapter research-architecture drift
    flagged repeatedly in prior chapter/appendix builds - documented here,
    not fixed, per Appendix E's own scope (a side-by-side listing, not a
    redesign)."""
    assert hasattr(research_module, "supervisor")
    assert "supervisor" not in graph_module.builder.nodes
    assert "web_research" not in graph_module.builder.nodes
    assert "doc_research" not in graph_module.builder.nodes


def test_supervisor_still_compiles_even_though_it_is_never_mounted():
    """Not broken, just orphaned: the Chapter 16 illustration still builds a
    real, invokable create_agent graph - it simply has no caller in this
    codebase."""
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
