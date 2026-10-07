"""Chapter 19, "Streaming" - atlas/stream.py.

See "Building the multiplexed stream". `stream_atlas` wraps `atlas/graph.py`'s
compiled `graph` with `stream_mode=["updates", "messages", "custom"]`,
`subgraphs=True`, `version="v2"`, and normalizes every chunk into
`{"kind", "source", "data"}`. These tests monkeypatch the same seams
`tests/test_graph.py` already does (`classify`/`search_kb`/`compose_answer`)
to avoid a live model call - the point under test is the multiplexing and
namespace-normalization this chapter adds, not the model's own token
streaming.
"""

from types import SimpleNamespace

from atlas import graph as graph_module
from atlas.stream import stream_atlas


def _decision(route: str):
    return SimpleNamespace(route=route)


def _stub_the_model_seams(monkeypatch, route: str = "answer") -> None:
    monkeypatch.setattr(graph_module, "classify", lambda messages: _decision(route))
    monkeypatch.setattr(graph_module, "search_kb", lambda messages: [])
    monkeypatch.setattr(
        graph_module, "compose_answer", lambda messages, retrieved, **_: "final answer"
    )


def test_stream_atlas_yields_events_shaped_as_kind_source_data(monkeypatch):
    _stub_the_model_seams(monkeypatch)
    config = {"configurable": {"thread_id": "test-thread-stream-atlas-shape"}}

    events = list(
        stream_atlas({"messages": [{"role": "user", "content": "hi"}]}, config)
    )

    assert events  # triage and answer both land as updates
    for event in events:
        assert set(event) == {"kind", "source", "data"}
        assert event["kind"] in {"updates", "messages", "custom"}


def test_stream_atlas_normalizes_the_root_namespace_to_main(monkeypatch):
    """chunk["ns"] is () for the root graph - stream_atlas maps that to
    ("main",) so a consumer never has to special-case the root graph."""
    _stub_the_model_seams(monkeypatch)
    config = {"configurable": {"thread_id": "test-thread-stream-atlas-ns"}}

    events = list(
        stream_atlas({"messages": [{"role": "user", "content": "hi"}]}, config)
    )

    assert all(event["source"] == ("main",) for event in events)


def test_stream_atlas_reports_an_updates_event_for_every_node_that_ran(monkeypatch):
    _stub_the_model_seams(monkeypatch)
    config = {"configurable": {"thread_id": "test-thread-stream-atlas-updates"}}

    events = list(
        stream_atlas({"messages": [{"role": "user", "content": "hi"}]}, config)
    )

    updated_nodes = {
        node
        for event in events
        if event["kind"] == "updates"
        for node in event["data"]
    }
    assert {"triage", "answer"} <= updated_nodes


def test_stream_atlas_carries_the_route_triage_decided_in_its_updates_payload(
    monkeypatch,
):
    _stub_the_model_seams(monkeypatch, route="escalate")
    config = {"configurable": {"thread_id": "test-thread-stream-atlas-escalate"}}

    events = list(
        stream_atlas({"messages": [{"role": "user", "content": "hi"}]}, config)
    )

    triage_updates = [
        event["data"]["triage"]
        for event in events
        if event["kind"] == "updates" and "triage" in event["data"]
    ]
    # triage's update also carries the Chapter 6 per-question guard reset
    assert [update["route"] for update in triage_updates] == ["escalate"]
