"""Chapter 3: atlas/graph_sketch.py - node SHAPES obey the state-transition
discipline (read state, return a delta, decide nothing about what runs next).
No StateGraph exists yet; that is Chapter 4."""

import pytest

from atlas import graph_sketch
from atlas.graph_sketch import (
    AtlasState,
    answer,
    escalate,
    retrieve,
    route_after_triage,
    triage,
)


def _state(**overrides) -> AtlasState:
    base: AtlasState = {"messages": [], "ticket": None, "retrieved": [], "route": ""}
    base.update(overrides)
    return base


def test_escalate_returns_a_delta_only_and_does_not_touch_the_input_state():
    state = _state(ticket=None)

    delta = escalate(state)

    assert delta == {"ticket": {"status": "escalated"}}
    assert state["ticket"] is None  # escalate did not mutate what it read


def test_route_after_triage_reads_the_route_key_and_decides_nothing_else():
    for route in ("answer", "retrieve", "escalate"):
        assert route_after_triage(_state(route=route)) == route


def test_triage_calls_classify_and_wraps_its_result_in_a_delta(monkeypatch):
    monkeypatch.setattr(graph_sketch, "classify", lambda messages: "retrieve")

    delta = triage(_state(messages=["hi"]))

    assert delta == {"route": "retrieve"}


def test_retrieve_calls_search_kb_and_wraps_its_result_in_a_delta(monkeypatch):
    monkeypatch.setattr(graph_sketch, "search_kb", lambda messages: ["hit-1"])

    delta = retrieve(_state(messages=["hi"]))

    assert delta == {"retrieved": ["hit-1"]}


def test_answer_calls_compose_answer_and_wraps_its_result_in_a_delta(monkeypatch):
    monkeypatch.setattr(
        graph_sketch, "compose_answer", lambda messages, retrieved: "reply"
    )

    delta = answer(_state(messages=["hi"], retrieved=["hit-1"]))

    assert delta == {"messages": ["reply"]}


def test_triage_propagates_the_classify_stub_until_chapter_7_fills_it_in():
    with pytest.raises(NotImplementedError):
        triage(_state(messages=["hi"]))
