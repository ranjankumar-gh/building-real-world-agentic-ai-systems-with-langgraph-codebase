"""Appendix F, "The Functional API" - atlas/functional_variant.py.

Exercises the appendix's two listings as real, runnable code: `reflect`
(`@entrypoint`/`@task`, with `previous` as the durable-memory primitive) and
`rank_candidates` (`@task` used inside an ordinary `StateGraph` node). Both
use the module's local, deterministic `extractor`/`judge_model` stand-ins,
so nothing here needs a live model call.
"""

from atlas.functional_variant import (
    RankState,
    build_ranking_graph,
    extract_facts,
    rank_candidates,
    reflect,
    score_one_candidate,
)


def test_reflect_extracts_facts_from_each_turn_concurrently():
    config = {"configurable": {"thread_id": "reflect-test-1"}}

    result = reflect.invoke(["Hello there world", "Another message"], config=config)

    facts = {item["fact"] for item in result}
    assert "hello" in facts
    assert "there" in facts
    assert "another" in facts
    assert "message" in facts


def test_reflect_previous_carries_forward_on_the_same_thread():
    config = {"configurable": {"thread_id": "reflect-test-2"}}

    first = reflect.invoke(["hello world"], config=config)
    second = reflect.invoke(["another message"], config=config)

    # `previous` on the second call is the first call's own return value -
    # the durable-memory primitive the appendix describes.
    assert second[: len(first)] == first
    assert len(second) > len(first)


def test_reflect_on_a_fresh_thread_starts_with_no_previous_facts():
    config = {"configurable": {"thread_id": "reflect-test-fresh"}}

    result = reflect.invoke(["a to it"], config=config)

    # None of these words is longer than 4 characters, so no facts
    # extracted - and there is no previous invocation on this thread_id.
    assert result == []


def test_extract_facts_task_returns_a_future_when_called_bare():
    # A bare, uncalled-from-a-runnable-context invocation is not the
    # documented usage - @task only runs inside @entrypoint or a
    # StateGraph node. Calling it bare raises, which is the appendix's own
    # "never bare at module scope" claim.
    import pytest

    with pytest.raises(RuntimeError):
        extract_facts("module scope call")


def test_score_one_candidate_task_also_requires_a_runnable_context():
    import pytest

    with pytest.raises(RuntimeError):
        score_one_candidate({"score": 1})


def test_rank_candidates_orders_high_score_first():
    state: RankState = {
        "candidates": [
            {"name": "a", "score": 1},
            {"name": "b", "score": 5},
            {"name": "c", "score": 3},
        ],
        "ranked": [],
    }

    graph = build_ranking_graph()
    result = graph.invoke(state)

    assert [c["name"] for c in result["ranked"]] == ["b", "c", "a"]


def test_rank_candidates_handles_tied_scores_without_comparing_dicts():
    # Regression: sorted(zip(scores, candidates), reverse=True) with no key
    # falls back to comparing the candidate dicts once scores tie, which
    # raises TypeError. rank_candidates must sort on score alone.
    state: RankState = {
        "candidates": [
            {"name": "x", "score": 5},
            {"name": "y", "score": 5},
        ],
        "ranked": [],
    }

    graph = build_ranking_graph()
    result = graph.invoke(state)

    assert {c["name"] for c in result["ranked"]} == {"x", "y"}


def test_rank_candidates_called_bare_also_requires_a_runnable_context():
    # rank_candidates itself is a plain function - not @task or @entrypoint -
    # but its body calls the @task-wrapped score_one_candidate, so calling
    # rank_candidates bare (outside a StateGraph node's own execution, which
    # is what build_ranking_graph provides) fails for the same reason
    # extract_facts/score_one_candidate do when called bare.
    import pytest

    state: RankState = {
        "candidates": [{"score": 2}, {"score": 9}],
        "ranked": [],
    }

    with pytest.raises(RuntimeError):
        rank_candidates(state)
