"""Appendix F, "The Functional API" - atlas/functional_variant.py.

Exercises the appendix's two listings as real, runnable code: `collect_facts`
(`@entrypoint`/`@task`, with `previous` carrying the thread's last result) and
`rank_candidates` (`@task` used inside an ordinary `StateGraph` node). Both
use the module's local, deterministic `extractor`/`judge_model` stand-ins,
so nothing here needs a live model call.
"""

from atlas.functional_variant import (
    RankState,
    build_ranking_graph,
    extract_facts,
    rank_candidates,
    collect_facts,
    score_one_candidate,
)


def test_collect_facts_extracts_facts_from_each_turn_concurrently():
    config = {"configurable": {"thread_id": "collect_facts-test-1"}}

    result = collect_facts.invoke(
        ["Hello there world", "Another message"], config=config
    )

    facts = {item["fact"] for item in result}
    assert "hello" in facts
    assert "there" in facts
    assert "another" in facts
    assert "message" in facts


def test_collect_facts_previous_carries_forward_on_the_same_thread():
    config = {"configurable": {"thread_id": "collect_facts-test-2"}}

    first = collect_facts.invoke(["hello world"], config=config)
    second = collect_facts.invoke(["another message"], config=config)

    # `previous` on the second call is the first call's own return value -
    # the durable-memory primitive the appendix describes.
    assert second[: len(first)] == first
    assert len(second) > len(first)


def test_collect_facts_on_a_fresh_thread_starts_with_no_previous_facts():
    config = {"configurable": {"thread_id": "collect_facts-test-fresh"}}

    result = collect_facts.invoke(["a to it"], config=config)

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


# --- the runtime facts the appendix states -------------------------------


def test_extract_facts_carries_its_retry_policy_on_the_decorator(monkeypatch):
    # a bare @task has no retry; this one retries a transient failure
    from atlas import functional_variant as fv

    calls = {"n": 0}

    class _FlakyExtractor:
        def invoke(self, turn: str) -> list[dict]:
            calls["n"] += 1
            if calls["n"] == 1:
                raise ConnectionError("transient")
            return [{"fact": turn}]

    monkeypatch.setattr(fv, "extractor", _FlakyExtractor())
    config = {"configurable": {"thread_id": "collect-facts-retry"}}

    assert collect_facts.invoke(["refund"], config=config) == [{"fact": "refund"}]
    assert calls["n"] == 2


def test_a_bare_task_does_not_retry():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.func import entrypoint, task

    calls = {"n": 0}

    @task
    def flaky() -> str:
        calls["n"] += 1
        raise ConnectionError("transient")

    @entrypoint(checkpointer=InMemorySaver())
    def run(_: str) -> str:
        return flaky().result()

    import pytest

    with pytest.raises(ConnectionError):
        run.invoke("go", {"configurable": {"thread_id": "bare-task"}})
    assert calls["n"] == 1


def test_previous_without_a_checkpointer_is_always_none():
    from langgraph.func import entrypoint

    @entrypoint()
    def count(_: str, *, previous: int | None = None) -> int:
        return (previous or 0) + 1

    config = {"configurable": {"thread_id": "no-saver"}}
    assert [count.invoke("go", config), count.invoke("go", config)] == [1, 1]


def test_a_resumed_thread_reruns_only_the_unfinished_task():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.func import entrypoint, task

    calls = {"a": 0, "b": 0}
    crash = {"b": True}

    @task
    def step_a() -> str:
        calls["a"] += 1
        return "a"

    @task
    def step_b(prev: str) -> str:
        calls["b"] += 1
        if crash["b"]:
            crash["b"] = False
            raise RuntimeError("crash")
        return prev + "b"

    @entrypoint(checkpointer=InMemorySaver())
    def pipeline(_: str) -> str:
        return step_b(step_a().result()).result()

    import pytest

    config = {"configurable": {"thread_id": "resume-unit"}}
    with pytest.raises(RuntimeError):
        pipeline.invoke("go", config)

    assert pipeline.invoke(None, config) == "ab"  # None: resume this thread
    assert calls == {"a": 1, "b": 2}


def test_an_entrypoint_mounts_as_a_node_in_a_state_graph():
    from typing import TypedDict

    from langgraph.func import entrypoint
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict):
        out: list[int]

    @entrypoint()
    def double(state: S) -> S:
        return {"out": [x * 2 for x in state["out"]]}

    builder = StateGraph(S)
    builder.add_node("double", double)
    builder.add_edge(START, "double")
    builder.add_edge("double", END)

    assert builder.compile().invoke({"out": [1, 2]}) == {"out": [2, 4]}
