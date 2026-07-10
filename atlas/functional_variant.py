"""Appendix F, "The Functional API" - atlas/functional_variant.py, a
standalone illustration of `@entrypoint`/`@task` as a second way to author a
durable LangGraph workflow, described in plain functions instead of an
explicit graph.

See Appendix F. This module exists only because the appendix is a genuinely
different authoring style worth knowing - nothing in the main chapters uses
it, and Atlas's real workflows (`atlas/graph.py`'s resolve flow,
`atlas/memory.py`'s `reflect`) stay `StateGraph`-based throughout the book,
for the reasons the appendix's "When imperative reads cleaner" section
argues: the moment routing itself needs to be reviewable, not just the work
being done, `StateGraph` wins.

`reflect` mirrors the appendix's first listing - an `@entrypoint` that folds
per-turn extraction (`@task`) into a running list of facts, with `previous`
as the durable-memory primitive: the last invocation's return value on the
same `thread_id`, available only with a checkpointer attached (the
functional API's own version of Chapter 9's resumption boundary).
`build_ranking_graph`/`rank_candidates` mirror the second listing - a
`@task` used from inside an ordinary `StateGraph` node for its own internal
retry/cache granularity, the practical interop point between the two
styles.

`extractor`/`judge_model` stand in for the LLM-backed calls the appendix's
prose implies (`extractor.invoke(turn)`, `judge_model.invoke(candidate)`) -
local and deterministic, so this module's tests need no live model access,
per the book's local-only promise. They are not Chapter 14's real
`extractor` (that one takes a full message list and returns structured
output; this appendix's version is illustrative pseudocode, not literally
Chapter 14's function).
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph


class _LocalExtractor:
    """A stand-in for an LLM-backed fact extractor. Deterministic: pulls
    out the "long" words in a turn as candidate facts, so `reflect` below
    is fully runnable and testable with no external model call."""

    def invoke(self, turn: str) -> list[dict]:
        words = [w.strip(".,!?").lower() for w in turn.split()]
        return [{"fact": w} for w in words if len(w) > 4]


class _LocalJudge:
    """A stand-in for an LLM-backed candidate scorer. Deterministic: reads
    a `score` field straight off the candidate."""

    def invoke(self, candidate: dict) -> float:
        return float(candidate.get("score", 0.0))


extractor = _LocalExtractor()
judge_model = _LocalJudge()


@task
def extract_facts(turn: str) -> list[dict]:
    """A retryable, cacheable unit of work - Ch10's RetryPolicy applies
    here too."""
    return extractor.invoke(turn)


@entrypoint(checkpointer=InMemorySaver())
def reflect(turns: list[str], *, previous: list[dict] | None = None) -> list[dict]:
    """The durable workflow itself. `previous` is the last invocation's
    return value on this thread - available only with a checkpointer."""
    all_facts = previous or []
    futures = [extract_facts(turn) for turn in turns]  # tasks run concurrently
    for future in futures:
        all_facts.extend(future.result())
    return all_facts


@task
def score_one_candidate(candidate: dict) -> float:
    return judge_model.invoke(candidate)


class RankState(TypedDict):
    """Illustrative only - 'candidates'/'ranked' are not real Atlas state
    channels; see the appendix's own caveat on `rank_candidates`."""

    candidates: list[dict]
    ranked: list[dict]


def rank_candidates(state: RankState) -> dict:
    """A StateGraph node using @task for its own internal retry/cache
    granularity - Ch17's Send would be the alternative if this needed the
    runtime's own parallel-fan-out bookkeeping instead."""
    candidates = state["candidates"]
    futures = [score_one_candidate(c) for c in candidates]
    scores = [f.result() for f in futures]
    ranked = [
        c
        for _, c in sorted(
            zip(scores, candidates), key=lambda pair: pair[0], reverse=True
        )
    ]
    return {"ranked": ranked}


def build_ranking_graph():
    """Compiles `rank_candidates` into the smallest possible StateGraph, so
    the appendix's "@task inside a StateGraph node" claim is backed by a
    real, invokable graph rather than an isolated function."""
    graph = StateGraph(RankState)
    graph.add_node("rank", rank_candidates)
    graph.set_entry_point("rank")
    graph.set_finish_point("rank")
    return graph.compile()
