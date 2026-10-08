"""Appendix F, "The Functional API" - atlas/functional_variant.py, a
standalone illustration of `@entrypoint`/`@task` as a second way to author a
durable LangGraph workflow, described in plain functions instead of an
explicit graph.

See Appendix F. This module exists only because the appendix is a genuinely
different authoring style worth knowing - nothing in the main chapters uses
it, and Atlas's main workflow (`atlas/graph.py`) stays a `StateGraph`
throughout the book, for the reasons the appendix's "When imperative reads
cleaner" section argues: the moment routing itself needs to be reviewable,
not just the work being done, `StateGraph` wins. (Chapter 14's `reflect`, in
`atlas/memory.py`, is neither: a plain function that `atlas/run.py` runs on
a thread pool after the turn.)

`collect_facts` mirrors the appendix's first listing - an `@entrypoint` that
folds per-turn extraction (`@task`) into a running list of facts. It is an
illustration, not Chapter 14's `reflect`: that makes one extractor call over
the whole numbered transcript and then compacts, so it keeps one current
value per key; this one grows its list on every call. `previous` is the last
invocation's return value on the same `thread_id`, available only with a
checkpointer: the functional API's form of the state a thread carries
between turns (Chapter 9). The resumption boundary is the task: each
finished task's result is saved, so re-invoking a crashed thread with `None`
re-runs only the task that did not finish.

`extract_facts` passes its `RetryPolicy` on the decorator: a bare `@task`
has no retry and no cache. A task's `cache_policy=` also does nothing unless
the entrypoint is given `cache=`, the same trap as a node cache without
`compile(cache=...)` (Chapter 8).

`build_ranking_graph`/`rank_candidates` mirror the second listing - a
`@task` used from inside an ordinary `StateGraph` node for its own internal
retry/cache granularity, the practical interop point between the two
styles.

`extractor`/`judge_model` are local, deterministic stand-ins for model
calls, so this module's tests need no live model access, per the book's
local-only promise. They are not Chapter 14's extractor, which takes a full
message list and returns structured output.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph
from langgraph.types import RetryPolicy


class _LocalExtractor:
    """A stand-in for an LLM-backed fact extractor. Deterministic: pulls
    out the "long" words in a turn as candidate facts, so `collect_facts` below
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


@task(retry_policy=RetryPolicy(max_attempts=3))  # a bare @task never retries
def extract_facts(turn: str) -> list[dict]:
    """One unit of work: its result is saved when it finishes."""
    return extractor.invoke(turn)


@entrypoint(checkpointer=InMemorySaver())
def collect_facts(
    turns: list[str], *, previous: list[dict] | None = None
) -> list[dict]:
    """Per-turn extraction, folded into the thread's running list.
    `previous` is the last invocation's return value on this thread."""
    all_facts = list(previous or [])
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
