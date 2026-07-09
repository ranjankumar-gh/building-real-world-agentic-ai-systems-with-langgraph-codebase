"""Chapter 5, "State Design and Reducers" - the same-superstep clobber,
minimized and fixed.

See "Reproducing and fixing the clobber". Two nodes fan out from `START`, so
they run in the same superstep, and both write one channel. `Bad.hits` has no
reducer, so the channel is `LastValue`: one writer per step, enforced by
`InvalidUpdateError` when two nodes race to write it in the same step. `Good`
is the one-line fix - annotate `hits` with a reducer (`operator.add`) and the
two writes concatenate instead of conflicting. Both graphs are kept side by
side, on purpose, so the fix reads as a one-line diff against the failure.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class Bad(TypedDict):
    hits: list  # no reducer -> LastValue -> one writer per step


class Good(TypedDict):
    hits: Annotated[list, operator.add]  # many writers, concatenated


def source_a(state: dict) -> dict:
    return {"hits": ["a1", "a2"]}


def source_b(state: dict) -> dict:
    return {"hits": ["b1"]}


def _build(state_schema: type) -> object:
    builder = StateGraph(state_schema)
    builder.add_node("a", source_a)
    builder.add_node("b", source_b)
    builder.add_edge(START, "a")  # a and b both start from START,
    builder.add_edge(START, "b")  # so they run in the SAME superstep
    builder.add_edge("a", END)
    builder.add_edge("b", END)
    return builder.compile()


bad_graph = _build(Bad)
good_graph = _build(Good)
