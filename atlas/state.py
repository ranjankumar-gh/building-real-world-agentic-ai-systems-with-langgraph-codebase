"""Chapter 5, "State Design and Reducers" - AtlasState with its full,
reducer-annotated contract.

See "Designing AtlasState". Every channel here states its own merge rule:
`messages` and `retrieved` accept multiple writers per superstep and merge;
`ticket` and `route` are single-writer, guarded by the runtime's default
`LastValue` channel - only one update per superstep, enforced by
`InvalidUpdateError` (see atlas/clobber_demo.py) if that contract is ever
violated. atlas/graph.py now imports `AtlasState` from here; Chapter 4's
inline, partially-annotated TypedDict is retired.

Chapter 6, "Conditional Edges and Dynamic Control Flow", adds two more
LastValue channels for the bounded retrieval retry and its recovery path:
`retrieve_attempts` (the explicit loop guard `route_after_retrieve` checks
instead of leaning on the recursion limit) and `error` (set only when
`retrieve` catches a `KnowledgeBaseUnavailable` failure, read with
`state.get("error")` since it is not written on every path).

Chapter 10, "Durable Execution, Long-Running Workflows, and State
Migration", adds `refund_done` - ADDITIVE, on purpose: a checkpoint written
before this chapter has no such key, and subscript access on it would raise
when that old checkpoint resumes. Read it with `state.get("refund_done",
False)`, never `state["refund_done"]`, so pre-Chapter-10 checkpoints keep
resuming safely (see "State migration without downtime").
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Doc(TypedDict):
    id: str
    text: str


def dedup_by_id(current: list[Doc], update: list[Doc]) -> list[Doc]:
    """Merge retrieved documents, dropping duplicates by id, preserving order.

    Associative enough for fan-in: whether several sources land in the same
    superstep or across retry passes, the result is the same set of unique
    documents, in first-seen order.
    """
    seen = {doc["id"] for doc in current}
    merged = list(current)
    for doc in update:
        if doc["id"] not in seen:
            merged.append(doc)
            seen.add(doc["id"])
    return merged


class AtlasState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # many writers, merged
    retrieved: Annotated[list[Doc], dedup_by_id]  # many writers, merged
    ticket: dict | None  # one writer per step, guarded (LastValue)
    route: str  # one writer per step, guarded (LastValue)
    retrieve_attempts: int  # the explicit loop guard; LastValue (only retrieve writes it)
    error: str | None  # recorded tool failure; LastValue (only retrieve writes it)
    refund_done: bool  # NEW this chapter - additive, read with a default
