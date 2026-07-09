"""Chapter 5, "State Design and Reducers" - AtlasState with its full,
reducer-annotated contract.

See "Designing AtlasState". Every channel here states its own merge rule:
`messages` and `retrieved` accept multiple writers per superstep and merge;
`ticket` and `route` are single-writer, guarded by the runtime's default
`LastValue` channel - only one update per superstep, enforced by
`InvalidUpdateError` (see atlas/clobber_demo.py) if that contract is ever
violated. atlas/graph.py now imports `AtlasState` from here; Chapter 4's
inline, partially-annotated TypedDict is retired.
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
