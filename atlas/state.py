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
`state.get("error")` since it is not written on every path). `triage`
resets both (`retrieve_attempts=0`, `error=None`) at the start of every
question, and empties `retrieved` with `Overwrite([])` - the reducer would
otherwise merge an empty list into the last question's documents - so the
cap and the documents stay per question once a checkpointer carries state
from one turn to the next.

Chapter 10, "Durable Execution, Long-Running Workflows, and State
Migration", adds `refund_done` - ADDITIVE, on purpose: a checkpoint written
before this chapter has no such key, and subscript access on it would raise
when that old checkpoint resumes. Read it with `state.get("refund_done",
False)`, never `state["refund_done"]`, so pre-Chapter-10 checkpoints keep
resuming safely (see "State migration without downtime").

Chapter 11, "Human-in-the-Loop", adds `approval` - the audit record the
approval gate writes on every decision it routes: the decision type, who
made it (`by`, carried in the resume value), and a UTC timestamp. It is a
channel of its own, so `escalate` overwriting `ticket` does not erase it, and
the checkpoint after the gate - the one the refund resumes from - carries it.
Additive like `refund_done`: read it with `state.get("approval")`.

Chapter 12, "Context Engineering", is the first reader of `Doc.score`
(declared in Chapter 5, set by the retriever): `atlas/context.py`'s
`select_docs` ranks retrieved documents on it and caps them to the retrieved
slice of the context budget.

Chapter 13, "Short-Term vs Long-Term Memory", adds `customer_profile`, the
channel `atlas/graph.py`'s `recall` writes before `triage`: the profile
entries the store holds for the ticket's customer, key to value. The
declaration is required - LangGraph drops a write to an undeclared key
without an error (Chapter 6). Additive like `refund_done`: a run with no
customer on its ticket never writes it, so read it with
`state.get("customer_profile", {})`.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Doc(TypedDict):
    id: str
    text: str
    score: float  # relevance, set by the retriever; Chapter 12 ranks on it


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
    retrieve_attempts: int  # the loop guard; triage resets it, retrieve counts
    error: str | None  # why the run escalated; triage clears it
    refund_done: bool  # NEW this chapter - additive, read with a default
    approval: dict | None  # Chapter 11: who decided the refund, and when
    customer_profile: dict[str, str]  # Chapter 13: recall loads it before triage
