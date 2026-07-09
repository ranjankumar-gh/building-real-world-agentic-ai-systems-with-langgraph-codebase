"""Stubs for the node bodies in atlas/graph_sketch.py (Chapter 3) and, from
Chapter 4, atlas/graph.py.

See Chapter 3, "Thinking in Graphs" - the point of that chapter is the SHAPE of
the graph, not these bodies, so classify/search_kb/compose_answer live here as
placeholders that raise NotImplementedError. Chapter 7 ("Tools, Models, and MCP")
replaces classify and search_kb with real implementations against the seeded
knowledge-base backend; compose_answer follows the same path. Do not implement
business logic here before then - a chapter that fills in a stub says so
explicitly.

Chapter 6, "Conditional Edges and Dynamic Control Flow", adds
`KnowledgeBaseUnavailable` - the KB tool's failure type. `atlas/graph.py`'s
`retrieve` node catches it and records the failure in state instead of
crashing the run or pretending the lookup worked. Chapter 7 moves this
exception to live alongside the real `search_kb` implementation in
atlas/tools.py, once that module exists.
"""

from typing import Literal


class KnowledgeBaseUnavailable(Exception):
    """Raised by the knowledge-base tool when it cannot be reached at all -
    distinct from a search that just came back empty. `search_kb` is still a
    stub (see below) and never raises this itself yet; the seeded backend
    that can raise it for real arrives with Chapter 7's atlas/tools.py."""


def classify(messages: list) -> Literal["answer", "retrieve", "escalate"]:
    """Read the conversation, decide the route. Stub - filled in for real in
    Chapter 7."""
    raise NotImplementedError("classify is a stub; implemented for real in Chapter 7")


def search_kb(messages: list) -> list:
    """Search the knowledge base for the active question. Stub - filled in for
    real in Chapter 7."""
    raise NotImplementedError("search_kb is a stub; implemented for real in Chapter 7")


def compose_answer(messages: list, retrieved: list) -> str:
    """Compose a reply from the conversation and whatever was retrieved. Stub -
    filled in for real in Chapter 7."""
    raise NotImplementedError(
        "compose_answer is a stub; implemented for real in Chapter 7"
    )
