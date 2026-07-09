"""Stubs for the node bodies in atlas/graph_sketch.py (Chapter 3) and, from
Chapter 4, atlas/graph.py.

See Chapter 3, "Thinking in Graphs" - the point of that chapter is the SHAPE of
the graph, not these bodies, so classify/search_kb/compose_answer live here as
placeholders that raise NotImplementedError. Chapter 7 ("Tools, Models, and MCP")
replaces classify and search_kb with real implementations against the seeded
knowledge-base backend; compose_answer follows the same path. Do not implement
business logic here before then - a chapter that fills in a stub says so
explicitly.
"""

from typing import Literal


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
