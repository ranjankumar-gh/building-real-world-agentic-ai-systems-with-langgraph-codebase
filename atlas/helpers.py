"""Helpers behind the node bodies in atlas/graph_sketch.py (Chapter 3) and, from
Chapter 4, atlas/graph.py.

See Chapter 3, "Thinking in Graphs" - the point of that chapter is the SHAPE of
the graph, not these bodies, so classify/search_kb/compose_answer originally
lived here as placeholders that raise NotImplementedError. Each stub raised
until the chapter that implemented it.

Chapter 6, "Conditional Edges and Dynamic Control Flow", added
`KnowledgeBaseUnavailable` - the KB tool's failure type - and `atlas/graph.py`'s
`retrieve` node has caught it since.

Chapter 7, "Tools, Models, MCP, and create_agent", replaces `classify` for
real: the validated version now lives in `atlas/triage.py`, and
`atlas/graph.py`'s `triage` node imports it from there instead of here.
`KnowledgeBaseUnavailable` moves too, into `atlas/tools.py`, alongside the
real `search_kb` tool that chapter builds - a standalone, tool-calling
`create_agent` (`atlas/agent.py`) can now actually search the knowledge base
and act on tickets. At the Chapter 7 tag `search_kb` and `compose_answer`
here are still stubs; the finished repo replaces both with the adapters below.

In the finished repo, `search_kb` and `compose_answer` are the adapters between
the graph's node calling convention (list of messages in, list of `Doc` out) and
the real, narrow tool Chapter 7 built. The tool takes a query string and returns
article text; a node takes state and returns a state delta. That mismatch is the
whole job of this module - it is a seam, not business logic.

Both are deliberately model-free. `compose_answer` templates the retrieved
article rather than calling a model, for the same reason
`atlas/sla_watch.py`'s `compose_checkin` does: the retrieve/answer path then
runs with no API key and no network, which is what makes the graph's routing,
retry, and escalation behavior testable offline.
"""

from atlas.state import Doc
from atlas.tools import search_kb as search_kb_tool
from atlas.tools import text_of

# The KB returns this exact sentinel when nothing matches - see atlas/tools.py.
_NO_MATCH = "No knowledge-base article matched."


def _last_user_text(messages: list) -> str:
    """The active question: the most recent human turn, as plain text."""
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return text_of(message)
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def search_kb(messages: list) -> list[Doc]:
    """Search the knowledge base for the active question.

    Adapts Chapter 7's `search_kb` tool to the node convention: pull the live
    question out of the conversation, ask the tool, and wrap the answer as a
    `Doc` so it merges through `dedup_by_id` on the `retrieved` channel.

    Returns `[]` on a miss rather than a "nothing found" document, because an
    empty list is what `route_after_retrieve` reads to drive the bounded retry
    and, eventually, the graceful escalation."""
    query = _last_user_text(messages)
    if not query:
        return []
    article = search_kb_tool.invoke({"query": query})
    if article == _NO_MATCH:
        return []
    return [Doc(id=f"kb:{query.strip().lower()[:64]}", text=article, score=1.0)]


def compose_answer(messages: list, retrieved: list[Doc]) -> str:
    """Compose a reply from the conversation and whatever was retrieved.

    Deterministic on purpose (see the module docstring). Grounded strictly in
    `retrieved`: with nothing retrieved this says so rather than inventing an
    answer, which is the behaviour the escalation path depends on."""
    if not retrieved:
        return (
            "I could not find an article covering that. Passing this to a "
            "support specialist who can help."
        )
    body = " ".join(doc["text"] for doc in retrieved)
    return f"{body} Let me know if that does not answer your question."
