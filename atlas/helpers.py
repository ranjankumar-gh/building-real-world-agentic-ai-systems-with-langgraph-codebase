"""Stubs for the node bodies in atlas/graph_sketch.py (Chapter 3) and, from
Chapter 4, atlas/graph.py.

See Chapter 3, "Thinking in Graphs" - the point of that chapter is the SHAPE of
the graph, not these bodies, so classify/search_kb/compose_answer originally
lived here as placeholders that raise NotImplementedError.

Chapter 6, "Conditional Edges and Dynamic Control Flow", added
`KnowledgeBaseUnavailable` - the KB tool's failure type - and `atlas/graph.py`'s
`retrieve` node has caught it since.

Chapter 7, "Tools, Models, MCP, and create_agent", replaces `classify` for
real: the validated version now lives in `atlas/triage.py`, and
`atlas/graph.py`'s `triage` node imports it from there instead of here.
`KnowledgeBaseUnavailable` moves too, into `atlas/tools.py`, alongside the
real `search_kb` tool that chapter builds - a standalone, tool-calling
`create_agent` (`atlas/agent.py`) can now actually search the knowledge base
and act on tickets.

`search_kb` and `compose_answer` stay here, still stubs, because
`atlas/graph.py`'s `retrieve` and `answer` node bodies are not yet rewired to
call the new tools - that integration (folding a tool-calling agent into
this graph as a node/sub-agent) is a later chapter's job. Do not implement
business logic here before then - a chapter that fills in a stub says so
explicitly, and none has yet for these two.
"""

def search_kb(messages: list) -> list:
    """Search the knowledge base for the active question. Stub - the real
    knowledge-base tool exists in atlas/tools.py as of Chapter 7, but this
    graph node's calling convention (list of messages in, list of hits out)
    hasn't been rewired to it yet."""
    raise NotImplementedError("search_kb is a stub; not yet wired to atlas.tools")


def compose_answer(messages: list, retrieved: list) -> str:
    """Compose a reply from the conversation and whatever was retrieved. Stub -
    still unimplemented; see the module docstring."""
    raise NotImplementedError(
        "compose_answer is a stub; not yet filled in by any chapter"
    )
