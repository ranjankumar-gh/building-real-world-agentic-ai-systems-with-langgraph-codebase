"""Chapter 7, "Tools, Models, MCP, and create_agent" - Atlas's real tools.

See "Building Atlas's real tools". This module replaces two of the three
`atlas.helpers` stubs with real implementations against seeded, in-repo
backends (Appendix G): `search_kb` (the knowledge-base lookup) and the
ticket operations `lookup_ticket` / `set_ticket_status` (new, narrow tools -
`atlas.helpers` never had a ticket stub to replace). `KnowledgeBaseUnavailable`
also moves here, next to the tool that can actually raise it.

Every tool below is deliberately the *narrowest* authority that does the
job - see "Tools as authority, not function calls". `set_ticket_status`
accepts a `Literal` of four values, not an arbitrary `fields: dict`, so the
model that invents a status like "clozed" cannot call it at all: the value
is not in the schema.

`text_of` also lives here, rewritten against the standardized
`message.content_blocks` - see "Reading the result: content blocks, not
guesswork". It replaces the `content`-guessing version in `atlas/naive.py`,
which stays as-is: that module is Chapter 1's frozen, deliberately fragile
artifact, not something later chapters update in place.

Chapter 16, "The Supervisor Pattern (and Swarm as Contrast)", adds
`web_search_tool` - a second seeded, mockable lookup alongside `search_kb`,
for the `web_research` specialist in `atlas/research.py`. Same shape as
`search_kb`: a small in-repo dict standing in for a live web-search API, so
the specialist's tool call runs fully offline in tests."""

from typing import Literal

from langchain_core.tools import tool

# Seeded, mockable backend - ships in the companion repo (Appendix G).
_KB: dict[str, str] = {
    "refund window": "Refunds are available within 30 days of purchase.",
    "reset password": "Use the 'Forgot password' link on the sign-in page.",
}


class KnowledgeBaseUnavailable(RuntimeError):
    """Raised when the KB backend cannot be reached."""


@tool
def search_kb(query: str) -> str:
    """Search the support knowledge base for an article answering the query.

    Use this for any factual product question before answering the user.
    Returns the article text, or a clear 'no match' message."""
    for key, article in _KB.items():
        if key in query.lower():
            return article
    return "No knowledge-base article matched."


TicketStatus = Literal["open", "pending", "resolved", "escalated"]

# Seeded backend.
_TICKETS: dict[str, dict] = {
    "T-1001": {"status": "open", "priority": "normal", "notes": []},
}


@tool
def lookup_ticket(ticket_id: str) -> dict:
    """Fetch the current state of a support ticket by its id."""
    ticket = _TICKETS.get(ticket_id)
    if ticket is None:
        return {"error": f"No ticket {ticket_id}."}  # typed miss, not an exception
    return dict(ticket)


@tool
def set_ticket_status(ticket_id: str, status: TicketStatus) -> str:
    """Set a ticket's status. STATUS must be one of the allowed values."""
    if ticket_id not in _TICKETS:
        return f"No ticket {ticket_id}."
    _TICKETS[ticket_id]["status"] = status
    return f"Ticket {ticket_id} set to {status}."


# Seeded, mockable backend for the Chapter 16 web-research specialist -
# stands in for a live web-search API so tests run fully offline.
_WEB: dict[str, str] = {
    "refund policy": "Most vendors offer refunds within 30 days; check the merchant's terms.",
    "langgraph": "LangGraph is a low-level orchestration framework for building stateful agents.",
}


@tool
def web_search_tool(query: str) -> str:
    """Search the public web for the query.

    Use this for open-ended research questions outside the internal
    knowledge base. Returns a snippet, or a clear 'no match' message."""
    for key, snippet in _WEB.items():
        if key in query.lower():
            return snippet
    return "No web result matched."


def text_of(message) -> str:
    """Plain text from a message's standardized content blocks."""
    return "".join(
        block["text"]
        for block in message.content_blocks
        if block["type"] == "text"
    )
