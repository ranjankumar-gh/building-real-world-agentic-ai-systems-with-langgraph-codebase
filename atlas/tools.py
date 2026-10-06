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
the specialist's tool call runs fully offline in tests.

Chapter 27, "Capstone", adds `list_at_risk_tickets`/`send_checkin` for the
SLA Watch vertical (`atlas/sla_watch.py`) - narrow by the same Chapter 7
discipline: one reads, one writes exactly one thing. Their seeded backend,
`_SLA_TICKETS`, is deliberately a separate object from `_TICKETS` above -
it tracks ticket age, not status."""

from typing import Literal

from langchain_core.messages import BaseMessage
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


# Chapter 27, "Capstone" - the seeded backend for SLA Watch's two new
# tools below. Deliberately a DIFFERENT object from `_TICKETS` above
# (Chapter 7): that dict tracks status/priority/notes, keyed for
# lookup_ticket/set_ticket_status; this one tracks ticket AGE, the fact
# SLA Watch actually needs. Giving both the same name would have this
# definition silently clobber Chapter 7's `_TICKETS` on import.
class _SLATicketBackend:
    """Seeded, mockable backend standing in for a live ticket-age query and
    send API (Appendix A)."""

    def __init__(self, tickets: list[dict]) -> None:
        self._tickets = tickets
        self.sent: list[dict] = []  # test-visible record of what actually sent
        self._receipts: dict[str, str] = {}  # idempotency ledger, keyed

    def reset(self) -> None:
        """Clear the send record and the idempotency ledger. For tests only:
        the ledger is deliberately process-global (that is what makes it work
        across a replay), so without this one test's send suppresses the
        next test's identical key."""
        self.sent.clear()
        self._receipts.clear()

    def at_risk(self, threshold_hours: int) -> list[dict]:
        return [t for t in self._tickets if t["hours_open"] >= threshold_hours]

    def send_message(self, key: str, ticket_id: str, message: str) -> str:
        """Idempotent by key - the same contract `charge_refund` uses across
        the checkpoint membrane (Chapter 10). A check-in is an irreversible
        customer-facing effect, so a replayed node must not re-send it."""
        if key in self._receipts:
            return self._receipts[key]
        self.sent.append({"ticket_id": ticket_id, "message": message})
        receipt = f"check-in sent for {ticket_id}"
        self._receipts[key] = receipt
        return receipt


_SLA_TICKETS = _SLATicketBackend(
    [
        {"ticket_id": "T-2001", "hours_open": 30},  # past the 24h threshold
        {"ticket_id": "T-2002", "hours_open": 10},  # not yet at risk
    ]
)


@tool
def list_at_risk_tickets(threshold_hours: int) -> list[dict]:
    """List open tickets that have been unresolved longer than threshold_hours."""
    return _SLA_TICKETS.at_risk(threshold_hours)   # seeded backend, Appendix A


@tool
def send_checkin(key: str, ticket_id: str, message: str) -> str:
    """Send a check-in message to the customer on one ticket. Narrow by
    design: this tool can send exactly one thing, to one ticket, nothing
    else - the same authority-surface discipline as set_ticket_status.

    `key` is a stable idempotency key (Chapter 10): a check-in is an
    irreversible effect past the membrane, so a resumed or replayed node
    must collapse onto the same key rather than messaging a customer twice."""
    return _SLA_TICKETS.send_message(key, ticket_id, message)


def text_of(message: BaseMessage) -> str:
    """Plain text from a message's standardized content blocks."""
    return "".join(
        block["text"]
        for block in message.content_blocks
        if block["type"] == "text"
    )
