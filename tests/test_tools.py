"""Chapter 7, "Tools, Models, MCP, and create_agent" - atlas/tools.py.
Chapter 16, "The Supervisor Pattern (and Swarm as Contrast)", adds coverage
for `web_search_tool`.

See "Building Atlas's real tools" and "Reading the result: content blocks,
not guesswork". These are the tools a tool-calling model actually calls -
tests exercise them the way the model would, via `.invoke(args_dict)`, not
by calling the underlying Python function directly."""

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from atlas.tools import (
    KnowledgeBaseUnavailable,
    lookup_ticket,
    search_kb,
    set_ticket_status,
    text_of,
    web_search_tool,
)


def test_search_kb_answers_a_matching_query():
    result = search_kb.invoke({"query": "What's the refund window?"})

    assert result == "Refunds are available within 30 days of purchase."


def test_search_kb_reports_no_match_for_an_unrelated_query():
    result = search_kb.invoke({"query": "Where is my order?"})

    assert result == "No knowledge-base article matched."


def test_knowledge_base_unavailable_is_a_distinct_exception_type():
    assert issubclass(KnowledgeBaseUnavailable, Exception)

    with pytest.raises(KnowledgeBaseUnavailable):
        raise KnowledgeBaseUnavailable("knowledge base is down")


def test_lookup_ticket_returns_the_seeded_ticket_state():
    result = lookup_ticket.invoke({"ticket_id": "T-1001"})

    assert result == {"status": "open", "priority": "normal", "notes": []}


def test_lookup_ticket_returns_a_typed_miss_instead_of_raising():
    """The Chapter 1 anti-pattern the callout warns against is a tool that
    turns a real failure into a fake success - a miss is the opposite case,
    and it must be visible to the model as data, not swallowed."""
    result = lookup_ticket.invoke({"ticket_id": "T-9999"})

    assert result == {"error": "No ticket T-9999."}


def test_set_ticket_status_writes_an_allowed_status():
    result = set_ticket_status.invoke({"ticket_id": "T-1001", "status": "resolved"})

    assert result == "Ticket T-1001 set to resolved."
    assert lookup_ticket.invoke({"ticket_id": "T-1001"})["status"] == "resolved"


def test_set_ticket_status_reports_an_unknown_ticket_without_writing():
    result = set_ticket_status.invoke({"ticket_id": "T-9999", "status": "resolved"})

    assert result == "No ticket T-9999."


def test_set_ticket_status_rejects_a_value_outside_the_literal_schema():
    """The fix for the chapter's hook: 'clozed' is not in the schema, so the
    call is rejected before it ever reaches the backend - a guardrail, not
    a hint."""
    with pytest.raises(ValidationError):
        set_ticket_status.invoke({"ticket_id": "T-1001", "status": "clozed"})


def test_set_ticket_status_tool_schema_only_admits_the_four_statuses():
    schema = set_ticket_status.args_schema.model_json_schema()

    assert set(schema["properties"]["status"]["enum"]) == {
        "open",
        "pending",
        "resolved",
        "escalated",
    }


def test_text_of_reads_a_plain_text_content_block():
    message = AIMessage(content="Refunds are available within 30 days.")

    assert text_of(message) == "Refunds are available within 30 days."


def test_text_of_joins_multiple_text_blocks_and_skips_non_text_blocks():
    message = AIMessage(
        content=[
            {"type": "text", "text": "Part one. "},
            {"type": "text", "text": "Part two."},
        ]
    )

    assert text_of(message) == "Part one. Part two."


def test_web_search_tool_answers_a_matching_query():
    result = web_search_tool.invoke({"query": "What is LangGraph?"})

    assert "orchestration" in result.lower()


def test_web_search_tool_reports_no_match_for_an_unrelated_query():
    result = web_search_tool.invoke({"query": "best pizza in town"})

    assert result == "No web result matched."
