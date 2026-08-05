"""Chapter 3: atlas/helpers.py - classify/search_kb/compose_answer began as
stubs. Chapter 7 moved classify to atlas.triage and KnowledgeBaseUnavailable
to atlas.tools, alongside the real search_kb tool. search_kb and
compose_answer are now the adapters between the graph's node calling
convention and that tool; see atlas/helpers.py's module docstring."""

from langchain_core.messages import AIMessage, HumanMessage

from atlas import helpers


def test_search_kb_adapts_the_tool_and_returns_docs_that_merge_by_id():
    """The node convention is messages in, list[Doc] out - the Doc shape is
    what lets `retrieved` merge through dedup_by_id on the state contract."""
    hits = helpers.search_kb([HumanMessage("what is the refund window?")])

    assert len(hits) == 1
    assert hits[0]["text"].startswith("Refunds are available")
    assert set(hits[0]) == {"id", "text", "score"}


def test_search_kb_returns_empty_on_a_miss_so_the_retry_loop_can_read_it():
    """A miss must be an empty list, not a 'nothing found' document:
    route_after_retrieve reads emptiness to drive the bounded retry and the
    graceful escalation."""
    assert helpers.search_kb([HumanMessage("do you sell submarines?")]) == []


def test_search_kb_reads_the_most_recent_human_turn():
    """Not the first turn, and not an assistant turn - the active question."""
    hits = helpers.search_kb(
        [
            HumanMessage("hello"),
            AIMessage("Hi, how can I help?"),
            HumanMessage("how do I reset password?"),
        ]
    )

    assert hits and "Forgot password" in hits[0]["text"]


def test_compose_answer_is_grounded_strictly_in_what_was_retrieved():
    docs = [{"id": "kb:x", "text": "Refunds are available within 30 days.", "score": 1.0}]

    assert "30 days" in helpers.compose_answer([], docs)


def test_compose_answer_declines_rather_than_inventing_when_nothing_retrieved():
    """The behaviour the escalation path depends on: no documents means say
    so, never fabricate an answer."""
    reply = helpers.compose_answer([HumanMessage("anything?")], [])

    assert "could not find" in reply
    assert "specialist" in reply


def test_classify_no_longer_lives_in_helpers():
    """Chapter 7 moves classify to atlas.triage - confirm it is gone from
    here rather than silently drifting out of sync with the real one."""
    assert not hasattr(helpers, "classify")


def test_knowledge_base_unavailable_no_longer_lives_in_helpers():
    """Chapter 7 moves this exception to atlas.tools, next to the real
    search_kb tool that can raise it."""
    assert not hasattr(helpers, "KnowledgeBaseUnavailable")
