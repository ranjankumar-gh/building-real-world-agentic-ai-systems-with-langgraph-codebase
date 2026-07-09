"""Chapter 3: atlas/helpers.py - classify/search_kb/compose_answer were
stubs until Chapter 7 filled two of them in for real: classify moved to
atlas.triage, and KnowledgeBaseUnavailable moved to atlas.tools alongside
the real search_kb tool. search_kb and compose_answer stay stubs here -
atlas/graph.py's retrieve/answer nodes aren't rewired to the real tools
yet; see atlas/helpers.py's module docstring."""

import pytest

from atlas import helpers


def test_search_kb_is_still_a_stub_after_chapter_7():
    with pytest.raises(NotImplementedError):
        helpers.search_kb([])


def test_compose_answer_is_still_a_stub_after_chapter_7():
    with pytest.raises(NotImplementedError):
        helpers.compose_answer([], [])


def test_classify_no_longer_lives_in_helpers():
    """Chapter 7 moves classify to atlas.triage - confirm it is gone from
    here rather than silently drifting out of sync with the real one."""
    assert not hasattr(helpers, "classify")


def test_knowledge_base_unavailable_no_longer_lives_in_helpers():
    """Chapter 7 moves this exception to atlas.tools, next to the real
    search_kb tool that can raise it."""
    assert not hasattr(helpers, "KnowledgeBaseUnavailable")
