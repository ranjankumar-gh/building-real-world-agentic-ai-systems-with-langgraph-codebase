"""Chapter 3: atlas/helpers.py - classify/search_kb/compose_answer are stubs
until Chapter 7 fills them in for real. Chapter 6 adds the
KnowledgeBaseUnavailable failure type that atlas/graph.py's retrieve node
catches."""

import pytest

from atlas import helpers
from atlas.helpers import KnowledgeBaseUnavailable


def test_classify_is_a_stub_until_chapter_7():
    with pytest.raises(NotImplementedError):
        helpers.classify([])


def test_search_kb_is_a_stub_until_chapter_7():
    with pytest.raises(NotImplementedError):
        helpers.search_kb([])


def test_compose_answer_is_a_stub_until_chapter_7():
    with pytest.raises(NotImplementedError):
        helpers.compose_answer([], [])


def test_knowledge_base_unavailable_is_a_distinct_exception_type():
    """Distinct from search_kb's stub NotImplementedError - this is the
    failure type a real, reachable-but-down knowledge base raises."""
    assert issubclass(KnowledgeBaseUnavailable, Exception)

    with pytest.raises(KnowledgeBaseUnavailable):
        raise KnowledgeBaseUnavailable("knowledge base is down")
