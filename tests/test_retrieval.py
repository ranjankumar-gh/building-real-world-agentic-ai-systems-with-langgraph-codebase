"""Appendix G, "Retrieval for Atlas (the RAG the Book Assumes)" -
atlas/retrieval.py: chunk_article, cosine_similarity, EmbeddedKB, the
search_kb_embedded node adapter and its bound drop-in make_search_kb, the
lock around the lazy index, and the search_kb_impl/recall_at_k functions
the appendix prints.

`EmbeddedKB` calls `default_embeddings()` (which needs a live
OPENAI_API_KEY) on its first search unless an `embeddings=` object is
passed in. Every test below passes a hand-rolled, deterministic, offline
`_FakeEmbeddings` (bag-of-hashed-words) or a `_FailingEmbeddings` that
stands in for a provider outage, so the pipeline's logic - chunking, cosine
ranking, the 0.3 no-match threshold, recall@k, and the escalation of a
provider failure through `retrieve` - runs with no network call and no API
key. One test at the bottom,
`test_default_embeddings_builds_a_real_client_and_answers_a_query`, IS
skip-guarded and exercises the real `default_embeddings()` path - see
`requires_openai`, mirroring tests/test_memory.py's guard for
`build_prod_store`."""

import hashlib
import os
import threading
import time

import pytest

from langchain_core.messages import HumanMessage

import atlas.graph as graph
import atlas.retrieval as retrieval
from atlas.context import select_docs
from atlas.retrieval import (
    EmbeddedKB,
    chunk_article,
    cosine_similarity,
    default_embeddings,
    make_search_kb,
    recall_at_k,
    search_kb_embedded,
    search_kb_impl,
)
from atlas.tools import _KB, KnowledgeBaseUnavailable

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="requires a live embedding provider (OPENAI_API_KEY)",
)


class _FakeEmbeddings:
    """Deterministic, offline stand-in for a real embeddings client: a
    bag-of-hashed-words vector, so texts sharing words score a higher
    cosine similarity than texts that share none - just enough signal to
    exercise ranking and the no-match threshold without a network call."""

    def __init__(self, dims: int = 64) -> None:
        self.dims = dims

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in text.lower().split():
            idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims
            vec[idx] += 1.0
        return vec


# --- chunk_article -----------------------------------------------------


def test_chunk_article_splits_long_text_with_overlap():
    # a run of distinct characters makes the overlap boundary checkable
    text = "".join(str(i % 10) for i in range(1000))
    chunks = chunk_article(text, size=400, overlap=50)

    assert len(chunks) == 3
    assert chunks[0] == text[0:400]
    assert chunks[1] == text[350:750]
    assert chunks[2] == text[700:1000]
    # the last 50 characters of chunk 0 equal the first 50 of chunk 1
    assert chunks[0][-50:] == chunks[1][:50]


def test_chunk_article_returns_the_whole_text_as_one_chunk_when_short():
    text = "Refunds are available within 30 days of purchase."
    chunks = chunk_article(text)

    assert chunks == [text]


def test_chunk_article_rejects_a_non_advancing_window():
    with pytest.raises(ValueError):
        chunk_article("some text", size=50, overlap=50)


def test_chunk_article_rejects_a_negative_overlap():
    with pytest.raises(ValueError):
        chunk_article("some text", size=50, overlap=-1)


def test_every_seeded_article_is_a_single_chunk():
    # the overlap is there for a real KB; the two seeded articles never split
    assert all(len(chunk_article(text)) == 1 for text in _KB.values())


# --- cosine_similarity ---------------------------------------------------


def test_cosine_similarity_of_identical_vectors_is_one():
    vec = [1.0, 2.0, 3.0]

    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_handles_a_degenerate_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_cosine_similarity_rejects_vectors_of_different_lengths():
    # zip() would silently truncate: two embedding models mixed in one index
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0, 1.0], [1.0, 0.0])


# --- EmbeddedKB.search ---------------------------------------------------

REFUND = "Refunds are available within 30 days of purchase."


class _FailingEmbeddings:
    """A provider that is down: every call raises its own error type."""

    def __init__(self) -> None:
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        raise ConnectionError("embedding provider unreachable")


def test_search_returns_scored_docs_carrying_the_article_id():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    results = kb.search("Refunds are available", k=1)

    assert len(results) == 1
    doc = results[0]
    assert doc["id"] == "kb:refund window#0"
    assert doc["text"] == REFUND
    assert 0.3 < doc["score"] <= 1.0


def test_search_returns_nothing_for_a_query_sharing_no_words_with_the_kb():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    assert kb.search("zzz qqq xxx", k=3) == []


def test_search_ranks_by_score_alone():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    results = kb.search("Refunds are available on the sign-in page", k=2)

    scores = [doc["score"] for doc in results]
    assert scores == sorted(scores, reverse=True)


def test_search_docs_go_through_select_docs():
    # the shape Chapter 12's select_docs ranks and caps: no TypeError
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    kept = select_docs(kb.search("Refunds are available"), 2000)

    assert kept and kept[0]["text"] == REFUND


def test_search_caches_the_index_across_calls():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    kb.search("Refunds are available")
    index_after_first_call = kb._index
    kb.search("Use the sign-in page")

    assert kb._index is index_after_first_call


def test_constructing_a_kb_needs_no_key_and_makes_no_call(monkeypatch):
    def boom() -> None:
        raise AssertionError("default_embeddings called before the first search")

    monkeypatch.setattr(retrieval, "default_embeddings", boom)

    kb = EmbeddedKB()

    assert kb._index is None


def test_a_provider_failure_raises_knowledge_base_unavailable():
    kb = EmbeddedKB(embeddings=_FailingEmbeddings())

    with pytest.raises(KnowledgeBaseUnavailable):
        kb.search("Refunds are available")


def test_a_missing_key_on_first_search_raises_knowledge_base_unavailable(
    monkeypatch,
):
    def no_key() -> None:
        raise RuntimeError("Missing credentials")  # what init_embeddings raises

    monkeypatch.setattr(retrieval, "default_embeddings", no_key)

    with pytest.raises(KnowledgeBaseUnavailable):
        EmbeddedKB().search("Refunds are available")


# --- the node adapter, through retrieve ------------------------------------


def _ask(question: str) -> dict:
    return {"messages": [HumanMessage(question)], "retrieve_attempts": 0}


def test_search_kb_embedded_reads_the_last_user_turn():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    docs = search_kb_embedded([HumanMessage("Refunds are available")], kb)

    assert [doc["id"] for doc in docs][:1] == ["kb:refund window#0"]


def test_search_kb_embedded_with_no_question_returns_nothing():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    assert search_kb_embedded([], kb) == []


def test_retrieve_with_the_embedded_adapter_routes_a_hit_to_answer(monkeypatch):
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())
    monkeypatch.setattr(graph, "search_kb", make_search_kb(kb))

    state = _ask("Refunds are available")
    update = graph.retrieve(state)

    assert update["retrieved"][0]["id"] == "kb:refund window#0"
    assert graph.route_after_retrieve({**state, **update}) == "answer"


def test_retrieve_with_the_embedded_adapter_escalates_a_provider_outage(
    monkeypatch,
):
    failing = _FailingEmbeddings()
    kb = EmbeddedKB(embeddings=failing)
    monkeypatch.setattr(graph, "search_kb", make_search_kb(kb))

    state = _ask("Refunds are available")
    update = graph.retrieve(state)

    assert "embedding provider" in update["error"]
    assert graph.route_after_retrieve({**state, **update}) == "escalate"
    assert failing.calls == 1  # recorded once, not retried


def test_retrieve_with_the_embedded_adapter_retries_a_miss_then_escalates(
    monkeypatch,
):
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())
    monkeypatch.setattr(graph, "search_kb", make_search_kb(kb))

    state = _ask("zzz qqq xxx")
    routes = []
    for _ in range(graph.MAX_RETRIEVE_ATTEMPTS):
        state = {**state, **graph.retrieve(state)}
        routes.append(graph.route_after_retrieve(state))

    assert routes == ["retrieve", "retrieve", "escalate"]


def test_make_search_kb_is_a_one_argument_drop_in_for_helpers_search_kb():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())
    search = make_search_kb(kb)

    question = [HumanMessage("Refunds are available")]
    assert search(question) == search_kb_embedded(question, kb)
    assert search([]) == []


def test_make_search_kb_with_no_kb_searches_the_shared_index(monkeypatch):
    monkeypatch.setattr(retrieval, "_default_kb", None)
    monkeypatch.setattr(retrieval, "default_embeddings", _FakeEmbeddings)

    docs = make_search_kb()([HumanMessage("Refunds are available")])

    assert docs and docs[0]["id"] == "kb:refund window#0"
    assert retrieval._default_kb is not None


class _SlowCountingEmbeddings(_FakeEmbeddings):
    """Slow enough that unsynchronised first searches would overlap."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self._count_lock = threading.Lock()

    def embed_query(self, text: str) -> list[float]:
        with self._count_lock:
            self.calls += 1
        time.sleep(0.01)
        return super().embed_query(text)


def _at_once(n: int, fn) -> list:
    barrier = threading.Barrier(n)
    results: list = [None] * n

    def run(i: int) -> None:
        barrier.wait()
        results[i] = fn()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_concurrent_first_searches_build_the_index_once():
    embeddings = _SlowCountingEmbeddings()
    kb = EmbeddedKB(embeddings=embeddings)
    chunks = sum(len(chunk_article(text)) for text in _KB.values())

    _at_once(8, lambda: kb.search("Refunds are available"))

    assert embeddings.calls == chunks + 8  # one build, one query embed each


def test_concurrent_first_callers_share_one_default_kb(monkeypatch):
    monkeypatch.setattr(retrieval, "_default_kb", None)
    built: list[EmbeddedKB] = []

    class _SlowKB(EmbeddedKB):
        def __init__(self) -> None:
            time.sleep(0.01)
            super().__init__(embeddings=_FakeEmbeddings())
            built.append(self)

    monkeypatch.setattr(retrieval, "EmbeddedKB", _SlowKB)

    kbs = _at_once(8, retrieval.shared_kb)

    assert len(built) == 1 and all(kb is built[0] for kb in kbs)


# --- search_kb_impl and recall_at_k ----------------------------------------


def test_search_kb_impl_matches_embeddedkb_search():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    assert search_kb_impl("Refunds are available", k=1, kb=kb) == kb.search(
        "Refunds are available", k=1
    )


def test_search_kb_impl_builds_one_shared_index_lazily(monkeypatch):
    monkeypatch.setattr(retrieval, "_default_kb", None)
    monkeypatch.setattr(retrieval, "default_embeddings", _FakeEmbeddings)

    search_kb_impl("Refunds are available")
    shared = retrieval._default_kb
    search_kb_impl("Use the sign-in page")

    assert shared is not None and retrieval._default_kb is shared


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How long are refunds available after purchase?", "refund window"),
        ("Where is the forgot password link?", "reset password"),
    ],
)
def test_recall_at_k_passes_on_the_seeded_kb(query, expected):
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    assert recall_at_k(query, expected, k=1, kb=kb) is True


def test_recall_at_k_is_false_when_the_wrong_article_is_expected():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    assert recall_at_k("Refunds are available", "reset password", k=1, kb=kb) is False


def test_recall_at_k_is_false_on_a_miss():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    assert recall_at_k("zzz qqq xxx", "refund window", kb=kb) is False


# --- the live-embedding-provider seam (external-service exception) -------


@requires_openai
def test_default_embeddings_builds_a_real_client_and_answers_a_query():
    """Skipped by default - see `requires_openai` above. Exercises the
    actual, un-faked path: EmbeddedKB() with no embeddings= argument calls
    default_embeddings() -> init_embeddings('openai:text-embedding-3-small'),
    a live network call, on its first search."""
    kb = EmbeddedKB()

    results = kb.search("What is the refund window?", k=1)

    assert results and results[0]["id"] == "kb:refund window#0"


def test_default_embeddings_without_a_key_raises_immediately():
    """Not skip-guarded: this asserts the FAILURE mode when no
    OPENAI_API_KEY is set, so it should only run in an environment without
    one - the default CI/dev posture for this seeded, no-external-accounts
    book."""
    if os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is set in this environment")

    with pytest.raises(Exception):
        default_embeddings()
