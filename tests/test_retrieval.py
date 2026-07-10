"""Appendix G, "Retrieval for Atlas (the RAG the Book Assumes)" -
atlas/retrieval.py: chunk_article, cosine_similarity, EmbeddedKB, and the
module-level search_kb_impl/recall_at_k wrappers.

`EmbeddedKB.__init__` calls `default_embeddings()` (which needs a live
OPENAI_API_KEY) unless an `embeddings=` object is passed in - the same
dev/prod seam as Chapter 13's `build_dev_store`/`build_prod_store`. Every
test below passes a hand-rolled, deterministic, offline `_FakeEmbeddings`
(bag-of-hashed-words) so the pipeline's actual logic - chunking, cosine
ranking, the 0.3 no-match threshold, recall@k - is exercised with no
network call and no API key, the same no-live-API convention as
tests/test_triage.py's monkeypatched classify(). One test at the bottom,
`test_default_embeddings_requires_a_live_openai_key`, IS skip-guarded and
exercises the real `default_embeddings()` path - see `requires_openai`,
mirroring tests/test_memory.py's guard for `build_prod_store`."""

import hashlib
import os

import pytest

from atlas.retrieval import (
    EmbeddedKB,
    chunk_article,
    cosine_similarity,
    default_embeddings,
    recall_at_k,
    search_kb_impl,
)

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


# --- cosine_similarity ---------------------------------------------------


def test_cosine_similarity_of_identical_vectors_is_one():
    vec = [1.0, 2.0, 3.0]

    assert cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_handles_a_degenerate_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# --- EmbeddedKB.search ---------------------------------------------------


def test_search_returns_the_chunk_matching_the_query():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    results = kb.search("Refunds are available", k=1)

    assert results == ["Refunds are available within 30 days of purchase."]


def test_search_returns_nothing_for_a_query_sharing_no_words_with_the_kb():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    results = kb.search("zzz qqq xxx", k=3)

    assert results == []


def test_search_caches_the_index_across_calls():
    fake = _FakeEmbeddings()
    kb = EmbeddedKB(embeddings=fake)

    kb.search("Refunds are available")
    index_after_first_call = kb._index
    kb.search("Use the sign-in page")

    assert kb._index is index_after_first_call


# --- EmbeddedKB.recall_at_k ------------------------------------------------


def test_recall_at_k_finds_the_expected_article_at_k1():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    assert kb.recall_at_k("Refunds are available", "refund window", k=1) is True


def test_recall_at_k_is_false_when_the_wrong_article_is_expected():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    assert kb.recall_at_k("Refunds are available", "reset password", k=1) is False


def test_recall_at_k_finds_the_reset_password_article():
    kb = EmbeddedKB(embeddings=_FakeEmbeddings())

    assert (
        kb.recall_at_k("Use the sign-in page to reset", "reset password", k=1) is True
    )


# --- module-level convenience wrappers -----------------------------------


def test_search_kb_impl_wrapper_matches_embeddedkb_search():
    fake = _FakeEmbeddings()

    assert search_kb_impl("Refunds are available", embeddings=fake, k=1) == [
        "Refunds are available within 30 days of purchase."
    ]


def test_recall_at_k_wrapper_matches_embeddedkb_recall_at_k():
    fake = _FakeEmbeddings()

    assert recall_at_k("Refunds are available", "refund window", embeddings=fake, k=1)


# --- the live-embedding-provider seam (external-service exception) -------


@requires_openai
def test_default_embeddings_builds_a_real_client_and_answers_a_query():
    """Skipped by default - see `requires_openai` above. Exercises the
    actual, un-faked path: EmbeddedKB() with no embeddings= argument calls
    default_embeddings() -> init_embeddings('openai:text-embedding-3-small'),
    a live network call."""
    kb = EmbeddedKB()

    results = kb.search("What is the refund window?", k=1)

    assert results  # a real embedding model should surface the refund article


def test_default_embeddings_without_a_key_raises_immediately():
    """Not skip-guarded: this asserts the FAILURE mode when no
    OPENAI_API_KEY is set, so it should only run in an environment without
    one - the default CI/dev posture for this seeded, no-external-accounts
    book."""
    if os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is set in this environment")

    with pytest.raises(Exception):
        default_embeddings()
