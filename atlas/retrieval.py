"""Appendix G, "Retrieval for Atlas (the RAG the Book Assumes)" - what
actually sits behind atlas/tools.py's `search_kb`.

Chapter 7 gave `search_kb` an interface - a query in, a ranked result or a
"no match" string out - and treated everything behind that interface as a
seeded backend, deliberately not re-taught (`atlas/tools.py`'s `_KB`, a
plain substring-matched dict). This module is the appendix's "smallest
pipeline that makes search_kb's contract real": chunk each seeded article,
embed every chunk, and answer a query with the closest chunks by cosine
similarity.

This is NOT wired in as `search_kb`'s default implementation, and importing
this module never makes a network call. `atlas/tools.py`'s dict-based
`search_kb` stays the default every other chapter's code and tests run
against - the book's own "no external account, no network call" promise
("Using the Code Examples"). The embedding pipeline here is the explicit,
optional seam a real deployment swaps in: the same dev/prod split as
Chapter 13's `build_dev_store`/`build_prod_store`. It needs a live
embedding provider (`OPENAI_API_KEY`, `langchain-openai` installed), so it
is exercised behind a skip guard in `tests/test_retrieval.py`, never on the
default test path. `default_embeddings()` only imports/calls
`init_embeddings` when a caller actually asks for it - constructing an
`EmbeddedKB` (or calling `search_kb_impl`/`recall_at_k`) without an
explicit `embeddings=` argument.

Reuses Chapter 13's `text-embedding-3-small` (1536 dimensions) rather than
a second embedding model - the same "keep the embedding model fixed across
write and query" discipline Chapter 13 argued for semantic memory applies
here too.
"""

from __future__ import annotations

import math
from typing import Protocol

from atlas.tools import _KB


class Embeddings(Protocol):
    """The narrow shape this module needs from an embeddings client - just
    `embed_query`, so a hand-rolled fake and a real `langchain` embeddings
    object both satisfy it without either depending on the other."""

    def embed_query(self, text: str) -> list[float]: ...


def default_embeddings() -> Embeddings:
    """Lazy import + construction, exactly like Chapter 13's
    `build_prod_store` - never called at import time, so importing this
    module carries no network dependency. Raises immediately (via
    `init_embeddings`) if `OPENAI_API_KEY` is not set, rather than failing
    later on the first real query."""
    from langchain.embeddings import init_embeddings

    return init_embeddings("openai:text-embedding-3-small")


def chunk_article(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    """Fixed-size chunking with overlap - the simplest strategy that keeps
    a chunk's meaning intact without splitting mid-sentence too often. Not
    a semantic chunker; the seeded KB is small enough not to need one."""
    if size <= overlap:
        raise ValueError("size must be greater than overlap")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors. Returns 0.0 for a
    degenerate (all-zero) vector instead of raising a divide-by-zero
    error."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddedKB:
    """Chunks and embeds `atlas.tools._KB` once, then answers a query with
    the closest chunks by cosine similarity - Appendix G's "smallest
    pipeline that makes search_kb's contract real," built over the SAME
    seeded articles Chapter 7's dict-based `search_kb` already uses, not a
    second knowledge base.

    Building the index calls `embed_query` once per chunk, so it is lazy:
    nothing is embedded until the first `.search()`/`.recall_at_k()` call,
    and the index is cached after that."""

    def __init__(self, embeddings: Embeddings | None = None) -> None:
        self._embeddings = embeddings if embeddings is not None else default_embeddings()
        self._index: list[tuple[str, str, list[float]]] | None = None

    def _build_index(self) -> list[tuple[str, str, list[float]]]:
        if self._index is None:
            self._index = [
                (article_id, chunk, self._embeddings.embed_query(chunk))
                for article_id, text in _KB.items()
                for chunk in chunk_article(text)
            ]
        return self._index

    def search(self, query: str, k: int = 3) -> list[str]:
        """The appendix's `search_kb_impl`: the top-k chunks by cosine
        similarity, filtered below score 0.3 (treat as no match)."""
        query_vec = self._embeddings.embed_query(query)
        scored = [
            (cosine_similarity(query_vec, vec), chunk)
            for _article_id, chunk, vec in self._build_index()
        ]
        top = sorted(scored, key=lambda pair: pair[0], reverse=True)[:k]
        return [chunk for score, chunk in top if score > 0.3]

    def recall_at_k(self, query: str, expected_article_id: str, k: int = 3) -> bool:
        """Did the expected article's content place in the top k? A
        retrieval-only metric - no model call, no agent involved - checked
        BEFORE trusting any agent-level eval result that depends on
        `search_kb` having worked.

        The appendix's own sketch checks `expected_article_id in chunk`
        (a substring match against the chunk TEXT); that only works if an
        article's id happens to appear inside its own body text, which
        `atlas.tools._KB`'s seeded articles do not guarantee (its keys are
        short phrases like "refund window", and the article bodies never
        repeat them verbatim). This implementation checks article-id
        membership in the top-k index entries instead - the same question,
        answered correctly against the real seeded KB shape."""
        query_vec = self._embeddings.embed_query(query)
        scored = [
            (cosine_similarity(query_vec, vec), article_id)
            for article_id, _chunk, vec in self._build_index()
        ]
        top_ids = {article_id for _score, article_id in sorted(scored, reverse=True)[:k]}
        return expected_article_id in top_ids


def search_kb_impl(query: str, embeddings: Embeddings | None = None, k: int = 3) -> list[str]:
    """Module-level convenience wrapper matching the appendix's own
    function name - builds a throwaway `EmbeddedKB` per call. Prefer
    constructing one `EmbeddedKB` and reusing it (the index is cached)
    when answering more than one query."""
    return EmbeddedKB(embeddings).search(query, k=k)


def recall_at_k(
    query: str,
    expected_article_id: str,
    embeddings: Embeddings | None = None,
    k: int = 3,
) -> bool:
    """Module-level convenience wrapper - see `EmbeddedKB.recall_at_k`."""
    return EmbeddedKB(embeddings).recall_at_k(query, expected_article_id, k=k)
