"""Appendix G, "Retrieval for Atlas (the RAG the Book Assumes)" - what
could sit behind atlas/tools.py's `search_kb`.

Chapter 7 gave `search_kb` an interface - a query in, an article or a "no
match" string out - and treated everything behind it as a seeded backend
(`atlas/tools.py`'s `_KB`, a plain substring-matched dict). The graph never
calls that tool directly: `atlas/helpers.py`'s `search_kb` adapts it to the
node convention, messages in and scored `Doc`s out, which Chapter 12's
`select_docs` ranks and caps. This module is the appendix's smallest
embedding pipeline that honours the same node contract: chunk each seeded
article, embed every chunk, and answer a query with the closest chunks as
`Doc(id, text=chunk, score=cosine)`.

It is NOT wired in as the default. `atlas/helpers.py`'s dict-backed adapter
stays what every chapter's code and tests run against - the book's "no
external account, no network call" promise. `make_search_kb()` returns the
drop-in a deployment puts in that adapter's place: a one-argument
`search(messages)`, the same signature as `atlas.helpers.search_kb`, with
the index bound (the shared one unless you pass your own `EmbeddedKB`).

The index is built once, on the first search, under a lock: `retrieve`'s
async form runs searches on worker threads, and concurrent first requests
would otherwise each embed the whole knowledge base.

Nothing here touches the provider until the first search: importing the
module, or constructing an `EmbeddedKB`, needs no key. Every provider call
goes through `EmbeddedKB._embed`, which turns any provider failure into
`KnowledgeBaseUnavailable`, the error Chapter 6's `retrieve` node records
and escalates on; a raw provider exception would instead be retried by
`retrieve`'s `RetryPolicy` and then fail the run.

Reuses Chapter 13's `text-embedding-3-small` (1536 dimensions): keep one
embedding model across write and query.
"""

from __future__ import annotations

import functools
import math
import threading
from collections.abc import Callable
from typing import Protocol

from atlas.helpers import _last_user_text
from atlas.state import Doc
from atlas.tools import _KB, KnowledgeBaseUnavailable

NO_MATCH_BELOW = 0.3  # a starting point; set it from your recall_at_k set


class Embeddings(Protocol):
    """The narrow shape this module needs from an embeddings client - just
    `embed_query`, so a hand-rolled fake and a real `langchain` embeddings
    object both satisfy it without either depending on the other."""

    def embed_query(self, text: str) -> list[float]: ...


def default_embeddings() -> Embeddings:
    """Chapter 13's model, imported and built only when called. Raises at
    once if `OPENAI_API_KEY` is not set."""
    from langchain.embeddings import init_embeddings

    return init_embeddings("openai:text-embedding-3-small")


def chunk_article(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    """Fixed-size windows with overlap: a sentence cut at one boundary
    reads whole in the neighboring chunk."""
    if overlap < 0 or size <= overlap:
        raise ValueError("need 0 <= overlap < size, or the window never advances")
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """0.0 for an all-zero vector rather than a division by zero."""
    if len(a) != len(b):
        raise ValueError(f"vectors differ in length: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class EmbeddedKB:
    """`_KB`'s articles, chunked and embedded on first search, then cached."""

    def __init__(self, embeddings: Embeddings | None = None) -> None:
        self._embeddings = embeddings  # None: default_embeddings() on first use
        self._index: list[tuple[str, str, list[float]]] | None = None
        self._lock = threading.RLock()  # one build, however many first requests

    def _embed(self, text: str) -> list[float]:
        try:
            with self._lock:
                if self._embeddings is None:
                    self._embeddings = default_embeddings()
            return self._embeddings.embed_query(text)
        except Exception as exc:  # any provider failure: escalate, never retry
            raise KnowledgeBaseUnavailable(f"embedding provider: {exc}") from exc

    def _build_index(self) -> list[tuple[str, str, list[float]]]:
        with self._lock:  # a failed build leaves None, so the next one retries
            if self._index is None:
                self._index = [
                    (f"kb:{article_id}#{n}", chunk, self._embed(chunk))
                    for article_id, text in _KB.items()
                    for n, chunk in enumerate(chunk_article(text))
                ]
            return self._index

    def search(self, query: str, k: int = 3) -> list[Doc]:
        """The top k chunks by cosine similarity, as the `Doc`s the
        `retrieve` node expects. Below `NO_MATCH_BELOW`, nothing."""
        query_vec = self._embed(query)
        docs = [
            Doc(id=doc_id, text=chunk, score=cosine_similarity(query_vec, vec))
            for doc_id, chunk, vec in self._build_index()
        ]
        top = sorted(docs, key=lambda doc: doc["score"], reverse=True)[:k]
        return [doc for doc in top if doc["score"] > NO_MATCH_BELOW]


_default_kb: EmbeddedKB | None = None
_default_kb_lock = threading.Lock()


def shared_kb() -> EmbeddedKB:
    """The one process-wide `EmbeddedKB`, created on first use."""
    global _default_kb
    with _default_kb_lock:
        if _default_kb is None:
            _default_kb = EmbeddedKB()
        return _default_kb


def search_kb_embedded(messages: list, kb: EmbeddedKB | None = None) -> list[Doc]:
    """`atlas.helpers.search_kb` over embeddings: same messages in, same
    `Doc`s out, so `retrieve`, `select_docs` and the escalation path are
    unchanged. Searches the shared index unless given a `kb`."""
    query = _last_user_text(messages)
    if not query:
        return []
    return (kb if kb is not None else shared_kb()).search(query)


def make_search_kb(kb: EmbeddedKB | None = None) -> Callable[[list], list[Doc]]:
    """The drop-in: `search(messages)`, with `kb` bound."""
    return functools.partial(search_kb_embedded, kb=kb)


def search_kb_impl(query: str, k: int = 3, kb: EmbeddedKB | None = None) -> list[Doc]:
    """The appendix's name for a search against one shared, lazily built
    index (or the `kb` passed in)."""
    return (kb if kb is not None else shared_kb()).search(query, k=k)


def recall_at_k(
    query: str, expected_article_id: str, k: int = 3, kb: EmbeddedKB | None = None
) -> bool:
    """Did a chunk of the expected article place in the top k? Checked on
    the article id each `Doc` carries, not the chunk text: `_KB`'s ids
    ("refund window") never appear in their own articles."""
    results = search_kb_impl(query, k=k, kb=kb)
    return any(doc["id"].startswith(f"kb:{expected_article_id}#") for doc in results)
