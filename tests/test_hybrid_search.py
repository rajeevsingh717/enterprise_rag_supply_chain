"""Tests for hybrid dense+sparse retrieval (§3.D) — a fake Qdrant client
stands in so these run without a live Qdrant instance."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_supply_chain.config import settings
from rag_supply_chain.retrieval.hybrid_search import RetrievedChunk, hybrid_search


@dataclass
class _Point:
    payload: dict
    score: float


@dataclass
class _QueryResponse:
    points: list[_Point] = field(default_factory=list)


class FakeQdrantClient:
    def __init__(self, points: list[_Point]) -> None:
        self._points = points
        self.last_call: dict | None = None

    def query_points(self, **kwargs):
        self.last_call = kwargs
        return _QueryResponse(points=self._points)


class FakeDenseEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * settings.embedding_dim for _ in texts]


def _point(chunk_id: str, score: float) -> _Point:
    return _Point(
        payload={
            "chunk_id": chunk_id,
            "text": f"text for {chunk_id}",
            "doc_title": "Doc",
            "source_uri": "/tmp/doc.md",
            "hierarchical_context": ["Section"],
        },
        score=score,
    )


def test_hybrid_search_returns_fused_results_in_order() -> None:
    client = FakeQdrantClient([_point("a", 0.9), _point("b", 0.5)])
    results = hybrid_search(client, "CIDR block sizing", FakeDenseEmbedder())

    assert [r.chunk_id for r in results] == ["a", "b"]
    assert all(isinstance(r, RetrievedChunk) for r in results)
    assert results[0].score == 0.9
    assert results[0].hierarchical_context == ["Section"]


def test_hybrid_search_uses_both_dense_and_sparse_prefetch() -> None:
    client = FakeQdrantClient([])
    hybrid_search(client, "some query", FakeDenseEmbedder())

    prefetch = client.last_call["prefetch"]
    assert len(prefetch) == 2
    using_names = {p.using for p in prefetch}
    assert using_names == {"dense", "sparse"}


def test_hybrid_search_respects_top_k_override() -> None:
    client = FakeQdrantClient([])
    hybrid_search(client, "q", FakeDenseEmbedder(), top_k=3)

    assert client.last_call["limit"] == 3


def test_hybrid_search_empty_collection_returns_empty_list() -> None:
    client = FakeQdrantClient([])
    results = hybrid_search(client, "q", FakeDenseEmbedder())
    assert results == []
