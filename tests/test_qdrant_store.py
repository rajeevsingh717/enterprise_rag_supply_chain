"""Tests for the Qdrant upsert path + dimensionality guard (§2.B) — a fake
Qdrant client stands in so these run without a live Qdrant instance."""

from __future__ import annotations

from dataclasses import dataclass

from rag_supply_chain.config import settings
from rag_supply_chain.workers.embeddings import SparseVectorData
from rag_supply_chain.workers.qdrant_store import ensure_collection, point_id_for, upsert_chunks


@dataclass
class _Collection:
    name: str


@dataclass
class _CollectionsResponse:
    collections: list[_Collection]


class FakeQdrantClient:
    def __init__(self, existing_collections: list[str] | None = None) -> None:
        self._collections = set(existing_collections or [])
        self.created_with: dict | None = None
        self.upserted_points: list = []

    def get_collections(self) -> _CollectionsResponse:
        return _CollectionsResponse([_Collection(n) for n in self._collections])

    def create_collection(self, collection_name: str, **kwargs) -> None:
        self._collections.add(collection_name)
        self.created_with = {"collection_name": collection_name, **kwargs}

    def upsert(self, collection_name: str, points: list) -> None:
        self.upserted_points.extend(points)


def _chunk_payload(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "parent_doc_id": "doc-1", "text": "hello world"}


def test_ensure_collection_creates_when_missing() -> None:
    client = FakeQdrantClient(existing_collections=[])
    ensure_collection(client)
    assert settings.qdrant_collection in client._collections
    assert client.created_with is not None


def test_ensure_collection_is_idempotent() -> None:
    client = FakeQdrantClient(existing_collections=[settings.qdrant_collection])
    ensure_collection(client)
    assert client.created_with is None  # already existed, no create_collection call


def test_upsert_accepts_correctly_dimensioned_vectors() -> None:
    client = FakeQdrantClient(existing_collections=[settings.qdrant_collection])
    payloads = [_chunk_payload("a"), _chunk_payload("b")]
    dense = [[0.1] * settings.embedding_dim, [0.2] * settings.embedding_dim]
    sparse = [SparseVectorData([1], [1.0]), SparseVectorData([2], [1.0])]

    result = upsert_chunks(client, payloads, dense, sparse)

    assert result.upserted_chunk_ids == ["a", "b"]
    assert result.dropped == []
    assert len(client.upserted_points) == 2


def test_upsert_drops_mismatched_dimension_vector() -> None:
    client = FakeQdrantClient(existing_collections=[settings.qdrant_collection])
    payloads = [_chunk_payload("good"), _chunk_payload("bad")]
    dense = [[0.1] * settings.embedding_dim, [0.1] * 7]  # wrong dim for "bad"
    sparse = [SparseVectorData([1], [1.0]), SparseVectorData([2], [1.0])]

    result = upsert_chunks(client, payloads, dense, sparse)

    assert result.upserted_chunk_ids == ["good"]
    assert len(result.dropped) == 1
    assert result.dropped[0][0] == "bad"
    assert len(client.upserted_points) == 1


def test_point_id_for_is_stable_and_valid_uuid() -> None:
    import uuid

    pid = point_id_for("doc-1:0")
    assert pid == point_id_for("doc-1:0")
    uuid.UUID(pid)  # does not raise
