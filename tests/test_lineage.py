"""Tests for tying the chunk registry to Qdrant purges (§3.C) — a fake
Qdrant client + SQLite in-memory registry stand in so these run without
live infra."""

from __future__ import annotations

from sqlalchemy import create_engine

from rag_supply_chain.registry.lineage import sync_document_chunks
from rag_supply_chain.registry.store import ensure_schema
from rag_supply_chain.workers.qdrant_store import point_id_for


class FakeQdrantClient:
    def __init__(self) -> None:
        self.deleted: list[list[str]] = []

    def delete(self, collection_name: str, points_selector: list) -> None:
        self.deleted.append(points_selector)


def _engine():
    engine = create_engine("sqlite:///:memory:")
    ensure_schema(engine)
    return engine


def test_no_purge_on_first_ingest() -> None:
    client = FakeQdrantClient()
    engine = _engine()

    stale = sync_document_chunks(client, engine, "doc-1", ["doc-1:0", "doc-1:1"])

    assert stale == []
    assert client.deleted == []


def test_purges_qdrant_when_reingested_doc_shrinks() -> None:
    client = FakeQdrantClient()
    engine = _engine()
    sync_document_chunks(client, engine, "doc-1", ["doc-1:0", "doc-1:1", "doc-1:2", "doc-1:3"])

    stale = sync_document_chunks(client, engine, "doc-1", ["doc-1:0", "doc-1:1"])

    assert stale == ["doc-1:2", "doc-1:3"]
    assert len(client.deleted) == 1
    assert client.deleted[0] == [point_id_for(cid) for cid in stale]


def test_no_qdrant_call_when_nothing_is_stale() -> None:
    client = FakeQdrantClient()
    engine = _engine()
    sync_document_chunks(client, engine, "doc-1", ["doc-1:0"])

    sync_document_chunks(client, engine, "doc-1", ["doc-1:0", "doc-1:1"])

    assert client.deleted == []
