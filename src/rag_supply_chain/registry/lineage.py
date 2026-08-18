"""Ties the chunk registry to Qdrant (§3.C): registering a document's
current chunk set and purging whatever became stale, as one operation."""

from __future__ import annotations

from typing import Protocol

from sqlalchemy.engine import Engine

from rag_supply_chain.registry.store import register_chunks
from rag_supply_chain.workers.qdrant_store import delete_chunks


class QdrantClientLike(Protocol):
    def delete(self, collection_name: str, points_selector: list) -> None: ...


def sync_document_chunks(
    qdrant_client: QdrantClientLike,
    registry_engine: Engine,
    doc_id: str,
    chunk_ids: list[str],
) -> list[str]:
    """Register `chunk_ids` as the current version of `doc_id` and purge any
    chunks left over from a prior version. Returns the purged chunk_ids."""
    stale_ids = register_chunks(registry_engine, doc_id, chunk_ids)
    if stale_ids:
        delete_chunks(qdrant_client, stale_ids)
    return stale_ids
