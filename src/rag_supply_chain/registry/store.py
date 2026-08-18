"""Chunk lineage registry (§3.C): tracks which chunk_ids belong to which
ingested version of a document, so re-ingesting a modified doc can purge
vectors that no longer exist in the new version instead of leaving them
orphaned in Qdrant forever.

Requires a stable doc_id per source file (see `ingestion.producer.new_doc_id`)
— without that, there's nothing for this registry to diff a re-ingest against.
"""

from __future__ import annotations

from sqlalchemy import create_engine, delete, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from rag_supply_chain.config import settings
from rag_supply_chain.registry.models import Base, DocumentChunk


def get_engine(dsn: str | None = None) -> Engine:
    return create_engine(dsn or settings.postgres_dsn)


def ensure_schema(engine: Engine) -> None:
    Base.metadata.create_all(engine)


def register_chunks(engine: Engine, doc_id: str, chunk_ids: list[str]) -> list[str]:
    """Register `chunk_ids` as the current version of `doc_id`.

    Returns the chunk_ids previously registered for this doc_id that are NOT
    present in `chunk_ids` — the stale vectors the caller must purge from
    Qdrant, because nothing in the new version will ever overwrite them.
    """
    new_ids = set(chunk_ids)
    with Session(engine) as session:
        existing = session.scalars(select(DocumentChunk).where(DocumentChunk.doc_id == doc_id)).all()
        prior_ids = {row.chunk_id for row in existing}
        next_version = max((row.version for row in existing), default=0) + 1

        stale_ids = sorted(prior_ids - new_ids)
        if stale_ids:
            session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.doc_id == doc_id,
                    DocumentChunk.chunk_id.in_(stale_ids),
                )
            )

        carried_over = prior_ids & new_ids
        if carried_over:
            session.execute(
                update(DocumentChunk)
                .where(DocumentChunk.doc_id == doc_id, DocumentChunk.chunk_id.in_(carried_over))
                .values(version=next_version)
            )
        for chunk_id in new_ids - carried_over:
            session.add(DocumentChunk(doc_id=doc_id, version=next_version, chunk_id=chunk_id))

        session.commit()

    return stale_ids


def tombstone_document(engine: Engine, doc_id: str) -> list[str]:
    """Remove every registry row for `doc_id` and return every chunk_id that
    was registered, for the caller to wipe from Qdrant (§3.C delete path)."""
    with Session(engine) as session:
        existing = session.scalars(select(DocumentChunk).where(DocumentChunk.doc_id == doc_id)).all()
        chunk_ids = [row.chunk_id for row in existing]
        session.execute(delete(DocumentChunk).where(DocumentChunk.doc_id == doc_id))
        session.commit()
    return chunk_ids
