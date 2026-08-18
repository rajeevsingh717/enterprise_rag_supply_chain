"""SQLAlchemy schema for the chunk lineage registry (§3.C).

One row per (doc_id, chunk_id): which chunks currently belong to which
ingested version of a document. Diffing a new version's chunk_id set
against a document's existing rows is how we detect chunks that no longer
exist in the new version (e.g. a doc re-chunked into fewer pieces) — those
are the stale vectors that must be purged from Qdrant. Chunk_ids shared
across versions (same doc_id:index, unchanged boundary) don't need a
diff — the embedding worker just overwrites that point.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, DateTime, Index, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    # Integer, not BigInteger: SQLite only aliases the primary key to its
    # implicit ROWID autoincrement for a column of exactly type INTEGER —
    # BigInteger maps to BIGINT there and silently loses autoincrement,
    # which fails every insert with a NOT NULL constraint on `id`. Postgres
    # (the real target) is unaffected either way.
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    chunk_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    created_at = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_document_chunks_doc_id", "doc_id"),)
