"""Tests for the chunk lineage registry (§3.C) — SQLite in-memory stands in
for Postgres so these run without a live database."""

from __future__ import annotations

from sqlalchemy import create_engine

from rag_supply_chain.registry.store import ensure_schema, register_chunks, tombstone_document


def _engine():
    engine = create_engine("sqlite:///:memory:")
    ensure_schema(engine)
    return engine


def test_first_version_registers_cleanly_with_no_stale_ids() -> None:
    engine = _engine()
    stale = register_chunks(engine, "doc-1", ["doc-1:0", "doc-1:1", "doc-1:2"])
    assert stale == []


def test_shrinking_doc_reports_dropped_indices_as_stale() -> None:
    engine = _engine()
    register_chunks(engine, "doc-1", ["doc-1:0", "doc-1:1", "doc-1:2", "doc-1:3", "doc-1:4"])

    stale = register_chunks(engine, "doc-1", ["doc-1:0", "doc-1:1", "doc-1:2"])

    assert stale == ["doc-1:3", "doc-1:4"]


def test_growing_doc_reports_no_stale_ids() -> None:
    engine = _engine()
    register_chunks(engine, "doc-1", ["doc-1:0", "doc-1:1"])

    stale = register_chunks(engine, "doc-1", ["doc-1:0", "doc-1:1", "doc-1:2", "doc-1:3"])

    assert stale == []


def test_reregistering_the_same_chunk_set_reports_no_stale_ids() -> None:
    engine = _engine()
    register_chunks(engine, "doc-1", ["doc-1:0", "doc-1:1"])

    stale = register_chunks(engine, "doc-1", ["doc-1:0", "doc-1:1"])

    assert stale == []


def test_different_docs_do_not_interfere() -> None:
    engine = _engine()
    register_chunks(engine, "doc-1", ["doc-1:0", "doc-1:1", "doc-1:2"])

    stale = register_chunks(engine, "doc-2", ["doc-2:0"])

    assert stale == []


def test_tombstone_returns_all_chunk_ids_and_clears_registry() -> None:
    engine = _engine()
    register_chunks(engine, "doc-1", ["doc-1:0", "doc-1:1", "doc-1:2"])

    chunk_ids = tombstone_document(engine, "doc-1")
    assert sorted(chunk_ids) == ["doc-1:0", "doc-1:1", "doc-1:2"]

    # registry is now empty for doc-1 — a fresh register sees no prior version
    stale = register_chunks(engine, "doc-1", ["doc-1:0"])
    assert stale == []


def test_tombstone_of_unknown_doc_returns_empty_list() -> None:
    engine = _engine()
    assert tombstone_document(engine, "never-seen") == []
