"""Tests for the embedding Celery task (§3.B): retry-with-backoff-and-jitter,
DLQ routing on exhausted retries, and the dimensionality-guard DLQ path.

Runs Celery in eager mode (synchronous, in-process) so these are fast and
need no live Redis broker. `task_eager_propagates` is left at its default
(False), which is what makes eager mode actually re-invoke the task body on
`self.retry()` instead of just raising — verified empirically, since it's
easy to get backwards.
"""

from __future__ import annotations

from rag_supply_chain.config import settings
from rag_supply_chain.workers import tasks
from rag_supply_chain.workers.celery_app import celery_app

celery_app.conf.task_always_eager = True
celery_app.conf.task_eager_propagates = False


class FakeDenseEmbedder:
    def __init__(self, dim: int = settings.embedding_dim, fail_times: int = 0) -> None:
        self.dim = dim
        self.fail_times = fail_times
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("simulated transient embedding-API failure")
        return [[0.1] * self.dim for _ in texts]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.upserted_points: list = []

    def get_collections(self):
        class _R:
            collections: list = []

        return _R()

    def create_collection(self, **kwargs) -> None:
        pass

    def upsert(self, collection_name: str, points: list) -> None:
        self.upserted_points.extend(points)


def _chunk(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "parent_doc_id": "doc-1", "text": "hello world"}


def _patch_components(monkeypatch, dense_embedder, qdrant_client) -> list[tuple[list[dict], str]]:
    dlq_calls: list[tuple[list[dict], str]] = []
    monkeypatch.setattr(tasks, "_get_dense_embedder", lambda: dense_embedder)
    monkeypatch.setattr(tasks, "_get_qdrant_client", lambda: qdrant_client)
    monkeypatch.setattr(
        tasks, "_route_to_dlq", lambda payloads, reason: dlq_calls.append((payloads, reason))
    )
    return dlq_calls


def test_successful_batch_upserts_and_reports_ok(monkeypatch) -> None:
    dense = FakeDenseEmbedder()
    qdrant = FakeQdrantClient()
    dlq_calls = _patch_components(monkeypatch, dense, qdrant)

    result = tasks.embed_and_upsert_batch.apply(args=[[_chunk("a"), _chunk("b")]]).get()

    assert result["status"] == "ok"
    assert result["upserted"] == 2
    assert result["dropped"] == 0
    assert result["normalization"]["before_tokens"] == 4
    assert len(qdrant.upserted_points) == 2
    assert dlq_calls == []


def test_transient_failure_then_success_does_not_hit_dlq(monkeypatch) -> None:
    dense = FakeDenseEmbedder(fail_times=2)  # fails twice, succeeds on the 3rd attempt
    qdrant = FakeQdrantClient()
    dlq_calls = _patch_components(monkeypatch, dense, qdrant)

    result = tasks.embed_and_upsert_batch.apply(args=[[_chunk("a")]]).get()

    assert result["status"] == "ok"
    assert dense.calls == 3
    assert dlq_calls == []


def test_repeated_failure_exhausts_retries_and_routes_to_dlq(monkeypatch) -> None:
    dense = FakeDenseEmbedder(fail_times=999)  # always fails
    qdrant = FakeQdrantClient()
    dlq_calls = _patch_components(monkeypatch, dense, qdrant)

    result = tasks.embed_and_upsert_batch.apply(args=[[_chunk("a"), _chunk("b")]]).get()

    assert result["status"] == "dlq"
    assert result["count"] == 2
    # initial attempt + embed_max_retries retries, then give up
    assert dense.calls == settings.embed_max_retries + 1
    assert len(dlq_calls) == 1
    assert dlq_calls[0][0] == [_chunk("a"), _chunk("b")]


def test_dimensionality_mismatch_drops_and_routes_only_that_chunk_to_dlq(monkeypatch) -> None:
    dense = FakeDenseEmbedder(dim=7)  # wrong dim vs settings.embedding_dim
    qdrant = FakeQdrantClient()
    dlq_calls = _patch_components(monkeypatch, dense, qdrant)

    result = tasks.embed_and_upsert_batch.apply(args=[[_chunk("a")]]).get()

    assert result["status"] == "ok"
    assert result["upserted"] == 0
    assert result["dropped"] == 1
    assert len(qdrant.upserted_points) == 0
    assert len(dlq_calls) == 1
    assert dlq_calls[0][0][0]["chunk_id"] == "a"
    assert "dimensionality" in dlq_calls[0][1]


def test_compute_backoff_grows_with_retry_count_and_respects_cap() -> None:
    import random

    random.seed(0)
    low = tasks.compute_backoff(0)
    high = tasks.compute_backoff(10)  # far past the cap
    assert 0 <= low <= settings.embed_backoff_base_seconds
    assert 0 <= high <= settings.embed_backoff_max_seconds


def test_sparse_vector_data_roundtrips_into_upsert() -> None:
    # sanity check that tasks.process_batch wires dense+sparse together correctly
    dense = FakeDenseEmbedder()
    qdrant = FakeQdrantClient()
    result = tasks.process_batch([_chunk("a")], dense, qdrant)
    assert result.upserted_chunk_ids == ["a"]
    point = qdrant.upserted_points[0]
    assert len(point.vector["dense"]) == settings.embedding_dim
    assert len(point.vector["sparse"].indices) > 0  # "hello world" has real terms
    assert len(point.vector["sparse"].indices) == len(point.vector["sparse"].values)
