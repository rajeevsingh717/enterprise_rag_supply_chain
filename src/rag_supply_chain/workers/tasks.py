"""Celery task: embed a batch of chunks and upsert to Qdrant (§3.B).

On a transient failure the task retries with exponential backoff **and
jitter** — the "full jitter" formula (AWS's architecture-blog formulation):
`sleep = random(0, min(cap, base * 2**attempt))`. This spreads retrying
workers out in time instead of having them all hammer the dependency again
at the exact same moment (the "thundering herd" the portfolio doc names).
After `settings.embed_max_retries` attempts, the batch is routed to
`chunks.dlq` instead of being retried forever.
"""

from __future__ import annotations

import json
import logging
import random
import threading
from typing import Any

from confluent_kafka import Producer
from qdrant_client import QdrantClient

from rag_supply_chain.config import settings
from rag_supply_chain.optimization.normalization import measure_normalization, normalize_text
from rag_supply_chain.workers.celery_app import celery_app
from rag_supply_chain.workers.embeddings import DenseEmbedder, sparse_embed
from rag_supply_chain.workers.qdrant_store import UpsertResult, ensure_collection, upsert_chunks

logger = logging.getLogger(__name__)

_dense_embedder: DenseEmbedder | None = None
_qdrant_client: QdrantClient | None = None
_dlq_producer: Producer | None = None
_init_lock = threading.Lock()  # the thread-pool worker shares these globals across threads


def _get_dense_embedder() -> DenseEmbedder:
    global _dense_embedder
    if _dense_embedder is None:
        with _init_lock:
            if _dense_embedder is None:
                _dense_embedder = DenseEmbedder()
    return _dense_embedder


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        with _init_lock:
            if _qdrant_client is None:
                client = QdrantClient(url=settings.qdrant_url)
                ensure_collection(client)
                _qdrant_client = client
    return _qdrant_client


def _get_dlq_producer() -> Producer:
    global _dlq_producer
    if _dlq_producer is None:
        with _init_lock:
            if _dlq_producer is None:
                _dlq_producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
    return _dlq_producer


def compute_backoff(retry_count: int) -> float:
    """Full-jitter exponential backoff: random(0, min(cap, base * 2**retry_count))."""
    ceiling = min(
        settings.embed_backoff_max_seconds,
        settings.embed_backoff_base_seconds * (2**retry_count),
    )
    return random.uniform(0, ceiling)


def process_batch(
    chunk_payloads: list[dict[str, Any]],
    dense_embedder: DenseEmbedder,
    qdrant_client: QdrantClient,
) -> UpsertResult:
    normalized_payloads = [{**chunk, "text": normalize_text(chunk["text"])} for chunk in chunk_payloads]
    texts = [c["text"] for c in normalized_payloads]
    dense_vectors = dense_embedder.embed(texts)
    sparse_vectors = sparse_embed(texts)
    return upsert_chunks(qdrant_client, normalized_payloads, dense_vectors, sparse_vectors)


def _route_to_dlq(chunk_payloads: list[dict[str, Any]], reason: str) -> None:
    producer = _get_dlq_producer()
    for chunk in chunk_payloads:
        out = {**chunk, "dlq_reason": reason}
        producer.produce(
            settings.topic_chunks_dlq,
            key=chunk["chunk_id"].encode("utf-8"),
            value=json.dumps(out).encode("utf-8"),
        )
    producer.flush(10.0)
    logger.error("DLQ: %d chunk(s) — %s", len(chunk_payloads), reason)  # alert per §2.B/§3.B


@celery_app.task(bind=True, max_retries=settings.embed_max_retries)
def embed_and_upsert_batch(self, chunk_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        result = process_batch(chunk_payloads, _get_dense_embedder(), _get_qdrant_client())
    except Exception as exc:  # noqa: BLE001 - any embedding/upsert failure is retryable
        if self.request.retries >= self.max_retries:
            logger.error(
                "batch of %d exhausted %d retries, routing to DLQ: %s",
                len(chunk_payloads),
                self.max_retries,
                exc,
            )
            _route_to_dlq(chunk_payloads, reason=str(exc))
            return {"status": "dlq", "count": len(chunk_payloads), "reason": str(exc)}

        delay = compute_backoff(self.request.retries)
        logger.warning(
            "batch of %d failed (attempt %d): %s — retrying in %.2fs",
            len(chunk_payloads),
            self.request.retries + 1,
            exc,
            delay,
        )
        raise self.retry(exc=exc, countdown=delay)

    if result.dropped:
        dropped_ids = {chunk_id for chunk_id, _ in result.dropped}
        _route_to_dlq(
            [c for c in chunk_payloads if c["chunk_id"] in dropped_ids],
            reason="embedding dimensionality guard",
        )

    return {
        "status": "ok",
        "upserted": len(result.upserted_chunk_ids),
        "dropped": len(result.dropped),
        "normalization": measure_normalization(
            [chunk["text"] for chunk in chunk_payloads],
            token_count=getattr(_get_dense_embedder(), "count_tokens", None)
            or (lambda text: len(text.split())),
        ).to_dict(),
    }
