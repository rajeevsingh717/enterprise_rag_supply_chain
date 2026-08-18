"""Qdrant collection setup + upsert with a dimensionality guard (§2.B / §3.D).

The Vector Index data-quality rule (§2.B) requires verifying embedding
dimensionality before it's allowed anywhere near the index: a vector of the
wrong size is dropped, not written, with an alert logged — never silently
inserted (which would corrupt cosine-distance math for every neighbor).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from rag_supply_chain.config import settings
from rag_supply_chain.workers.embeddings import SparseVectorData

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


@dataclass
class UpsertResult:
    upserted_chunk_ids: list[str] = field(default_factory=list)
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (chunk_id, reason)


def point_id_for(chunk_id: str) -> str:
    """Qdrant point IDs must be an int or UUID; derive a stable UUID from chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection in existing:
        return
    try:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=settings.embedding_dim, distance=Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()},
        )
    except UnexpectedResponse as exc:
        # concurrent workers can race to create the collection on first
        # startup; losing the race is not an error, it just means another
        # worker already did the job.
        if "already exists" not in str(exc):
            raise


def upsert_chunks(
    client: QdrantClient,
    chunk_payloads: list[dict],
    dense_vectors: list[list[float]],
    sparse_vectors: list[SparseVectorData],
) -> UpsertResult:
    result = UpsertResult()
    points: list[PointStruct] = []

    for payload, dense, sparse in zip(chunk_payloads, dense_vectors, sparse_vectors, strict=True):
        chunk_id = payload["chunk_id"]
        if len(dense) != settings.embedding_dim:
            reason = f"dense vector dim {len(dense)} != expected {settings.embedding_dim}"
            logger.error("dropping vector for %s: %s", chunk_id, reason)  # alert per §2.B
            result.dropped.append((chunk_id, reason))
            continue

        points.append(
            PointStruct(
                id=point_id_for(chunk_id),
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: SparseVector(indices=sparse.indices, values=sparse.values),
                },
                payload=payload,
            )
        )
        result.upserted_chunk_ids.append(chunk_id)

    if points:
        client.upsert(collection_name=settings.qdrant_collection, points=points)

    return result


def delete_chunks(client: QdrantClient, chunk_ids: list[str]) -> None:
    """Wipe the vectors for `chunk_ids` (§3.C: stale-version purge and
    document tombstones both land here)."""
    if not chunk_ids:
        return
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=[point_id_for(chunk_id) for chunk_id in chunk_ids],
    )
