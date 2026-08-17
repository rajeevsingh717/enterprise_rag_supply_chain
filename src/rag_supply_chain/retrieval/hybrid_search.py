"""Hybrid retrieval (§3.D): dense + sparse search fused server-side with RRF.

Dense catches semantic matches; sparse catches exact keyword/alphanumeric-spec
matches dense embeddings blur (see the sparse-embedding rationale in
`workers/embeddings.py`). Neither alone is enough — Qdrant's Query API runs
both as prefetch passes over the collection's two named vectors and fuses the
ranked lists with Reciprocal Rank Fusion, so a chunk that ranks well on either
signal surfaces without hand-written score blending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from rag_supply_chain.config import settings
from rag_supply_chain.workers.embeddings import sparse_embed
from rag_supply_chain.workers.qdrant_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


class DenseEmbedderLike(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    doc_title: str
    source_uri: str
    hierarchical_context: list[str]


def hybrid_search(
    client: QdrantClient,
    query: str,
    dense_embedder: DenseEmbedderLike,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k if top_k is not None else settings.retrieval_top_k
    dense_vector = dense_embedder.embed([query])[0]
    sparse_vector = sparse_embed([query])[0]

    response = client.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=settings.retrieval_prefetch_limit,
            ),
            Prefetch(
                query=SparseVector(indices=sparse_vector.indices, values=sparse_vector.values),
                using=SPARSE_VECTOR_NAME,
                limit=settings.retrieval_prefetch_limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    return [
        RetrievedChunk(
            chunk_id=point.payload["chunk_id"],
            text=point.payload["text"],
            score=point.score,
            doc_title=point.payload.get("doc_title", ""),
            source_uri=point.payload.get("source_uri", ""),
            hierarchical_context=point.payload.get("hierarchical_context", []),
        )
        for point in response.points
    ]
