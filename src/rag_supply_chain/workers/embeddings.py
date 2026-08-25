"""Dense + sparse embedding for the async worker (§3.D hybrid search inputs).

Dense: the same local sentence-transformers model used at chunk time
(384-dim, §3.A/§3.D). Injectable so tests don't need torch/network.

Sparse: a hashed term-frequency vector, not full BM25 — real BM25 needs
corpus-wide document-frequency statistics that a per-batch worker doesn't
have. Terms hash into a fixed-size bucket space (collisions are rare enough
at this vocabulary scale to not matter) and get a log-saturated count, which
gives the same "does this exact keyword appear" signal BM25 provides for
hybrid retrieval, without a second model or corpus pass. Documented
deviation, not a silently-dropped feature.
"""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass
from typing import Protocol

from rag_supply_chain.config import settings

SPARSE_VOCAB_SIZE = 2**16
_TOKEN = re.compile(r"[a-z0-9]+")


class DenseModel(Protocol):
    def encode(self, texts: list[str], normalize_embeddings: bool = ...) -> list[list[float]]: ...


@dataclass
class SparseVectorData:
    indices: list[int]
    values: list[float]


class DenseEmbedder:
    def __init__(self, model: DenseModel | None = None) -> None:
        self._model = model or _load_default_model()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(v) for v in vectors]

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))


def sparse_embed(texts: list[str]) -> list[SparseVectorData]:
    return [_sparse_embed_one(t) for t in texts]


def _sparse_embed_one(text: str) -> SparseVectorData:
    counts: dict[int, int] = {}
    for token in _TOKEN.findall(text.lower()):
        bucket = zlib.crc32(token.encode("utf-8")) % SPARSE_VOCAB_SIZE
        counts[bucket] = counts.get(bucket, 0) + 1
    indices = sorted(counts)
    values = [1.0 + math.log(counts[i]) for i in indices]
    return SparseVectorData(indices=indices, values=values)


def _load_default_model() -> DenseModel:
    from sentence_transformers import SentenceTransformer

    # Pinned to CPU: this loads inside a Celery prefork worker (a forked child
    # process), and macOS's MPS/Metal backend is not fork-safe — touching it
    # post-fork crashes the child with SIGABRT. MPS also doesn't parallelize
    # across forked processes the way CPU does, so there's no upside to auto-
    # detecting it here even off of macOS.
    return SentenceTransformer(settings.embedding_model, device="cpu")
