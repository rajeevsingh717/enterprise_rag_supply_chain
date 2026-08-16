"""Tests for dense/sparse embedding (§3.B/§3.D) — no torch/network needed:
DenseEmbedder takes an injected fake model."""

from __future__ import annotations

from rag_supply_chain.workers.embeddings import DenseEmbedder, sparse_embed


class FakeDenseModel:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
        return [[float(len(t))] * self.dim for t in texts]


def test_dense_embedder_uses_injected_model() -> None:
    embedder = DenseEmbedder(model=FakeDenseModel(dim=4))
    vectors = embedder.embed(["hi", "hello"])
    assert len(vectors) == 2
    assert all(len(v) == 4 for v in vectors)


def test_sparse_embed_is_deterministic() -> None:
    a = sparse_embed(["the quick brown fox"])[0]
    b = sparse_embed(["the quick brown fox"])[0]
    assert a.indices == b.indices
    assert a.values == b.values


def test_sparse_embed_reflects_repeated_terms() -> None:
    once = sparse_embed(["fox"])[0]
    twice = sparse_embed(["fox fox"])[0]
    assert once.indices == twice.indices  # same single bucket
    assert twice.values[0] > once.values[0]  # log-saturated count is higher


def test_sparse_embed_empty_text_has_no_terms() -> None:
    result = sparse_embed([""])[0]
    assert result.indices == []
    assert result.values == []


def test_sparse_embed_different_texts_usually_differ() -> None:
    a = sparse_embed(["networking subnets routing"])[0]
    b = sparse_embed(["database indexing replication"])[0]
    assert set(a.indices) != set(b.indices)
