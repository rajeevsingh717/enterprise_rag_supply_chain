"""Semantic chunker (§3.A): merges sentences into chunks along topic
boundaries rather than fixed character/token windows.

Boundary signal: cosine similarity between a sentence's embedding and the
running mean embedding of the chunk being built. A sharp drop signals a
topic shift. A section-header change (different `breadcrumb`) is always
also treated as a topic shift, since a new heading is an explicit one.
Token bounds (128 <= N <= 512, §2.B) are enforced on top of the semantic
signal: a chunk below the minimum stays open regardless of similarity, and
one that would exceed the maximum is force-split at the current sentence
boundary regardless of similarity.

The embedding model is injected (`model`), not hardcoded, so tests can swap
in a cheap deterministic stand-in instead of downloading/running the real
sentence-transformers model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rag_supply_chain.chunking.headers import Segment, segment_html, segment_markdown
from rag_supply_chain.chunking.metadata import strip_frontmatter
from rag_supply_chain.chunking.sentences import split_sentences
from rag_supply_chain.config import settings

DEFAULT_SIMILARITY_THRESHOLD = 0.55


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...


class EmbeddingModel(Protocol):
    tokenizer: Tokenizer

    def encode(self, texts: list[str], normalize_embeddings: bool = ...) -> np.ndarray: ...


@dataclass
class _Sentence:
    text: str
    breadcrumb: tuple[str, ...]


@dataclass
class Chunk:
    text: str
    parent_doc_id: str
    hierarchical_context: tuple[str, ...]
    token_count: int


class SemanticChunker:
    """Loads the embedding model once; reuse one instance across a process."""

    def __init__(
        self,
        model: EmbeddingModel | None = None,
        min_tokens: int | None = None,
        max_tokens: int | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._model = model or _load_default_model()
        self.min_tokens = min_tokens if min_tokens is not None else settings.chunk_min_tokens
        self.max_tokens = max_tokens if max_tokens is not None else settings.chunk_max_tokens
        self.similarity_threshold = similarity_threshold

    def chunk_document(self, text: str, mime_type: str, parent_doc_id: str) -> list[Chunk]:
        segments = self._segment(text, mime_type)
        sentences = self._to_sentences(segments)
        if not sentences:
            return []

        embeddings = self._model.encode([s.text for s in sentences], normalize_embeddings=True)
        return self._merge(sentences, embeddings, parent_doc_id)

    def _segment(self, text: str, mime_type: str) -> list[Segment]:
        if mime_type == "text/markdown":
            return segment_markdown(strip_frontmatter(text))
        if mime_type == "text/html":
            return segment_html(text)
        return [Segment(text, ())]  # PDF / plain text: no header structure to key off

    @staticmethod
    def _to_sentences(segments: list[Segment]) -> list[_Sentence]:
        return [
            _Sentence(sentence, seg.breadcrumb)
            for seg in segments
            for sentence in split_sentences(seg.text)
        ]

    def _token_count(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))

    def _merge(
        self, sentences: list[_Sentence], embeddings: np.ndarray, parent_doc_id: str
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        current: list[int] = [0]
        running_mean = embeddings[0].copy()
        current_tokens = self._token_count(sentences[0].text)

        for i in range(1, len(sentences)):
            sentence_tokens = self._token_count(sentences[i].text)
            candidate_tokens = current_tokens + sentence_tokens

            breadcrumb_changed = sentences[i].breadcrumb != sentences[current[0]].breadcrumb
            similarity = float(np.dot(running_mean, embeddings[i]))
            topic_shift = breadcrumb_changed or similarity < self.similarity_threshold

            would_overflow = candidate_tokens > self.max_tokens
            natural_boundary = topic_shift and current_tokens >= self.min_tokens

            if would_overflow or natural_boundary:
                chunks.append(self._build_chunk(sentences, current, parent_doc_id))
                current = [i]
                running_mean = embeddings[i].copy()
                current_tokens = sentence_tokens
            else:
                current.append(i)
                running_mean += (embeddings[i] - running_mean) / len(current)
                current_tokens = candidate_tokens

        chunks.append(self._build_chunk(sentences, current, parent_doc_id))
        return chunks

    def _build_chunk(
        self, sentences: list[_Sentence], indices: list[int], parent_doc_id: str
    ) -> Chunk:
        text = " ".join(sentences[j].text for j in indices)
        breadcrumb = sentences[indices[0]].breadcrumb
        prefixed = self._with_breadcrumb(text, breadcrumb)
        return Chunk(
            text=prefixed,
            parent_doc_id=parent_doc_id,
            hierarchical_context=breadcrumb,
            token_count=self._token_count(prefixed),
        )

    @staticmethod
    def _with_breadcrumb(text: str, breadcrumb: tuple[str, ...]) -> str:
        crumbs = [c for c in breadcrumb if c]
        if not crumbs:
            return text
        return f"[{' -> '.join(crumbs)}] {text}"


def _load_default_model() -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)
