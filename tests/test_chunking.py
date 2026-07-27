"""Tests for metadata extraction + semantic chunking (Phase 2 "done when" check).

Uses a deterministic word-overlap `FakeModel` instead of the real
sentence-transformers model, so these run fast and offline under plain
`pytest` — no torch, no model download. The real model is exercised
manually via `rag-chunk process`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pypdf import PdfWriter

from rag_supply_chain.chunking.headers import segment_html, segment_markdown
from rag_supply_chain.chunking.metadata import extract_metadata
from rag_supply_chain.chunking.semantic_chunker import SemanticChunker


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return text.split()


class FakeModel:
    """Cosine similarity by word overlap — shared vocabulary -> high similarity."""

    tokenizer = _FakeTokenizer()

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vocab: dict[str, int] = {}
        for t in texts:
            for w in t.lower().split():
                vocab.setdefault(w, len(vocab))
        dim = max(len(vocab), 1)
        dense = np.zeros((len(texts), dim))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                dense[i, vocab[w]] += 1
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return dense / norms


class TopicVectorModel:
    """Near-one-hot embedding per sentence, keyed by a caller-supplied topic
    label, plus tiny noise so no two sentences are identical.

    Bag-of-words similarity between short, lexically-diverse technical
    sentences is noisy even *within* one topic (individual sentences often
    share only stopwords), so it can't reliably demonstrate the merge
    algorithm's boundary logic on its own. This model instead encodes a
    ground-truth topic signal directly, isolating "does the algorithm react
    correctly to a real cohesion drop" from "is this particular toy
    embedding a good similarity proxy."
    """

    tokenizer = _FakeTokenizer()

    def __init__(self, topic_of: dict[str, int], num_topics: int, seed: int = 0) -> None:
        self._topic_of = topic_of
        self._num_topics = num_topics
        self._rng = np.random.default_rng(seed)

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        dense = np.zeros((len(texts), self._num_topics))
        for i, t in enumerate(texts):
            vec = self._rng.normal(0, 0.05, size=self._num_topics)
            vec[self._topic_of[t]] += 1.0
            dense[i] = vec
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        return dense / norms


# --- headers -----------------------------------------------------------------


def test_segment_markdown_builds_breadcrumbs() -> None:
    text = """# Doc Title

intro text

## Section A

content a

### Subsection A1

content a1
"""
    segments = segment_markdown(text)
    breadcrumbs = [s.breadcrumb for s in segments]
    assert breadcrumbs == [
        ("Doc Title",),
        ("Doc Title", "Section A"),
        ("Doc Title", "Section A", "Subsection A1"),
    ]


def test_segment_markdown_handles_header_level_jump() -> None:
    text = """# Title

### Deep Section

body
"""
    segments = segment_markdown(text)
    assert segments[-1].breadcrumb == ("Title", "", "Deep Section")


def test_segment_html_builds_breadcrumbs() -> None:
    html = "<h1>Doc</h1><p>intro</p><h2>Sec</h2><p>body</p>"
    segments = segment_html(html)
    assert [s.breadcrumb for s in segments] == [("Doc",), ("Doc", "Sec")]


# --- metadata ------------------------------------------------------------------


def test_metadata_from_markdown_frontmatter(tmp_path: Path) -> None:
    text = '---\nversion: "2.3"\ntitle: My Doc\n---\n\nbody text\n'
    result = extract_metadata(
        doc_id="d1",
        source_uri="x.md",
        timestamp=0.0,
        mime_type="text/markdown",
        path=tmp_path / "x.md",
        text=text,
    )
    assert result.version == "2.3"
    assert result.title == "My Doc"


def test_metadata_regex_fallback_without_frontmatter(tmp_path: Path) -> None:
    text = "Version: 4.1\n\nSome intro paragraph.\n"
    result = extract_metadata(
        doc_id="d1",
        source_uri="x.md",
        timestamp=0.0,
        mime_type="text/markdown",
        path=tmp_path / "x.md",
        text=text,
    )
    assert result.version == "4.1"


def test_metadata_defaults_when_nothing_found(tmp_path: Path) -> None:
    text = "Just a plain paragraph with no metadata markers.\n"
    result = extract_metadata(
        doc_id="d1",
        source_uri="x.md",
        timestamp=0.0,
        mime_type="text/markdown",
        path=tmp_path / "x.md",
        text=text,
    )
    assert result.version == "1"
    assert result.title is None


def test_metadata_from_html_meta_tags(tmp_path: Path) -> None:
    text = '<html><head><title>HTML Doc</title><meta name="version" content="9"></head></html>'
    result = extract_metadata(
        doc_id="d1",
        source_uri="x.html",
        timestamp=0.0,
        mime_type="text/html",
        path=tmp_path / "x.html",
        text=text,
    )
    assert result.version == "9"
    assert result.title == "HTML Doc"


def test_metadata_from_pdf_info(tmp_path: Path) -> None:
    pdf_path = tmp_path / "x.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Title": "PDF Doc"})
    with pdf_path.open("wb") as f:
        writer.write(f)

    result = extract_metadata(
        doc_id="d1",
        source_uri="x.pdf",
        timestamp=0.0,
        mime_type="application/pdf",
        path=pdf_path,
        text="",
    )
    assert result.title == "PDF Doc"
    assert result.version == "1"  # pypdf has no standard version field


# --- semantic chunker ------------------------------------------------------------


NETWORKING_SUBNETS = (
    "Subnets divide a larger network into smaller routable segments. "
    "Each subnet uses a CIDR block to define its address range. "
    "Routers forward packets between subnets based on the routing table. "
    "Network administrators size subnets according to expected host count. "
    "A well-planned subnet layout reduces broadcast traffic significantly. "
    "Overlapping subnet ranges cause routing conflicts across the network. "
    "Cloud providers let you define custom subnet ranges per availability zone. "
    "Subnet masks determine which bits identify the network versus the host."
)

DATABASE_INDEXING = (
    "Database indexes speed up query lookups by avoiding full table scans. "
    "A B-tree index keeps keys sorted for efficient range queries. "
    "The query planner chooses an index based on estimated selectivity. "
    "Composite indexes cover queries that filter on multiple columns. "
    "Index maintenance adds overhead to every insert and update statement. "
    "Covering indexes let the database answer a query without touching the table. "
    "Poorly chosen indexes bloat storage without improving query performance. "
    "Database administrators monitor index usage to prune unused ones."
)


def _multi_section_markdown() -> str:
    return f"""# System Design

## Networking

### Subnets

{NETWORKING_SUBNETS}

## Databases

### Indexing

{DATABASE_INDEXING}
"""


def test_multi_section_doc_yields_boundary_clean_chunks() -> None:
    # similarity_threshold=0: isolates header-driven boundaries from the
    # semantic-similarity signal (covered separately below), so this test
    # verifies section structure -> chunk structure cleanly.
    chunker = SemanticChunker(
        model=FakeModel(), min_tokens=20, max_tokens=200, similarity_threshold=0.0
    )
    chunks = chunker.chunk_document(
        _multi_section_markdown(), mime_type="text/markdown", parent_doc_id="doc-1"
    )

    assert len(chunks) == 2

    subnets_chunk, indexing_chunk = chunks
    assert subnets_chunk.hierarchical_context == ("System Design", "Networking", "Subnets")
    assert indexing_chunk.hierarchical_context == ("System Design", "Databases", "Indexing")

    # hierarchical_context is injected into the chunk text itself (§3.A)
    assert subnets_chunk.text.startswith("[System Design -> Networking -> Subnets]")
    assert "Subnets divide" in subnets_chunk.text
    assert "Database indexes" not in subnets_chunk.text  # no bleed across the boundary

    assert indexing_chunk.text.startswith("[System Design -> Databases -> Indexing]")
    assert "Database indexes" in indexing_chunk.text
    assert "Subnets divide" not in indexing_chunk.text

    for chunk in chunks:
        assert chunk.token_count >= 20
        assert chunk.parent_doc_id == "doc-1"


def test_chunker_force_splits_when_exceeding_max_tokens() -> None:
    # one section, long enough that even with high semantic cohesion (same
    # topic throughout) the token cap must force a split.
    long_text = "# Topic\n\n" + (NETWORKING_SUBNETS + " ") * 4
    chunker = SemanticChunker(model=FakeModel(), min_tokens=10, max_tokens=60)

    chunks = chunker.chunk_document(long_text, mime_type="text/markdown", parent_doc_id="doc-2")

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 60 + 10  # small tolerance for breadcrumb prefix tokens


def test_semantic_similarity_splits_within_same_section() -> None:
    # both paragraphs sit under the same header (no breadcrumb change), so
    # any split here must come from the cosine-similarity cohesion signal.
    from rag_supply_chain.chunking.sentences import split_sentences

    networking_sentences = split_sentences(NETWORKING_SUBNETS)
    database_sentences = split_sentences(DATABASE_INDEXING)
    topic_of = {s: 0 for s in networking_sentences} | {s: 1 for s in database_sentences}
    model = TopicVectorModel(topic_of, num_topics=2)

    text = f"# Notes\n\n{NETWORKING_SUBNETS}\n\n{DATABASE_INDEXING}\n"
    chunker = SemanticChunker(model=model, min_tokens=20, max_tokens=500)

    chunks = chunker.chunk_document(text, mime_type="text/markdown", parent_doc_id="doc-4")

    assert len(chunks) == 2
    assert all(c.hierarchical_context == ("Notes",) for c in chunks)
    assert "Subnets divide" in chunks[0].text
    assert "Database indexes" in chunks[1].text


def test_chunker_strips_frontmatter_before_chunking() -> None:
    text = f'---\nversion: "1.0"\ntitle: Doc\n---\n\n# Networking\n\n{NETWORKING_SUBNETS}\n'
    chunker = SemanticChunker(model=FakeModel(), min_tokens=5, max_tokens=200)

    chunks = chunker.chunk_document(text, mime_type="text/markdown", parent_doc_id="doc-5")

    assert all("---" not in c.text for c in chunks)
    assert all("version:" not in c.text for c in chunks)


def test_chunker_keeps_short_doc_below_min_as_single_chunk() -> None:
    chunker = SemanticChunker(model=FakeModel(), min_tokens=200, max_tokens=500)
    chunks = chunker.chunk_document(
        "# Title\n\nJust one short sentence.", mime_type="text/markdown", parent_doc_id="doc-3"
    )
    assert len(chunks) == 1
