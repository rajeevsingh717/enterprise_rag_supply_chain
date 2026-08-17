"""Tests for RAG answer generation (§3.D) — a fake Anthropic client stands
in so these run without network access or an API key."""

from __future__ import annotations

from dataclasses import dataclass, field

from rag_supply_chain.retrieval.generation import generate_answer
from rag_supply_chain.retrieval.hybrid_search import RetrievedChunk


@dataclass
class _Citation:
    cited_text: str
    document_index: int


@dataclass
class _TextBlock:
    text: str
    citations: list[_Citation] = field(default_factory=list)
    type: str = "text"


@dataclass
class _Response:
    content: list[_TextBlock]


class _FakeMessages:
    def __init__(self, response: _Response) -> None:
        self._response = response
        self.last_call: dict | None = None

    def create(self, **kwargs):
        self.last_call = kwargs
        return self._response


class FakeAnthropicClient:
    def __init__(self, response: _Response) -> None:
        self.messages = _FakeMessages(response)


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"body of {chunk_id}",
        score=0.8,
        doc_title="System Design Notes",
        source_uri="/tmp/system_design.md",
        hierarchical_context=["Networking"],
    )


def test_generate_answer_with_no_chunks_short_circuits_without_calling_claude() -> None:
    result = generate_answer("anything", [], client=FakeAnthropicClient(_Response(content=[])))
    assert "No relevant context" in result.text


def test_generate_answer_maps_citations_back_to_source_chunk() -> None:
    response = _Response(
        content=[
            _TextBlock(text="Subnets divide a network. "),
            _TextBlock(
                text="See the CIDR example.",
                citations=[_Citation(cited_text="CIDR block", document_index=0)],
            ),
        ]
    )
    chunks = [_chunk("doc:0")]

    result = generate_answer("what is a subnet?", chunks, client=FakeAnthropicClient(response))

    assert result.text == "Subnets divide a network. See the CIDR example."
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "doc:0"
    assert result.citations[0].doc_title == "System Design Notes"
    assert result.citations[0].cited_text == "CIDR block"


def test_generate_answer_builds_one_document_block_per_chunk_with_citations_enabled() -> None:
    fake_client = FakeAnthropicClient(_Response(content=[]))
    chunks = [_chunk("a"), _chunk("b")]

    generate_answer("q", chunks, client=fake_client)

    content = fake_client.messages.last_call["messages"][0]["content"]
    doc_blocks = [b for b in content if b["type"] == "document"]
    assert len(doc_blocks) == 2
    assert all(b["citations"] == {"enabled": True} for b in doc_blocks)
    assert content[-1] == {"type": "text", "text": "q"}
