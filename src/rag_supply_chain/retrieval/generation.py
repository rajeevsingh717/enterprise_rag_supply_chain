"""RAG answer generation (§3.D): assemble retrieved chunks into per-document
content blocks and let Claude's native citations feature ground the answer,
rather than asking the model to cite by hand (which it can fabricate).

One retrieved chunk = one `document` content block with `citations: enabled`.
The response comes back as interleaved text blocks, some carrying a
`citations` array that points at a `document_index` — that index lines up
1:1 with the chunk's position in `chunks`, so citations map straight back to
`chunk_id`/`source_uri` without any string matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from rag_supply_chain.config import settings
from rag_supply_chain.retrieval.hybrid_search import RetrievedChunk
from rag_supply_chain.telemetry import TokenUsage, usage_from_response

SYSTEM_PROMPT = (
    "Answer the user's question using only the provided documents. "
    "If the documents don't contain the answer, say so plainly instead of guessing."
)


@dataclass
class Citation:
    cited_text: str
    chunk_id: str
    doc_title: str
    source_uri: str


@dataclass
class AnswerResult:
    text: str
    citations: list[Citation] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)


def _build_document_blocks(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": chunk.text},
            "title": chunk.doc_title or chunk.chunk_id,
            "citations": {"enabled": True},
        }
        for chunk in chunks
    ]


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    client: anthropic.Anthropic | None = None,
) -> AnswerResult:
    if not chunks:
        return AnswerResult(text="No relevant context was found for this question.")

    client = client or anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.gen_model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [*_build_document_blocks(chunks), {"type": "text", "text": question}],
            }
        ],
    )

    text_parts: list[str] = []
    citations: list[Citation] = []
    for block in response.content:
        if block.type != "text":
            continue
        text_parts.append(block.text)
        for citation in block.citations or []:
            chunk = chunks[citation.document_index]
            citations.append(
                Citation(
                    cited_text=citation.cited_text,
                    chunk_id=chunk.chunk_id,
                    doc_title=chunk.doc_title,
                    source_uri=chunk.source_uri,
                )
            )

    usage = usage_from_response(
        response,
        settings.gen_input_cost_per_million,
        settings.gen_output_cost_per_million,
    )
    return AnswerResult(text="".join(text_parts), citations=citations, usage=usage)
