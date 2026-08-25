"""One-shot synchronous demo: validate/chunk/index, query, then evaluate.

This deliberately reuses the production component functions while avoiding the
need to supervise three long-running Kafka/Celery processes during a demo.
The asynchronous path remains the normal operational architecture.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from rag_supply_chain.chunking.consumer import extract_text
from rag_supply_chain.chunking.metadata import extract_metadata
from rag_supply_chain.chunking.semantic_chunker import SemanticChunker
from rag_supply_chain.config import settings
from rag_supply_chain.eval.cli import ClaudeJudge
from rag_supply_chain.eval.harness import evaluate, load_dataset
from rag_supply_chain.ingestion.producer import new_doc_id
from rag_supply_chain.ingestion.validation import validate_file
from rag_supply_chain.optimization.normalization import measure_normalization
from rag_supply_chain.registry.lineage import sync_document_chunks
from rag_supply_chain.registry.store import ensure_schema, get_engine
from rag_supply_chain.retrieval.generation import generate_answer
from rag_supply_chain.retrieval.hybrid_search import hybrid_search
from rag_supply_chain.workers.embeddings import DenseEmbedder
from rag_supply_chain.workers.tasks import process_batch

console = Console(stderr=True)


class DemoPipeline:
    def __init__(self, client, embedder: DenseEmbedder) -> None:
        self.client = client
        self.embedder = embedder

    def retrieve(self, question: str):
        return hybrid_search(self.client, question, self.embedder)

    def answer(self, question: str, chunks):
        return generate_answer(question, chunks)


def run(
    document: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "sample_docs/system_design.md"
    ),
    dataset: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "eval/qa-v1.json"
    ),
    output: Annotated[Path | None, typer.Option(help="Optional JSON output path.")] = None,
) -> None:
    """Run the real synchronous indexing, query, and evaluation vertical slice."""
    try:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        validation = validate_file(document)
        if not validation.ok or validation.mime_type is None:
            raise RuntimeError(f"document validation failed: {validation.reason}")

        model = SentenceTransformer(settings.embedding_model, device="cpu")
        embedder = DenseEmbedder(model=model)
        text = extract_text(document, validation.mime_type)
        doc_id = new_doc_id(str(document.resolve()))
        metadata = extract_metadata(
            doc_id,
            str(document),
            document.stat().st_mtime,
            validation.mime_type,
            document,
            text,
        )
        chunks = SemanticChunker(model=model).chunk_document(text, validation.mime_type, doc_id)
        payloads = [
            {
                "chunk_id": f"{doc_id}:{index}",
                "parent_doc_id": doc_id,
                "text": chunk.text,
                "hierarchical_context": list(chunk.hierarchical_context),
                "token_count": chunk.token_count,
                "doc_version": metadata.version,
                "doc_title": metadata.title,
                "source_uri": metadata.source_uri,
            }
            for index, chunk in enumerate(chunks)
        ]
        normalization = measure_normalization(
            [payload["text"] for payload in payloads], token_count=embedder.count_tokens
        )
        client = QdrantClient(url=settings.qdrant_url)
        engine = get_engine()
        ensure_schema(engine)
        sync_document_chunks(client, engine, doc_id, [payload["chunk_id"] for payload in payloads])
        indexed = process_batch(payloads, embedder, client)

        pipeline = DemoPipeline(client, embedder)
        qa = load_dataset(dataset)
        query_question = qa.cases[0].question
        query_answer = pipeline.answer(query_question, pipeline.retrieve(query_question))
        evaluation = evaluate(qa, pipeline, ClaudeJudge())
        report = {
            "schema_version": 1,
            "document": str(document),
            "indexing": {
                "chunks": len(chunks),
                "upserted": len(indexed.upserted_chunk_ids),
                "dropped": len(indexed.dropped),
                "normalization": normalization.to_dict(),
            },
            "query": {
                "question": query_question,
                "answer": query_answer.text,
                "citations": len(query_answer.citations),
                "usage": asdict(query_answer.usage),
            },
            "evaluation": evaluation,
        }
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
            console.print(f"Wrote demo results to {output}")
        else:
            typer.echo(rendered)
    except Exception as exc:
        console.print(
            f"[red]Demo could not run.[/red] {exc}\n"
            "Check infrastructure, the local embedding model, and Claude credentials."
        )
        raise typer.Exit(2) from exc


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
