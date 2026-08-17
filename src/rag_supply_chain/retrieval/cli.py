"""CLI entrypoint for hybrid retrieval + RAG answer generation.

    rag-query "how does replication handle a primary failure?"

One-shot: embed the question, hybrid-search Qdrant, generate a cited answer.
"""

from __future__ import annotations

import typer
from rich.console import Console

from rag_supply_chain.retrieval.generation import generate_answer
from rag_supply_chain.retrieval.hybrid_search import hybrid_search
from rag_supply_chain.workers.embeddings import DenseEmbedder

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def ask(question: str, top_k: int = typer.Option(None, help="Override retrieval_top_k.")) -> None:
    """Answer a question against the indexed corpus, with citations."""
    from qdrant_client import QdrantClient

    from rag_supply_chain.config import settings

    client = QdrantClient(url=settings.qdrant_url)
    dense_embedder = DenseEmbedder()

    chunks = hybrid_search(client, question, dense_embedder, top_k=top_k)
    if not chunks:
        console.print("[yellow]No relevant chunks found in the index.[/yellow]")
        raise typer.Exit(0)

    result = generate_answer(question, chunks)

    console.print(result.text)
    if result.citations:
        console.print("\n[bold]Sources:[/bold]")
        seen: set[str] = set()
        for citation in result.citations:
            if citation.chunk_id in seen:
                continue
            seen.add(citation.chunk_id)
            console.print(f"  - {citation.doc_title} ({citation.source_uri})")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
