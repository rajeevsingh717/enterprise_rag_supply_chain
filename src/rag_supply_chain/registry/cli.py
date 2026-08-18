"""CLI for the chunk lineage registry (§3.C).

    rag-registry tombstone <path>   # wipe every vector + registry row for a doc

There's no automatic file-delete watcher yet (the ingestion watcher only
reacts to new/modified files), so a document delete is triggered explicitly
here rather than inferred from the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def tombstone(path: Path) -> None:
    """Delete every vector and registry row for the document at `path`."""
    from qdrant_client import QdrantClient

    from rag_supply_chain.config import settings
    from rag_supply_chain.ingestion.producer import new_doc_id
    from rag_supply_chain.registry.store import ensure_schema, get_engine, tombstone_document
    from rag_supply_chain.workers.qdrant_store import delete_chunks

    doc_id = new_doc_id(str(path.resolve()))
    engine = get_engine()
    ensure_schema(engine)
    chunk_ids = tombstone_document(engine, doc_id)

    if not chunk_ids:
        console.print(f"[yellow]No registered chunks found for {path} (doc_id={doc_id}).[/yellow]")
        raise typer.Exit(0)

    client = QdrantClient(url=settings.qdrant_url)
    delete_chunks(client, chunk_ids)
    console.print(f"[green]Tombstoned {len(chunk_ids)} chunk(s) for {path} (doc_id={doc_id}).[/green]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
