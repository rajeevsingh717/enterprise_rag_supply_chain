"""CLI entrypoint for the chunking engine.

    rag-chunk consume            # long-running: consume documents.raw -> chunks.embed
    rag-chunk process <file>     # one-shot: chunk a single file and print the result

The one-shot form is what exercises the Phase 2 "done when" check directly,
without needing a live Kafka message to trigger it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag_supply_chain.chunking.consumer import extract_text
from rag_supply_chain.chunking.metadata import extract_metadata
from rag_supply_chain.chunking.semantic_chunker import SemanticChunker
from rag_supply_chain.ingestion.validation import SUPPORTED_EXTENSIONS

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def consume() -> None:
    """Consume documents.raw, chunk each doc, produce to chunks.embed."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from rag_supply_chain.chunking.consumer import ChunkingConsumer

    ChunkingConsumer().run()


@app.command()
def process(path: Path) -> None:
    """Chunk a single file and print the resulting chunks (no Kafka needed)."""
    mime_type = SUPPORTED_EXTENSIONS.get(path.suffix.lower())
    if mime_type is None:
        console.print(f"[red]unsupported extension {path.suffix!r}[/red]")
        raise typer.Exit(1)

    text = extract_text(path, mime_type)
    metadata = extract_metadata(
        doc_id="local-test",
        source_uri=str(path),
        timestamp=0.0,
        mime_type=mime_type,
        path=path,
        text=text,
    )
    chunker = SemanticChunker()
    chunks = chunker.chunk_document(text, mime_type, parent_doc_id=metadata.doc_id)

    table = Table(title=f"{path.name} — {len(chunks)} chunk(s) (version={metadata.version})")
    table.add_column("#")
    table.add_column("Tokens")
    table.add_column("Context")
    table.add_column("Text", overflow="fold")
    for i, c in enumerate(chunks):
        crumb = " -> ".join(c.hierarchical_context) or "-"
        preview = c.text[:120] + ("…" if len(c.text) > 120 else "")
        table.add_row(str(i), str(c.token_count), crumb, preview)
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
