"""CLI entrypoint for the ingestion engine.

    rag-ingest watch              # long-running: watch settings.inbox_dir
    rag-ingest process <file>     # one-shot: validate + route a single file

The one-shot form is what exercises the Phase 1 "done when" check directly,
without waiting on filesystem events.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from rag_supply_chain.config import settings
from rag_supply_chain.ingestion.producer import DocumentProducer
from rag_supply_chain.ingestion.watcher import process_file
from rag_supply_chain.ingestion.watcher import watch as watch_inbox

app = typer.Typer(add_completion=False)


@app.command()
def watch(
    inbox: str = typer.Option(settings.inbox_dir, "--inbox", help="Directory to watch."),
) -> None:
    """Watch the inbox directory and route documents to Kafka."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    watch_inbox(inbox)


@app.command()
def process(path: Path) -> None:
    """Validate and route a single file, then exit."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    producer = DocumentProducer()
    process_file(path, producer)
    producer.flush()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
