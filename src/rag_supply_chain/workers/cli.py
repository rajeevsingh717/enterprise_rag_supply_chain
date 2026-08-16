"""CLI entrypoint for the embedding worker's Kafka bridge.

    rag-embed     # long-running: chunks.embed -> Celery task queue

The Celery worker itself is started the normal Celery way:

    celery -A rag_supply_chain.workers.celery_app worker --loglevel=info
"""

from __future__ import annotations

import logging

import typer

app = typer.Typer(add_completion=False)


@app.command()
def bridge() -> None:
    """Consume chunks.embed and dispatch dynamically-batched Celery tasks."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from rag_supply_chain.workers.bridge import EmbeddingBridge

    EmbeddingBridge().run()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
