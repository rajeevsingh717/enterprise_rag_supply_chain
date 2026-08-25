"""Measure normalization without indexing or calling external model APIs."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from rag_supply_chain.optimization.normalization import measure_normalization

app = typer.Typer(add_completion=False)


@app.command()
def measure(path: Path) -> None:
    """Print deterministic before/after token estimates for a UTF-8 text file."""
    text = path.read_text(encoding="utf-8")
    result = measure_normalization([text])
    typer.echo(json.dumps(result.to_dict(), indent=2, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
