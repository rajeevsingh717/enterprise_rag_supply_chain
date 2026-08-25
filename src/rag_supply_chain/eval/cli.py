"""CLI for the Phase 6 evaluation vertical slice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import anthropic
import typer
from rich.console import Console

from rag_supply_chain.config import settings
from rag_supply_chain.eval.harness import (
    EvaluationError,
    JudgeScores,
    QACase,
    evaluate,
    load_dataset,
    validate_score,
)
from rag_supply_chain.retrieval.generation import AnswerResult, generate_answer
from rag_supply_chain.retrieval.hybrid_search import RetrievedChunk, hybrid_search
from rag_supply_chain.telemetry import usage_from_response
from rag_supply_chain.workers.embeddings import DenseEmbedder

app = typer.Typer(add_completion=False)
console = Console(stderr=True)
DEFAULT_FIXTURE = Path("eval/qa-v1.json")


@app.callback()
def callback() -> None:
    """Run repeatable quality checks against the RAG pipeline."""


class LivePipeline:
    def __init__(self) -> None:
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=settings.qdrant_url)
        self.embedder = DenseEmbedder()

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return hybrid_search(self.client, question, self.embedder)

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> AnswerResult:
        return generate_answer(question, chunks)


class ClaudeJudge:
    def __init__(self) -> None:
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def score(
        self, case: QACase, chunks: list[RetrievedChunk], answer: AnswerResult
    ) -> JudgeScores:
        context = "\n\n".join(chunk.text for chunk in chunks)
        prompt = f"""Score this RAG result from 0 to 1 on each named criterion.
Return only a JSON object with numeric keys faithfulness, answer_relevance,
and context_precision. Faithfulness measures support by context; answer_relevance
measures how directly the answer addresses the question; context_precision
measures how much retrieved context is useful for answering it.

Question: {case.question}
Reference answer: {case.reference_answer}
Retrieved context: {context}
Generated answer: {answer.text}
"""
        response = self.client.messages.create(
            model=settings.judge_model,
            max_tokens=256,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            object_start = text.index("{")
            raw, _ = json.JSONDecoder().raw_decode(text[object_start:])
        except (ValueError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"Judge returned invalid JSON for case {case.id}: {text}") from exc
        return JudgeScores(
            faithfulness=validate_score(raw.get("faithfulness"), "faithfulness"),
            answer_relevance=validate_score(raw.get("answer_relevance"), "answer_relevance"),
            context_precision=validate_score(raw.get("context_precision"), "context_precision"),
            usage=usage_from_response(
                response,
                settings.judge_input_cost_per_million,
                settings.judge_output_cost_per_million,
            ),
        )


@app.command()
def run(
    dataset: Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = DEFAULT_FIXTURE,
    output: Annotated[
        Path | None, typer.Option(help="Write versioned JSON results here.")
    ] = None,
    skip_judge: Annotated[
        bool, typer.Option(help="Run deterministic metrics without Claude judge.")
    ] = False,
    min_retrieval_hit_rate: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Exit non-zero when retrieval hit rate is below this value.",
        ),
    ] = None,
) -> None:
    """Evaluate the indexed corpus using the existing retrieval and answer path."""
    try:
        qa = load_dataset(dataset)
        result = evaluate(qa, LivePipeline(), None if skip_judge else ClaudeJudge())
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        console.print(
            "[red]Evaluation could not run.[/red] "
            f"{exc}\nCheck Qdrant/index availability, local model files, and Claude credentials."
        )
        raise typer.Exit(2) from exc

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        console.print(f"Wrote evaluation results to {output}")
    else:
        typer.echo(rendered)

    actual = result["metrics"]["retrieval_hit"]
    if min_retrieval_hit_rate is not None and actual < min_retrieval_hit_rate:
        console.print(
            f"[red]Regression gate failed:[/red] retrieval_hit={actual:.3f} "
            f"< {min_retrieval_hit_rate:.3f}"
        )
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
