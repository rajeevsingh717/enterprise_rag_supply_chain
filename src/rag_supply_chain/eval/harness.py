"""Small evaluation harness built on the project's existing retrieval/RAG path.

Retrieval and citation metrics are deterministic. The three answer-quality
metrics are supplied by a judge implementation because they require semantic
assessment; the CLI's live implementation uses the configured Claude judge.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Protocol

from rag_supply_chain.retrieval.generation import AnswerResult
from rag_supply_chain.retrieval.hybrid_search import RetrievedChunk
from rag_supply_chain.telemetry import TokenUsage


class EvaluationError(RuntimeError):
    """A user-actionable evaluation failure."""


@dataclass(frozen=True)
class QACase:
    id: str
    question: str
    reference_answer: str
    expected_source_uri_contains: str


@dataclass(frozen=True)
class QADataset:
    schema_version: int
    dataset_version: str
    cases: list[QACase]


@dataclass(frozen=True)
class JudgeScores:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class CaseResult:
    id: str
    retrieval_hit: float
    reciprocal_rank: float
    source_precision: float
    citation_hit: float
    faithfulness: float | None
    answer_relevance: float | None
    context_precision: float | None
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


class Pipeline(Protocol):
    def retrieve(self, question: str) -> list[RetrievedChunk]: ...

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> AnswerResult: ...


class Judge(Protocol):
    def score(
        self,
        case: QACase,
        chunks: list[RetrievedChunk],
        answer: AnswerResult,
    ) -> JudgeScores: ...


def load_dataset(path: Path) -> QADataset:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Could not load QA dataset {path}: {exc}") from exc
    if raw.get("schema_version") != 1:
        raise EvaluationError(
            f"Unsupported QA schema_version {raw.get('schema_version')!r}; expected 1."
        )
    try:
        cases = [QACase(**item) for item in raw["cases"]]
        dataset = QADataset(
            schema_version=raw["schema_version"],
            dataset_version=raw["dataset_version"],
            cases=cases,
        )
    except (KeyError, TypeError) as exc:
        raise EvaluationError(f"Invalid QA dataset {path}: {exc}") from exc
    if not dataset.cases or len({case.id for case in dataset.cases}) != len(dataset.cases):
        raise EvaluationError("QA dataset must contain cases with unique IDs.")
    return dataset


def evaluate(dataset: QADataset, pipeline: Pipeline, judge: Judge | None = None) -> dict:
    results: list[CaseResult] = []
    for case in dataset.cases:
        chunks = pipeline.retrieve(case.question)
        answer = pipeline.answer(case.question, chunks)
        matches = [case.expected_source_uri_contains in chunk.source_uri for chunk in chunks]
        rank = next((index for index, match in enumerate(matches, 1) if match), None)
        cited_sources = {citation.source_uri for citation in answer.citations}
        judged = judge.score(case, chunks, answer) if judge is not None else None
        usages = [answer.usage, judged.usage if judged else TokenUsage()]
        costs = [usage.cost_usd for usage in usages]
        results.append(
            CaseResult(
                id=case.id,
                retrieval_hit=float(rank is not None),
                reciprocal_rank=1.0 / rank if rank is not None else 0.0,
                source_precision=sum(matches) / len(matches) if matches else 0.0,
                citation_hit=float(
                    any(case.expected_source_uri_contains in uri for uri in cited_sources)
                ),
                faithfulness=judged.faithfulness if judged else None,
                answer_relevance=judged.answer_relevance if judged else None,
                context_precision=judged.context_precision if judged else None,
                input_tokens=sum(usage.input_tokens for usage in usages),
                output_tokens=sum(usage.output_tokens for usage in usages),
                cost_usd=sum(costs) if all(cost is not None for cost in costs) else None,
            )
        )

    metric_names = (
        "retrieval_hit",
        "reciprocal_rank",
        "source_precision",
        "citation_hit",
        "faithfulness",
        "answer_relevance",
        "context_precision",
    )
    aggregates = {}
    for name in metric_names:
        values = [getattr(result, name) for result in results]
        present = [value for value in values if value is not None]
        aggregates[name] = fmean(present) if present else None
    aggregates["input_tokens"] = sum(result.input_tokens for result in results)
    aggregates["output_tokens"] = sum(result.output_tokens for result in results)
    costs = [result.cost_usd for result in results]
    aggregates["cost_usd"] = sum(costs) if all(cost is not None for cost in costs) else None
    return {
        "schema_version": 2,
        "dataset_version": dataset.dataset_version,
        "case_count": len(results),
        "judge_enabled": judge is not None,
        "metrics": aggregates,
        "cases": [asdict(result) for result in results],
    }


def validate_score(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise EvaluationError(f"Judge returned non-numeric {name}.")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise EvaluationError(f"Judge returned {name} outside [0, 1]: {score}.")
    return score
