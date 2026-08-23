from __future__ import annotations

import json

import pytest

from rag_supply_chain.eval.cli import ClaudeJudge
from rag_supply_chain.eval.harness import (
    EvaluationError,
    JudgeScores,
    QACase,
    QADataset,
    evaluate,
    load_dataset,
)
from rag_supply_chain.retrieval.generation import AnswerResult, Citation
from rag_supply_chain.retrieval.hybrid_search import RetrievedChunk


def _chunk(source: str, chunk_id: str = "chunk-1") -> RetrievedChunk:
    return RetrievedChunk(chunk_id, "context", 0.9, "Doc", source, [])


class FakePipeline:
    def retrieve(self, question: str) -> list[RetrievedChunk]:
        return [_chunk("unrelated.md", "noise"), _chunk("sample/system_design.md")]

    def answer(self, question: str, chunks: list[RetrievedChunk]) -> AnswerResult:
        return AnswerResult(
            "grounded answer",
            [Citation("support", chunks[1].chunk_id, "Doc", chunks[1].source_uri)],
        )


class FakeJudge:
    def score(self, case, chunks, answer) -> JudgeScores:
        return JudgeScores(0.9, 0.8, 0.7)


def _dataset() -> QADataset:
    return QADataset(1, "test-v1", [QACase("q1", "question", "reference", "system_design")])


def test_evaluate_captures_deterministic_and_judge_metrics() -> None:
    result = evaluate(_dataset(), FakePipeline(), FakeJudge())

    assert result["dataset_version"] == "test-v1"
    assert result["metrics"]["retrieval_hit"] == 1.0
    assert result["metrics"]["reciprocal_rank"] == 0.5
    assert result["metrics"]["source_precision"] == 0.5
    assert result["metrics"]["citation_hit"] == 1.0
    assert result["metrics"]["faithfulness"] == 0.9


def test_evaluate_without_judge_marks_semantic_metrics_unavailable() -> None:
    result = evaluate(_dataset(), FakePipeline())
    assert result["judge_enabled"] is False
    assert result["metrics"]["faithfulness"] is None


def test_load_dataset_rejects_unknown_schema(tmp_path) -> None:
    path = tmp_path / "qa.json"
    path.write_text(json.dumps({"schema_version": 2, "cases": []}))
    with pytest.raises(EvaluationError, match="schema_version"):
        load_dataset(path)


def test_claude_judge_accepts_json_wrapped_in_markdown_and_rationale() -> None:
    class TextBlock:
        type = "text"
        text = (
            '```json\n{"faithfulness": 0.9, "answer_relevance": 0.8, '
            '"context_precision": 0.7}\n```\nRationale follows.'
        )

    class Messages:
        def create(self, **kwargs):
            return type("Response", (), {"content": [TextBlock()]})()

    judge = ClaudeJudge.__new__(ClaudeJudge)
    judge.client = type("Client", (), {"messages": Messages()})()
    scores = judge.score(_dataset().cases[0], [_chunk("system_design.md")], AnswerResult("a"))

    assert scores == JudgeScores(0.9, 0.8, 0.7)
