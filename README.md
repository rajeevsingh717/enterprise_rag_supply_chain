# Enterprise RAG Supply Chain

## Phase 6 evaluation slice

The repository includes a small, versioned QA set at `eval/qa-v1.json` and a
`rag-eval` command that exercises the existing dense+sparse Qdrant retrieval and
Claude answer path. It records deterministic retrieval hit rate, reciprocal rank,
expected-source precision, and citation hit rate. By default, the configured
Claude judge also scores faithfulness, answer relevance, and context precision.

After installing the project, starting Qdrant, and indexing `sample_docs/`:

```shell
rag-eval run --output eval/results-local.json
```

For retrieval/citation metrics without a judge call:

```shell
rag-eval run --skip-judge --min-retrieval-hit-rate 1.0
```

The command exits `2` with a dependency hint if Qdrant, the local embedding
model, the index, or Claude credentials are unavailable. A failed retrieval gate
exits `1`. Results are versioned JSON suitable for reviewing and later checking
in as an approved baseline; no baseline scores are committed yet.

Open decisions: expand and review the QA corpus, approve regression tolerances,
decide how baselines are stored/versioned, and control judge-model/prompt drift.
`source_precision` is a deterministic expected-source proxy, not the semantic
context-precision judge score.
