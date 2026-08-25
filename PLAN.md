# Build Plan — Enterprise RAG Supply Chain

Companion to [PORTFOLIO.md](./PORTFOLIO.md). This is the buildable, runnable version of that
architecture blueprint. Decisions locked with the user:

- **Scope:** runnable vertical slice — every component real but pragmatic, runs end-to-end on a laptop via `docker-compose`.
- **Provider:** configurable Claude generation/judge models; local `sentence-transformers` for embeddings (no embedding API cost).
- **Goal:** learn the stack hands-on — use *real* infra (broker, vector DB, async workers), not fakes.

---

## Tech Stack

| Concern | Choice | Why |
| :--- | :--- | :--- |
| Message broker | **Redpanda** (Kafka API) | Kafka-wire-compatible, single container (no Zookeeper). Teaches the real Kafka client. |
| Vector DB | **Qdrant** | Native hybrid: dense + sparse(BM25-style) + built-in RRF fusion. Matches §3.D exactly. |
| Async workers | **Celery + Redis** | The canonical distributed task queue the doc (§3.B) describes; dynamic batching + retry live here. |
| Embeddings | **sentence-transformers** `BAAI/bge-small-en-v1.5` (384-dim, local) | Free, runnable offline, has a matching sparse model for hybrid. |
| Generation + eval judge | **Claude** (configured by `GEN_MODEL` / `JUDGE_MODEL`) | Keeps model selection explicit and replaceable as provider offerings change. |
| Metadata registry | **Postgres** + SQLAlchemy | Transactional `doc_id ↔ chunk_ids` mapping for tombstoning (§3.C). |
| Orchestration | **docker-compose** | One `up` brings the whole supply chain online. |
| Language | **Python 3.11+** | sentence-transformers, anthropic SDK, kafka/qdrant clients. |

### Deviations from PORTFOLIO.md (flagged honestly)
- **Embedding dim = 384, not 1536.** Local model. The §2.B "exactly 1536" rule becomes "exactly 384 for bge-small". Documented, not silently dropped.
- **Ingestion is a directory/file watcher, not true CDC/Debezium.** Same event-driven shape (change → event → topic), pragmatic source.
- **Hybrid ratio** (§4.3 "60/40 dense/sparse") is not asserted: the current Qdrant server-side RRF is unweighted; comparative tuning remains follow-up work.

---

## Phased Build

### Phase 0 — Scaffold & Infra
- `docker-compose.yml`: redpanda, qdrant, postgres, redis.
- Project skeleton: `src/`, `pyproject.toml`, `.env.example`, config module, `Makefile`.
- Health-check script that confirms all four services are reachable.
- **Done when:** `docker-compose up` + `make health` is green.

### Phase 1 — Ingestion Engine (§2.B Ingestion)
- Watch an `inbox/` dir; on new file: MIME/corruption validation (PDF, MD, HTML).
- Valid → produce to Kafka `documents.raw`. Invalid → `documents.dlq` + alert log.
- **Done when:** dropping a good + a corrupt PDF routes each to the right topic.

### Phase 2 — Metadata Extraction + Semantic Chunker (§3.A)
- Consumer: extract mandatory metadata (doc_id, version, source_uri, timestamp); regex fallback.
- Semantic chunker: sliding-window cosine similarity across sentence embeddings; boundary on cohesion drop; enforce 128 ≤ tokens ≤ 512.
- Enrich each chunk: `parent_doc_id`, `hierarchical_context` (section-title breadcrumb injected into text).
- **Done when:** a multi-section doc yields boundary-clean chunks with metadata + context.

### Phase 3 — Async Embedding Worker (§3.B)
- Celery worker consumes chunks; dynamic micro-batching; embed with sentence-transformers (dense + sparse).
- Exponential backoff **with jitter** on transient failures; DLQ on repeated failure.
- Upsert dense + sparse vectors to Qdrant; dimensionality guard (== 384).
- **Done when:** a burst of chunks drains through the queue into Qdrant without loss.

### Phase 4 — Lineage & Tombstone Lifecycle (§3.C)
- Postgres registry: `doc_id → [chunk_ids]`, versioned.
- On doc update: transactional purge of old chunk vectors + upsert of new. On delete: tombstone event → wipe vectors.
- **Done when:** re-ingesting a modified doc leaves zero stale vectors (asserted by count).

### Phase 5 — Hybrid Retrieval + RAG Answer (§3.D)
- Query Qdrant dense + sparse; fuse with RRF (tunable weight).
- Assemble context → `claude-opus-4-8` with citations → answer.
- Simple CLI/HTTP query entrypoint.
- **Done when:** an alphanumeric-spec query that pure-vector misses is retrieved via hybrid.

### Phase 6 — Eval Harness / LLMOps Guardrail (§5)
- Claude-judge (Ragas-style or custom) scoring: faithfulness, answer relevance, context precision.
- A fixed QA set + baseline scores; a `make eval` gate that fails on regression.
- **Done when:** changing a chunking param and running `make eval` shows a score delta and passes/fails the gate.
- **Status:** vertical slice implemented with a versioned QA set, saved live result,
  judge metrics, deterministic retrieval metrics, usage telemetry, and a retrieval
  threshold. Multi-metric baseline comparison/tolerances remain follow-up work.

### Phase 7 — Cost & Pruning + Polish (§2.C, §4)
- Token/cost tracking per pipeline run.
- Boilerplate/whitespace normalization pre-embedding; measure token reduction (target ≥15%).
- README with the architecture diagram, a `make demo` one-shot, and the operational-learnings metrics reproduced from real runs.
- **Done when:** `make demo` runs ingest → query → eval and prints the headline metrics.
- **Status:** implemented as a synchronous one-shot demo reusing production
  component functions. Token counts come from provider/model telemetry; cost is
  estimated only when current rates are configured. The ≥15% reduction remains
  a measured target, not a fabricated claim.

---

## Proposed Layout
```
enterprise_rag_supply_chain/
├── docker-compose.yml
├── pyproject.toml
├── Makefile
├── .env.example
├── inbox/                      # drop docs here (watched)
├── sample_docs/               # seed corpus for the demo
└── src/rag_supply_chain/
    ├── config.py
    ├── ingestion/             # watcher, MIME validation, producer, DLQ
    ├── chunking/              # semantic chunker, metadata enrichment
    ├── workers/              # celery app, embedding tasks, backoff
    ├── registry/             # postgres registry, tombstone/purge
    ├── retrieval/            # qdrant hybrid query, RRF, claude answer
    ├── eval/                 # judge harness, baseline gate
    └── telemetry/            # token/cost tracking
```

## Decisions implemented
1. Query entrypoint is CLI-first.
2. Generation and judge models are configured separately.
3. A small generated multi-section technical corpus is checked in for the demo.
