# Technology Guide — Enterprise RAG Supply Chain

A plain-language tour of every technology in this project, organized by where it sits in the
pipeline. For each: *what it is*, *its job here*, and *why we picked it*. Companion to
[PLAN.md](./PLAN.md) and [PORTFOLIO.md](./PORTFOLIO.md).

## Mental model

A RAG system has two halves: an **offline "supply chain"** that turns raw documents into
searchable vectors, and an **online "query" path** that answers questions. This project's whole
thesis is treating the offline half like a real production data pipeline.

```
OFFLINE (supply chain):  inbox → Redpanda → chunk → embed → Qdrant + Postgres
ONLINE (query):          question → Qdrant → Claude → answer → eval
INFRA underneath it all: Docker + Python
```

---

## The foundation

**Python** — The language everything is written in. Chosen because the entire ML/RAG ecosystem
(embeddings, vector DB clients, the Anthropic SDK) is Python-first.

**Docker & docker-compose** — Docker packages each service (broker, DB, etc.) into an isolated
**container** so it runs identically on any machine. `docker-compose` is the conductor: one
`docker-compose.yml` declares all services, and `docker-compose up` starts the whole stack at
once. This is why the project runs on a laptop with one command instead of installing five
databases by hand.

---

## Ingestion & transport

**Redpanda (speaks the Kafka protocol)** — A **message broker**: a durable pipe that decouples the
thing *producing* data from the thing *consuming* it.
- **Job here:** when a document arrives, ingestion drops an event onto a Kafka **topic** (a named
  queue). A downstream worker reads from it whenever it's ready. If the embedding worker is slow or
  crashes, events wait safely in the topic — nothing is lost. Bad documents route to a **Dead
  Letter Queue (DLQ)** topic instead of blocking the pipeline. This is the portfolio's §3.B "detach
  ingestion from embedding" claim, made real.
- **Why Redpanda:** Kafka-compatible (you learn the real Kafka API) but runs as a single
  lightweight container — no Zookeeper, no JVM tuning.
- **Concepts to learn:** *producer* (writes events), *consumer* (reads them), *topic*, *offset* (a
  consumer's bookmark), *DLQ*.

---

## Turning text into searchable vectors

**sentence-transformers** — A library of **local embedding models**. An *embedding* converts text
into a list of numbers (a vector) that captures its meaning, so similar meanings sit close together
in vector space.
- **Job here:** every chunk gets embedded into a **384-dimensional dense vector**. Runs entirely on
  your machine — no API, no cost.
- **Why local:** embedding 20–30 GB of text through a paid API would be the expensive part; doing
  it locally makes it free.
- **Two flavors it produces:** **dense** vectors (semantic meaning — "how similar in concept") and
  **sparse** vectors (keyword presence, BM25-style — "does the exact term appear"). Both are needed
  for hybrid search.

**Semantic chunking** *(a technique, not a library)* — Instead of blindly cutting documents every N
characters (which severs sentences mid-thought), measure the *cosine similarity* between
consecutive sentences' embeddings and cut only where the topic shifts. This is §3.A — the "senior
edge." No LLM involved; local embeddings + math.

---

## Storage

**Qdrant** — A **vector database**: built to store millions of vectors and answer "find me the 5
closest vectors to this query" in milliseconds.
- **Job here:** holds every chunk's dense + sparse vectors and does the retrieval at query time.
- **Why Qdrant:** it natively supports **hybrid search** — querying dense and sparse indexes
  together and fusing the results with **Reciprocal Rank Fusion (RRF)**, exactly §3.D. Many vector
  DBs make you build that yourself.
- **Concepts:** *dense vs sparse retrieval*, *RRF* (a formula that merges two ranked lists into one
  balanced ranking), *top-K*.

**PostgreSQL + SQLAlchemy** — Postgres is a battle-tested **relational (SQL) database**; SQLAlchemy
is the Python library that lets you talk to it with objects instead of raw SQL.
- **Job here:** the **metadata registry** — the transactional record of
  `document_id → [chunk_ids]`. When a document is updated or deleted, look up its old chunk IDs
  here, purge those vectors from Qdrant, and insert the new ones. This is §3.C **tombstoning** — it
  keeps "ghost vectors" from stale documents out of answers.
- **Why a relational DB for this:** you need *transactions* (all-or-nothing updates) and reliable
  lookups by ID — exactly what SQL is for.

---

## Async processing

**Celery + Redis** — Celery is a **distributed task queue**: it hands units of work ("embed this
batch of chunks") to background worker processes. Redis is the fast in-memory store Celery uses to
track those tasks.
- **Job here:** the embedding worker layer (§3.B). Instead of embedding chunks one-by-one inline,
  tasks queue up and workers drain them, with **dynamic batching** (bundling chunks for efficiency)
  and **exponential backoff with jitter** (on failure, wait 1s, 2s, 4s… plus randomness, so
  retrying workers don't stampede).
- **Why:** this is the pattern that keeps a pipeline alive under bursty load — a core
  "production-grade" signal.

---

## The query path

**Claude (Anthropic API)** — The large language model that does the actual reasoning.
- **Two distinct jobs:** the configured generation model synthesizes a cited answer, and the
  separately configured judge model grades answers in the eval harness.
- **Why Claude:** latest, most capable models; the SDK handles auth, retries, and prompt caching
  (which cuts repeat-context cost ~90%).

**RAG (Retrieval-Augmented Generation)** *(the overall pattern)* — Rather than asking the LLM to
answer from memory (where it hallucinates), first *retrieve* relevant context from your documents
and *feed it in*, so the answer is grounded in your data. Everything above exists to make that
retrieval fast, fresh, and accurate.

---

## Quality control

**Ragas / LLM-as-judge eval** — An automated harness that scores answer quality on **faithfulness**
(did it stick to the context?), **answer relevance** (did it answer the question?), and **context
precision** (was the retrieval good?). Haiku does the scoring.
- **Job here:** §5 quality measurement and an initial retrieval threshold. `make eval` writes a
  current result without overwriting the saved baseline. Multi-metric baseline tolerances remain
  explicit follow-up work rather than an overstated automatic gate.

---

## One-glance summary

| Technology | Layer | Portfolio section |
| :--- | :--- | :--- |
| Docker / compose | Infra | — |
| Redpanda (Kafka) | Ingestion transport | §1, §3.B |
| sentence-transformers | Embeddings (local) | §3.A |
| Qdrant | Vector storage + hybrid search | §3.D |
| Postgres + SQLAlchemy | Metadata registry / tombstoning | §3.C |
| Celery + Redis | Async embedding workers | §3.B |
| Claude (Opus/Haiku) | Generation + eval | §3.D, §5 |
| Ragas-style eval | Quality gate | §5 |
