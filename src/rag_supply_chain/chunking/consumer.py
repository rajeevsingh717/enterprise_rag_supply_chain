"""Kafka consumer for Phase 2 (§3.A).

Consumes `documents.raw` (produced by the ingestion engine), extracts
mandatory metadata and semantically chunks each document, then produces
each chunk to `chunks.embed` for the embedding worker (Phase 3).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from confluent_kafka import Consumer, Producer
from qdrant_client import QdrantClient

from rag_supply_chain.chunking.metadata import extract_metadata
from rag_supply_chain.chunking.semantic_chunker import Chunk, SemanticChunker
from rag_supply_chain.config import settings
from rag_supply_chain.registry.lineage import sync_document_chunks
from rag_supply_chain.registry.store import ensure_schema, get_engine

logger = logging.getLogger(__name__)


def extract_text(path: Path, mime_type: str) -> str:
    if mime_type == "application/pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


class ChunkingConsumer:
    def __init__(self, chunker: SemanticChunker | None = None) -> None:
        self._consumer = Consumer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "group.id": "chunking-consumer",
                "auto.offset.reset": "earliest",
            }
        )
        self._consumer.subscribe([settings.topic_documents_raw])
        self._producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
        self._chunker = chunker or SemanticChunker()
        self._registry_engine = get_engine()
        ensure_schema(self._registry_engine)
        self._qdrant = QdrantClient(url=settings.qdrant_url)

    def process_message(self, payload: dict[str, Any]) -> list[Chunk]:
        path = Path(payload["source_uri"])
        mime_type = payload["mime_type"]
        text = extract_text(path, mime_type)

        metadata = extract_metadata(
            doc_id=payload["doc_id"],
            source_uri=payload["source_uri"],
            timestamp=payload["timestamp"],
            mime_type=mime_type,
            path=path,
            text=text,
        )
        chunks = self._chunker.chunk_document(text, mime_type, parent_doc_id=metadata.doc_id)

        for idx, chunk in enumerate(chunks):
            out = {
                "chunk_id": f"{metadata.doc_id}:{idx}",
                "parent_doc_id": chunk.parent_doc_id,
                "text": chunk.text,
                "hierarchical_context": list(chunk.hierarchical_context),
                "token_count": chunk.token_count,
                "doc_version": metadata.version,
                "doc_title": metadata.title,
                "source_uri": metadata.source_uri,
            }
            self._producer.produce(
                settings.topic_chunks,
                key=out["chunk_id"].encode("utf-8"),
                value=json.dumps(out).encode("utf-8"),
            )
        self._producer.poll(0)
        logger.info("doc %s -> %d chunk(s)", metadata.doc_id, len(chunks))

        chunk_ids = [f"{metadata.doc_id}:{idx}" for idx in range(len(chunks))]
        stale_ids = sync_document_chunks(self._qdrant, self._registry_engine, metadata.doc_id, chunk_ids)
        if stale_ids:
            logger.info(
                "doc %s: purged %d stale chunk(s) from a prior version", metadata.doc_id, len(stale_ids)
            )

        return chunks

    def run(self) -> None:
        logger.info("chunking consumer listening on %s", settings.topic_documents_raw)
        try:
            while True:
                msg = self._consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    logger.error("consumer error: %s", msg.error())
                    continue
                payload = json.loads(msg.value())
                try:
                    self.process_message(payload)
                except Exception:
                    logger.exception("failed to process %s", payload.get("doc_id"))
                self._consumer.commit(msg)
        except KeyboardInterrupt:
            pass
        finally:
            self._producer.flush()
            self._consumer.close()
