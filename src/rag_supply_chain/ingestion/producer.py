"""Kafka/Redpanda producer for the ingestion engine.

Routes each ingested document to `documents.raw` (valid) or `documents.dlq`
(invalid, with a reason) per §2.B. One producer instance is shared across the
watcher process; `flush()` should be called before exit so buffered messages
are actually delivered.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from confluent_kafka import Producer

from rag_supply_chain.config import settings

logger = logging.getLogger(__name__)


def _delivery_callback(err, msg) -> None:
    if err is not None:
        logger.error("delivery failed for %s: %s", msg.key(), err)


class DocumentProducer:
    def __init__(self) -> None:
        self._producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})

    def produce_raw(self, doc_id: str, path: Path, mime_type: str | None) -> None:
        payload = self._payload(doc_id, path, mime_type)
        self._produce(settings.topic_documents_raw, doc_id, payload)

    def produce_dlq(self, doc_id: str, path: Path, mime_type: str | None, reason: str) -> None:
        payload = self._payload(doc_id, path, mime_type)
        payload["reason"] = reason
        logger.error("DLQ: %s (%s) — %s", path, doc_id, reason)  # alert log per §2.B
        self._produce(settings.topic_documents_dlq, doc_id, payload)

    def flush(self, timeout: float = 10.0) -> int:
        return self._producer.flush(timeout)

    @staticmethod
    def _payload(doc_id: str, path: Path, mime_type: str | None) -> dict[str, Any]:
        return {
            "doc_id": doc_id,
            "source_uri": str(path.resolve()),
            "mime_type": mime_type,
            "size_bytes": path.stat().st_size if path.exists() else None,
            "timestamp": time.time(),
        }

    def _produce(self, topic: str, key: str, payload: dict[str, Any]) -> None:
        self._producer.produce(
            topic,
            key=key.encode("utf-8"),
            value=json.dumps(payload).encode("utf-8"),
            callback=_delivery_callback,
        )
        self._producer.poll(0)


def new_doc_id(source_uri: str) -> str:
    """Deterministic per source path (§3.C): re-ingesting the same file must
    resolve to the same doc_id, or the lineage registry has nothing to diff
    against and can never recognize a re-ingest as an update to an existing
    document rather than a brand-new one."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, source_uri))
