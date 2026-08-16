"""Kafka -> Celery bridge (§3.B).

Consumes `chunks.embed` and hands off dynamically-sized batches to the
Celery embedding task (see `batching.DynamicBatcher`), committing offsets
only after a batch has been successfully dispatched.

Handles SIGTERM as well as SIGINT/KeyboardInterrupt: a process manager (or a
plain `kill`, whose default signal is SIGTERM — not SIGINT) stopping this
process mid-batch must not silently drop whatever chunks are sitting in the
buffer waiting for the size/time threshold. Without this, a partial batch
in flight at shutdown is lost — exactly the kind of loss §3.B exists to
prevent.
"""

from __future__ import annotations

import json
import logging
import signal
from typing import Any

from confluent_kafka import Consumer

from rag_supply_chain.config import settings
from rag_supply_chain.workers.batching import DynamicBatcher
from rag_supply_chain.workers.tasks import embed_and_upsert_batch

logger = logging.getLogger(__name__)


class EmbeddingBridge:
    def __init__(self) -> None:
        self._consumer = Consumer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "group.id": "embedding-bridge",
                "auto.offset.reset": "earliest",
            }
        )
        self._consumer.subscribe([settings.topic_chunks])
        self._batcher: DynamicBatcher[dict[str, Any]] = DynamicBatcher(
            max_size=settings.embed_batch_size,
            max_interval_seconds=settings.embed_batch_interval_seconds,
        )
        self._stop = False
        signal.signal(signal.SIGTERM, self._request_stop)
        signal.signal(signal.SIGINT, self._request_stop)

    def _request_stop(self, signum: int, frame: object) -> None:
        self._stop = True

    def run(self) -> None:
        logger.info("embedding bridge listening on %s", settings.topic_chunks)
        try:
            while not self._stop:
                msg = self._consumer.poll(0.2)
                if msg is not None:
                    if msg.error():
                        logger.error("consumer error: %s", msg.error())
                    else:
                        self._batcher.add(json.loads(msg.value()))

                if self._batcher.should_flush():
                    self._flush()
        finally:
            if len(self._batcher):
                self._flush()
            self._consumer.close()

    def _flush(self) -> None:
        batch = self._batcher.flush()
        embed_and_upsert_batch.delay(batch)
        self._consumer.commit(asynchronous=False)
        logger.info("dispatched batch of %d chunk(s)", len(batch))
