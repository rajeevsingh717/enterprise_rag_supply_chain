"""Filesystem watcher for the ingestion engine.

Watches `settings.inbox_dir`; each new, fully-written file is validated and
routed to `documents.raw` or `documents.dlq` (§2.B). "Fully written" is
detected by polling file size until it stops changing, since editors/copies
can trigger multiple filesystem events for one file.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rag_supply_chain.ingestion.producer import DocumentProducer, new_doc_id
from rag_supply_chain.ingestion.validation import validate_file

logger = logging.getLogger(__name__)

SETTLE_INTERVAL = 0.5  # seconds between size checks
SETTLE_CHECKS = 2  # consecutive stable reads required before a file is "done"


def _wait_until_settled(path: Path) -> bool:
    """Poll file size until stable; returns False if the file disappears mid-write."""
    stable_count = 0
    last_size = -1
    while stable_count < SETTLE_CHECKS:
        if not path.exists():
            return False
        size = path.stat().st_size
        stable_count = stable_count + 1 if size == last_size else 0
        last_size = size
        time.sleep(SETTLE_INTERVAL)
    return True


def process_file(path: Path, producer: DocumentProducer) -> None:
    doc_id = new_doc_id()
    result = validate_file(path)
    if result.ok:
        producer.produce_raw(doc_id, path, result.mime_type)
        logger.info("routed %s -> raw (doc_id=%s)", path.name, doc_id)
    else:
        producer.produce_dlq(doc_id, path, result.mime_type, result.reason or "unknown")
        logger.warning("routed %s -> dlq (doc_id=%s): %s", path.name, doc_id, result.reason)


class _InboxHandler(FileSystemEventHandler):
    def __init__(self, producer: DocumentProducer) -> None:
        self._producer = producer
        self._in_flight: set[str] = set()

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._handle(Path(event.src_path))

    def on_moved(self, event) -> None:
        if not event.is_directory:
            self._handle(Path(event.dest_path))

    def _handle(self, path: Path) -> None:
        key = str(path.resolve())
        if key in self._in_flight:
            return
        self._in_flight.add(key)
        try:
            if _wait_until_settled(path):
                process_file(path, self._producer)
        finally:
            self._in_flight.discard(key)


def watch(inbox_dir: str | Path) -> None:
    inbox = Path(inbox_dir)
    inbox.mkdir(parents=True, exist_ok=True)
    producer = DocumentProducer()
    handler = _InboxHandler(producer)
    observer = Observer()
    observer.schedule(handler, str(inbox), recursive=False)
    observer.start()
    logger.info("watching %s", inbox.resolve())
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        producer.flush()
