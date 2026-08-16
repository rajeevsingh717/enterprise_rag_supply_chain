"""Dynamic micro-batching (§3.B): buffer items until a size or time
threshold is hit, whichever comes first, then flush.

Kept independent of Kafka so the flush-trigger logic (the part worth
getting right and testing precisely) doesn't need a live broker to test.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class DynamicBatcher(Generic[T]):
    def __init__(
        self,
        max_size: int,
        max_interval_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_size = max_size
        self.max_interval_seconds = max_interval_seconds
        self._clock = clock
        self._buffer: list[T] = []
        self._started_at: float | None = None

    def add(self, item: T) -> None:
        self._buffer.append(item)
        if self._started_at is None:
            self._started_at = self._clock()

    def __len__(self) -> int:
        return len(self._buffer)

    def should_flush(self) -> bool:
        if not self._buffer:
            return False
        if len(self._buffer) >= self.max_size:
            return True
        started_at = self._started_at
        return started_at is not None and self._clock() - started_at >= self.max_interval_seconds

    def flush(self) -> list[T]:
        items = self._buffer
        self._buffer = []
        self._started_at = None
        return items
