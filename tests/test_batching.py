"""Tests for the dynamic micro-batcher (§3.B) — size- and time-triggered flush."""

from __future__ import annotations

from rag_supply_chain.workers.batching import DynamicBatcher


class FakeClock:
    """A settable clock, so tests can jump time forward without sleeping."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_flushes_on_size_threshold() -> None:
    clock = FakeClock()
    batcher: DynamicBatcher[int] = DynamicBatcher(max_size=3, max_interval_seconds=100, clock=clock)

    batcher.add(1)
    assert not batcher.should_flush()
    batcher.add(2)
    assert not batcher.should_flush()
    batcher.add(3)
    assert batcher.should_flush()

    assert batcher.flush() == [1, 2, 3]
    assert not batcher.should_flush()  # buffer is empty again


def test_flushes_on_time_threshold_even_below_size() -> None:
    clock = FakeClock()
    batcher: DynamicBatcher[str] = DynamicBatcher(
        max_size=100, max_interval_seconds=2.0, clock=clock
    )

    batcher.add("x")
    assert not batcher.should_flush()
    clock.t = 5.0  # 5s later, well past the 2s interval
    assert batcher.should_flush()


def test_does_not_flush_before_interval_elapses() -> None:
    clock = FakeClock()
    batcher: DynamicBatcher[str] = DynamicBatcher(
        max_size=100, max_interval_seconds=2.0, clock=clock
    )

    batcher.add("x")
    clock.t = 1.0  # only 1s elapsed, interval is 2s
    assert not batcher.should_flush()


def test_empty_batcher_never_flushes() -> None:
    batcher: DynamicBatcher[int] = DynamicBatcher(max_size=1, max_interval_seconds=0.0)
    assert not batcher.should_flush()


def test_flush_resets_timer_for_next_batch() -> None:
    clock = FakeClock()
    batcher: DynamicBatcher[str] = DynamicBatcher(
        max_size=100, max_interval_seconds=2.0, clock=clock
    )
    batcher.add("a")
    clock.t = 3.0
    assert batcher.should_flush()  # 3s elapsed
    batcher.flush()

    batcher.add("b")  # timer restarts at t=3.0
    assert not batcher.should_flush()  # no time has passed since the restart
