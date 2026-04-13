"""UTC clock abstraction.

CRITICAL: Every timestamp in the system must come from a Clock instance.
Never call datetime.now(), datetime.utcnow(), or time.time() directly.

This abstraction has three concrete implementations:
- SystemClock   — production: delegates to the OS clock (always UTC-aware)
- FixedClock    — unit tests: returns a frozen, manually-advanceable time
- BacktestClock — backtest:  advances bar-by-bar, enforces monotonic time

Without this abstraction, backtesting and deterministic unit tests are impossible.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone


class Clock(ABC):
    """Abstract UTC clock used by every component in the system."""

    @abstractmethod
    def now_utc(self) -> datetime:
        """Return the current moment as a timezone-aware UTC datetime."""
        ...

    @abstractmethod
    def today_utc(self) -> date:
        """Return today's date in UTC."""
        ...


class SystemClock(Clock):
    """Production clock — reads from the OS system time."""

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def today_utc(self) -> date:
        return datetime.now(timezone.utc).date()


class FixedClock(Clock):
    """Deterministic clock for unit testing.

    Returns a fixed point in time that can be advanced manually via
    `advance()` or replaced entirely via `set_time()`.

    Example:
        clock = FixedClock(datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc))
        clock.now_utc()       # → 2024-01-15 10:00:00+00:00
        clock.advance(hours=1)
        clock.now_utc()       # → 2024-01-15 11:00:00+00:00
    """

    def __init__(self, fixed_time: datetime) -> None:
        _require_aware(fixed_time, "FixedClock")
        self._time = fixed_time

    def now_utc(self) -> datetime:
        return self._time

    def today_utc(self) -> date:
        return self._time.date()

    def advance(self, **kwargs: int | float) -> None:
        """Advance the clock by a timedelta (keyword args passed to timedelta)."""
        self._time += timedelta(**kwargs)

    def set_time(self, new_time: datetime) -> None:
        """Replace the current time with an arbitrary new value."""
        _require_aware(new_time, "FixedClock.set_time")
        self._time = new_time


class BacktestClock(Clock):
    """Monotonic clock for backtesting.

    Time advances bar-by-bar via `tick()`. Attempting to go backwards
    raises ValueError to catch look-ahead bugs early.

    Example:
        clock = BacktestClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        for bar in bars:
            clock.tick(bar.timestamp_utc)
            # system processes bar at bar.timestamp_utc, never in the future
    """

    def __init__(self, start_time: datetime) -> None:
        _require_aware(start_time, "BacktestClock")
        self._time = start_time

    def now_utc(self) -> datetime:
        return self._time

    def today_utc(self) -> date:
        return self._time.date()

    def tick(self, new_time: datetime) -> None:
        """Advance clock to `new_time`. Must be >= current time."""
        _require_aware(new_time, "BacktestClock.tick")
        if new_time < self._time:
            raise ValueError(
                f"BacktestClock cannot go backwards: "
                f"current={self._time.isoformat()}, "
                f"requested={new_time.isoformat()}"
            )
        self._time = new_time


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_aware(dt: datetime, source: str) -> None:
    if dt.tzinfo is None:
        raise ValueError(
            f"{source} requires a timezone-aware datetime, got naive: {dt.isoformat()!r}"
        )
