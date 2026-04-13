"""Tests for core/clock.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from metatrade.core.clock import BacktestClock, FixedClock, SystemClock

UTC = timezone.utc


class TestSystemClock:
    def test_now_utc_is_aware(self) -> None:
        clock = SystemClock()
        now = clock.now_utc()
        assert now.tzinfo is not None

    def test_now_utc_is_utc(self) -> None:
        clock = SystemClock()
        now = clock.now_utc()
        assert now.utcoffset() == timedelta(0)

    def test_today_utc_returns_date(self) -> None:
        from datetime import date
        clock = SystemClock()
        today = clock.today_utc()
        assert isinstance(today, date)

    def test_now_utc_increases_over_time(self) -> None:
        import time
        clock = SystemClock()
        t1 = clock.now_utc()
        time.sleep(0.01)
        t2 = clock.now_utc()
        assert t2 > t1


class TestFixedClock:
    def test_returns_fixed_time(self) -> None:
        t = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        clock = FixedClock(t)
        assert clock.now_utc() == t

    def test_today_utc_matches(self) -> None:
        t = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
        clock = FixedClock(t)
        assert clock.today_utc() == t.date()

    def test_advance_hours(self) -> None:
        t = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
        clock = FixedClock(t)
        clock.advance(hours=2)
        assert clock.now_utc() == datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)

    def test_advance_minutes(self) -> None:
        t = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
        clock = FixedClock(t)
        clock.advance(minutes=30)
        assert clock.now_utc() == datetime(2024, 6, 1, 10, 30, 0, tzinfo=UTC)

    def test_advance_multiple_times(self) -> None:
        t = datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC)
        clock = FixedClock(t)
        clock.advance(hours=1)
        clock.advance(hours=1)
        assert clock.now_utc() == datetime(2024, 6, 1, 2, 0, 0, tzinfo=UTC)

    def test_set_time(self) -> None:
        t = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
        new_t = datetime(2024, 12, 31, 23, 59, 0, tzinfo=UTC)
        clock = FixedClock(t)
        clock.set_time(new_t)
        assert clock.now_utc() == new_t

    def test_set_time_can_go_backwards(self) -> None:
        """FixedClock (unlike BacktestClock) allows going backwards — for test flexibility."""
        t = datetime(2024, 6, 1, 10, 0, 0, tzinfo=UTC)
        earlier = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        clock = FixedClock(t)
        clock.set_time(earlier)
        assert clock.now_utc() == earlier

    def test_rejects_naive_datetime_on_init(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            FixedClock(datetime(2024, 1, 1, 10, 0, 0))  # no tzinfo

    def test_rejects_naive_datetime_on_set_time(self) -> None:
        clock = FixedClock(datetime(2024, 1, 1, tzinfo=UTC))
        with pytest.raises(ValueError, match="timezone-aware"):
            clock.set_time(datetime(2024, 6, 1, 10, 0, 0))  # no tzinfo

    def test_now_utc_is_immutable_without_advance(self) -> None:
        t = datetime(2024, 6, 1, tzinfo=UTC)
        clock = FixedClock(t)
        assert clock.now_utc() == clock.now_utc()


class TestBacktestClock:
    def test_returns_start_time(self) -> None:
        t = datetime(2024, 1, 1, tzinfo=UTC)
        clock = BacktestClock(t)
        assert clock.now_utc() == t

    def test_today_utc(self) -> None:
        t = datetime(2024, 1, 15, 9, 30, tzinfo=UTC)
        clock = BacktestClock(t)
        assert clock.today_utc() == t.date()

    def test_tick_advances_time(self) -> None:
        t0 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        t1 = datetime(2024, 1, 1, 11, 0, tzinfo=UTC)
        clock = BacktestClock(t0)
        clock.tick(t1)
        assert clock.now_utc() == t1

    def test_tick_same_time_is_allowed(self) -> None:
        t = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        clock = BacktestClock(t)
        clock.tick(t)  # same time — idempotent, no error
        assert clock.now_utc() == t

    def test_tick_backwards_raises(self) -> None:
        t1 = datetime(2024, 1, 1, 11, 0, tzinfo=UTC)
        t0 = datetime(2024, 1, 1, 10, 0, tzinfo=UTC)
        clock = BacktestClock(t1)
        with pytest.raises(ValueError, match="cannot go backwards"):
            clock.tick(t0)

    def test_tick_rejects_naive_datetime(self) -> None:
        clock = BacktestClock(datetime(2024, 1, 1, tzinfo=UTC))
        with pytest.raises(ValueError, match="timezone-aware"):
            clock.tick(datetime(2024, 1, 2))

    def test_rejects_naive_datetime_on_init(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            BacktestClock(datetime(2024, 1, 1))

    def test_multiple_ticks(self) -> None:
        clock = BacktestClock(datetime(2024, 1, 1, tzinfo=UTC))
        times = [datetime(2024, 1, 1, h, 0, tzinfo=UTC) for h in range(1, 6)]
        for t in times:
            clock.tick(t)
        assert clock.now_utc() == times[-1]
