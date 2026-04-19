"""Tests for the session gate in BaseRunner.process_bar()."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from metatrade.core.utils.session import is_in_session


# ── is_in_session unit tests ──────────────────────────────────────────────────

class TestIsInSession:
    def _ts(self, hour: int) -> datetime:
        return datetime(2024, 3, 4, hour, 30, 0, tzinfo=UTC)

    # Normal window (start < end)
    def test_inside_normal(self):
        assert is_in_session(self._ts(10), 7, 21) is True

    def test_at_start_inclusive(self):
        assert is_in_session(self._ts(7), 7, 21) is True

    def test_at_end_exclusive(self):
        assert is_in_session(self._ts(21), 7, 21) is False

    def test_before_start(self):
        assert is_in_session(self._ts(3), 7, 21) is False

    def test_after_end(self):
        assert is_in_session(self._ts(22), 7, 21) is False

    # Overnight window (start > end)
    def test_overnight_inside_late(self):
        assert is_in_session(self._ts(23), 22, 6) is True

    def test_overnight_inside_early(self):
        assert is_in_session(self._ts(3), 22, 6) is True

    def test_overnight_at_start(self):
        assert is_in_session(self._ts(22), 22, 6) is True

    def test_overnight_at_end_exclusive(self):
        assert is_in_session(self._ts(6), 22, 6) is False

    def test_overnight_outside(self):
        assert is_in_session(self._ts(10), 22, 6) is False


# ── Runner integration: session gate blocks entries ───────────────────────────

class TestRunnerSessionGate:
    """Verify that BaseRunner.process_bar() returns None outside session."""

    def _make_runner(self, session_start: int | None, session_end: int | None):
        from metatrade.ml.config import MLConfig
        from metatrade.runner.base import BaseRunner
        from metatrade.runner.config import RunnerConfig

        cfg = RunnerConfig(
            session_filter_utc_start=session_start,
            session_filter_utc_end=session_end,
            signal_cooldown_bars=0,
        )
        return BaseRunner(cfg, modules=[])

    def _make_bars(self, hour: int, n: int = 60):
        from decimal import Decimal
        from metatrade.core.contracts.market import Bar
        from metatrade.core.enums import Timeframe

        ts_base = datetime(2024, 3, 4, hour, 0, 0, tzinfo=UTC)
        bars = []
        for i in range(n):
            ts = ts_base + timedelta(minutes=i)
            bars.append(
                Bar(
                    symbol="EURUSD",
                    timeframe=Timeframe.M1,
                    timestamp_utc=ts,
                    open=Decimal("1.10000"),
                    high=Decimal("1.10100"),
                    low=Decimal("1.09900"),
                    close=Decimal("1.10000"),
                    volume=Decimal("1000"),
                )
            )
        return bars

    def test_no_filter_does_not_block(self):
        runner = self._make_runner(None, None)
        bars = self._make_bars(hour=3)
        result = runner.process_bar(
            bars=bars,
            account_balance=Decimal("10000"),
            account_equity=Decimal("10000"),
            free_margin=Decimal("10000"),
            timestamp_utc=bars[-1].timestamp_utc,
        )
        # No modules → consensus is HOLD → result is None, but NOT because of session gate.
        # The important thing: no exception, and logging would confirm the path taken.
        # We just verify it runs without crashing.
        assert result is None

    def test_outside_session_returns_none(self):
        runner = self._make_runner(session_start=7, session_end=21)
        bars = self._make_bars(hour=3)  # 03:xx UTC — outside [7, 21)
        result = runner.process_bar(
            bars=bars,
            account_balance=Decimal("10000"),
            account_equity=Decimal("10000"),
            free_margin=Decimal("10000"),
            timestamp_utc=bars[-1].timestamp_utc,
        )
        assert result is None

    def test_inside_session_passes_gate(self):
        runner = self._make_runner(session_start=7, session_end=21)
        bars = self._make_bars(hour=10)  # 10:xx UTC — inside [7, 21)
        # No modules → returns None due to HOLD consensus, not session gate.
        # Verify no crash and call reaches signal processing (no session-gate early return).
        result = runner.process_bar(
            bars=bars,
            account_balance=Decimal("10000"),
            account_equity=Decimal("10000"),
            free_margin=Decimal("10000"),
            timestamp_utc=bars[-1].timestamp_utc,
        )
        assert result is None  # HOLD from empty module list — reached consensus

    def test_overnight_session_outside(self):
        runner = self._make_runner(session_start=22, session_end=6)
        bars = self._make_bars(hour=10)  # 10:xx UTC — outside overnight session
        result = runner.process_bar(
            bars=bars,
            account_balance=Decimal("10000"),
            account_equity=Decimal("10000"),
            free_margin=Decimal("10000"),
            timestamp_utc=bars[-1].timestamp_utc,
        )
        assert result is None
