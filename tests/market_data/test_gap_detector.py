"""Tests for gap detector."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.market_data.gap_detector.gap_detector import GapDetector, _is_weekend_gap
from metatrade.market_data.storage.duckdb_store import DuckDBBarStore

UTC = timezone.utc
# Monday 2024-01-15 10:00 UTC
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def make_bar(ts: datetime, symbol: str = "EURUSD") -> Bar:
    return Bar(
        symbol=symbol, timeframe=Timeframe.H1, timestamp_utc=ts,
        open=Decimal("1.10000"), high=Decimal("1.10500"),
        low=Decimal("1.09800"), close=Decimal("1.10200"),
        volume=Decimal("1000"),
    )


@pytest.fixture
def store() -> DuckDBBarStore:
    s = DuckDBBarStore(":memory:")
    yield s
    s.close()


class TestGapDetector:
    def test_no_gaps_in_contiguous_series(self, store: DuckDBBarStore) -> None:
        bars = [make_bar(T0 + timedelta(hours=i)) for i in range(10)]
        store.save_bars(bars)
        detector = GapDetector(store, max_gap_multiplier=3.0)
        gaps = detector.detect_gaps(
            "EURUSD", Timeframe.H1,
            date_from=T0, date_to=T0 + timedelta(hours=12),
        )
        assert gaps == []

    def test_detects_gap_in_series(self, store: DuckDBBarStore) -> None:
        # Bars 0-4, then skip 5-6, then bar 7
        bars = (
            [make_bar(T0 + timedelta(hours=i)) for i in range(5)] +
            [make_bar(T0 + timedelta(hours=7))]
        )
        store.save_bars(bars)
        detector = GapDetector(store, max_gap_multiplier=2.0)
        gaps = detector.detect_gaps(
            "EURUSD", Timeframe.H1,
            date_from=T0, date_to=T0 + timedelta(hours=10),
        )
        assert len(gaps) == 1
        assert gaps[0].expected_bars_missing == 2

    def test_gap_start_and_end(self, store: DuckDBBarStore) -> None:
        t_before = T0 + timedelta(hours=3)
        t_after = T0 + timedelta(hours=7)
        bars = [
            make_bar(T0),
            make_bar(T0 + timedelta(hours=1)),
            make_bar(T0 + timedelta(hours=2)),
            make_bar(t_before),
            make_bar(t_after),  # gap: 4 hours interval, 3 missing bars
        ]
        store.save_bars(bars)
        detector = GapDetector(store, max_gap_multiplier=2.0)
        gaps = detector.detect_gaps(
            "EURUSD", Timeframe.H1,
            date_from=T0, date_to=T0 + timedelta(hours=10),
        )
        assert len(gaps) == 1
        assert gaps[0].gap_start_utc == t_before
        assert gaps[0].gap_end_utc == t_after

    def test_empty_store_no_gaps(self, store: DuckDBBarStore) -> None:
        detector = GapDetector(store)
        gaps = detector.detect_gaps(
            "EURUSD", Timeframe.H1,
            date_from=T0, date_to=T0 + timedelta(hours=10),
        )
        assert gaps == []

    def test_single_bar_no_gaps(self, store: DuckDBBarStore) -> None:
        store.save_bars([make_bar(T0)])
        detector = GapDetector(store)
        gaps = detector.detect_gaps(
            "EURUSD", Timeframe.H1,
            date_from=T0, date_to=T0 + timedelta(hours=10),
        )
        assert gaps == []

    def test_gap_duration_seconds(self, store: DuckDBBarStore) -> None:
        bars = [make_bar(T0), make_bar(T0 + timedelta(hours=5))]
        store.save_bars(bars)
        detector = GapDetector(store, max_gap_multiplier=2.0)
        gaps = detector.detect_gaps(
            "EURUSD", Timeframe.H1,
            date_from=T0, date_to=T0 + timedelta(hours=10),
        )
        assert gaps[0].gap_duration_seconds == 5 * 3600

    def test_detect_all_symbols(self, store: DuckDBBarStore) -> None:
        # EURUSD contiguous — no gaps
        store.save_bars([make_bar(T0 + timedelta(hours=i)) for i in range(5)])
        # GBPUSD has a gap
        store.save_bars([
            make_bar(T0, symbol="GBPUSD"),
            make_bar(T0 + timedelta(hours=5), symbol="GBPUSD"),
        ])
        detector = GapDetector(store, max_gap_multiplier=2.0)
        result = detector.detect_all_symbols(
            symbols=["EURUSD", "GBPUSD"],
            timeframes=[Timeframe.H1],
            date_from=T0, date_to=T0 + timedelta(hours=10),
        )
        assert "EURUSD/H1" not in result
        assert "GBPUSD/H1" in result


class TestWeekendGapHelper:
    def test_friday_to_monday_is_weekend(self) -> None:
        # Friday 22:00 UTC
        friday = datetime(2024, 1, 19, 22, 0, tzinfo=UTC)  # Friday
        monday = datetime(2024, 1, 22, 0, 0, tzinfo=UTC)   # Monday
        assert _is_weekend_gap(friday, monday)

    def test_monday_to_tuesday_not_weekend(self) -> None:
        monday = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
        tuesday = datetime(2024, 1, 16, 10, 0, tzinfo=UTC)
        assert not _is_weekend_gap(monday, tuesday)
