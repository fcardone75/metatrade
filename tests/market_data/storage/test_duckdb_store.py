"""Tests for DuckDB bar store."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.market_data.errors import BarStoreError
from metatrade.market_data.models import BarQuery
from metatrade.market_data.storage.duckdb_store import DuckDBBarStore

UTC = UTC
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def make_bar(symbol: str = "EURUSD", ts: datetime = T0, close: str = "1.10200") -> Bar:
    return Bar(
        symbol=symbol,
        timeframe=Timeframe.H1,
        timestamp_utc=ts,
        open=Decimal("1.10000"),
        high=Decimal("1.10500"),
        low=Decimal("1.09800"),
        close=Decimal(close),
        volume=Decimal("1000"),
    )


def make_bars(count: int, symbol: str = "EURUSD", step_hours: int = 1) -> list[Bar]:
    return [
        make_bar(symbol=symbol, ts=T0 + timedelta(hours=i * step_hours))
        for i in range(count)
    ]


@pytest.fixture
def store() -> DuckDBBarStore:
    s = DuckDBBarStore(":memory:")
    yield s
    s.close()


class TestDuckDBBarStoreLifecycle:
    def test_opens_in_memory(self) -> None:
        with DuckDBBarStore(":memory:") as store:
            assert store is not None

    def test_close_is_idempotent(self) -> None:
        store = DuckDBBarStore(":memory:")
        store.close()
        store.close()  # second close must not raise

    def test_context_manager(self) -> None:
        with DuckDBBarStore(":memory:") as store:
            store.save_bars([make_bar()])

    def test_read_only_cannot_write(self) -> None:
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
            path = f.name
        os.unlink(path)  # remove the empty file so DuckDB can create a valid one
        try:
            with DuckDBBarStore(path) as w:
                w.save_bars([make_bar()])
            with DuckDBBarStore(path, read_only=True) as r:
                with pytest.raises(BarStoreError, match="read-only"):
                    r.save_bars([make_bar()])
        finally:
            os.unlink(path)


class TestDuckDBBarStoreSave:
    def test_save_single_bar(self, store: DuckDBBarStore) -> None:
        written = store.save_bars([make_bar()])
        assert written == 1

    def test_save_empty_list(self, store: DuckDBBarStore) -> None:
        written = store.save_bars([])
        assert written == 0

    def test_save_multiple_bars(self, store: DuckDBBarStore) -> None:
        bars = make_bars(10)
        written = store.save_bars(bars)
        assert written == 10

    def test_upsert_idempotent(self, store: DuckDBBarStore) -> None:
        bar = make_bar()
        store.save_bars([bar])
        store.save_bars([bar])  # second save should not create duplicate
        assert store.count_bars("EURUSD", Timeframe.H1) == 1

    def test_save_updates_existing_bar(self, store: DuckDBBarStore) -> None:
        bar1 = make_bar(close="1.10200")
        bar2 = make_bar(close="1.10400")  # same ts, different close (within [low, high])
        store.save_bars([bar1])
        store.save_bars([bar2])
        count = store.count_bars("EURUSD", Timeframe.H1)
        assert count == 1

    def test_save_multiple_symbols(self, store: DuckDBBarStore) -> None:
        store.save_bars([make_bar("EURUSD")])
        store.save_bars([make_bar("GBPUSD")])
        assert store.count_bars("EURUSD", Timeframe.H1) == 1
        assert store.count_bars("GBPUSD", Timeframe.H1) == 1


class TestDuckDBBarStoreQuery:
    def test_get_bars_returns_all_in_range(self, store: DuckDBBarStore) -> None:
        bars = make_bars(5)
        store.save_bars(bars)
        query = BarQuery(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            date_from=T0,
            date_to=T0 + timedelta(hours=10),
        )
        result = store.get_bars(query)
        assert len(result) == 5

    def test_get_bars_filters_by_date(self, store: DuckDBBarStore) -> None:
        bars = make_bars(10)
        store.save_bars(bars)
        query = BarQuery(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            date_from=T0 + timedelta(hours=2),
            date_to=T0 + timedelta(hours=5),
        )
        result = store.get_bars(query)
        assert all(
            T0 + timedelta(hours=2) <= b.timestamp_utc <= T0 + timedelta(hours=5)
            for b in result
        )

    def test_get_bars_ordered_asc(self, store: DuckDBBarStore) -> None:
        bars = make_bars(5)
        store.save_bars(bars)
        query = BarQuery(
            symbol="EURUSD", timeframe=Timeframe.H1,
            date_from=T0, date_to=T0 + timedelta(hours=10),
        )
        result = store.get_bars(query)
        timestamps = [b.timestamp_utc for b in result]
        assert timestamps == sorted(timestamps)

    def test_get_bars_empty_range(self, store: DuckDBBarStore) -> None:
        query = BarQuery(
            symbol="EURUSD", timeframe=Timeframe.H1,
            date_from=T0, date_to=T0 + timedelta(hours=5),
        )
        assert store.get_bars(query) == []

    def test_get_bars_converts_to_decimal(self, store: DuckDBBarStore) -> None:
        store.save_bars([make_bar(close="1.10234")])
        query = BarQuery(
            symbol="EURUSD", timeframe=Timeframe.H1,
            date_from=T0 - timedelta(seconds=1),
            date_to=T0 + timedelta(seconds=1),
        )
        result = store.get_bars(query)
        assert isinstance(result[0].close, Decimal)

    def test_bar_roundtrip_preserves_data(self, store: DuckDBBarStore) -> None:
        original = Bar(
            symbol="EURUSD", timeframe=Timeframe.H1, timestamp_utc=T0,
            open=Decimal("1.10000"), high=Decimal("1.10500"),
            low=Decimal("1.09800"), close=Decimal("1.10200"),
            volume=Decimal("1500"), spread=Decimal("0.00020"),
        )
        store.save_bars([original])
        query = BarQuery(
            symbol="EURUSD", timeframe=Timeframe.H1,
            date_from=T0 - timedelta(seconds=1),
            date_to=T0 + timedelta(seconds=1),
        )
        result = store.get_bars(query)
        assert len(result) == 1
        r = result[0]
        assert r.symbol == original.symbol
        assert r.timeframe == original.timeframe
        assert r.open == original.open
        assert r.high == original.high
        assert r.low == original.low
        assert r.close == original.close


class TestDuckDBBarStoreMetadata:
    def test_latest_bar_time_none_when_empty(self, store: DuckDBBarStore) -> None:
        assert store.get_latest_bar_time("EURUSD", Timeframe.H1) is None

    def test_oldest_bar_time_none_when_empty(self, store: DuckDBBarStore) -> None:
        assert store.get_oldest_bar_time("EURUSD", Timeframe.H1) is None

    def test_latest_bar_time(self, store: DuckDBBarStore) -> None:
        bars = make_bars(5)
        store.save_bars(bars)
        latest = store.get_latest_bar_time("EURUSD", Timeframe.H1)
        assert latest == T0 + timedelta(hours=4)

    def test_oldest_bar_time(self, store: DuckDBBarStore) -> None:
        bars = make_bars(5)
        store.save_bars(bars)
        oldest = store.get_oldest_bar_time("EURUSD", Timeframe.H1)
        assert oldest == T0

    def test_count_bars(self, store: DuckDBBarStore) -> None:
        store.save_bars(make_bars(7))
        assert store.count_bars("EURUSD", Timeframe.H1) == 7

    def test_count_bars_zero_when_empty(self, store: DuckDBBarStore) -> None:
        assert store.count_bars("EURUSD", Timeframe.H1) == 0

    def test_coverage(self, store: DuckDBBarStore) -> None:
        store.save_bars(make_bars(5))
        coverage = store.get_coverage("EURUSD", Timeframe.H1)
        assert coverage.has_data
        assert coverage.total_bars == 5
        assert coverage.oldest_bar_utc == T0
        assert coverage.coverage_days is not None and coverage.coverage_days > 0

    def test_coverage_empty(self, store: DuckDBBarStore) -> None:
        coverage = store.get_coverage("EURUSD", Timeframe.H1)
        assert not coverage.has_data
        assert coverage.coverage_days is None


class TestDuckDBBarStoreDelete:
    def test_delete_bars_in_range(self, store: DuckDBBarStore) -> None:
        store.save_bars(make_bars(10))
        deleted = store.delete_bars(
            "EURUSD", Timeframe.H1,
            T0 + timedelta(hours=2),
            T0 + timedelta(hours=4),
        )
        assert deleted == 3  # hours 2, 3, 4
        assert store.count_bars("EURUSD", Timeframe.H1) == 7

    def test_delete_nonexistent_returns_zero(self, store: DuckDBBarStore) -> None:
        deleted = store.delete_bars("EURUSD", Timeframe.H1, T0, T0 + timedelta(hours=5))
        assert deleted == 0
