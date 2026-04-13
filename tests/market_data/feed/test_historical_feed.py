"""Tests for HistoricalFeed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.market_data.errors import FeedNotInitializedError, InsufficientHistoryError
from metatrade.market_data.feed.historical_feed import HistoricalFeed
from metatrade.market_data.storage.duckdb_store import DuckDBBarStore

UTC = timezone.utc
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def make_bar(ts: datetime) -> Bar:
    return Bar(
        symbol="EURUSD", timeframe=Timeframe.H1, timestamp_utc=ts,
        open=Decimal("1.10000"), high=Decimal("1.10500"),
        low=Decimal("1.09800"), close=Decimal("1.10200"),
        volume=Decimal("1000"),
    )


@pytest.fixture
def store_with_bars() -> DuckDBBarStore:
    store = DuckDBBarStore(":memory:")
    bars = [make_bar(T0 + timedelta(hours=i)) for i in range(10)]
    store.save_bars(bars)
    return store


@pytest.fixture
def feed(store_with_bars: DuckDBBarStore) -> HistoricalFeed:
    f = HistoricalFeed(store_with_bars)
    f.initialize()
    return f


class TestHistoricalFeed:
    def test_initialize_sets_connected(self) -> None:
        store = DuckDBBarStore(":memory:")
        feed = HistoricalFeed(store)
        assert not feed.is_connected()
        feed.initialize()
        assert feed.is_connected()

    def test_shutdown_disconnects(self, feed: HistoricalFeed) -> None:
        feed.shutdown()
        assert not feed.is_connected()

    def test_get_latest_bars_count(self, feed: HistoricalFeed) -> None:
        bars = feed.get_latest_bars("EURUSD", Timeframe.H1, 5)
        assert len(bars) == 5

    def test_get_latest_bars_all(self, feed: HistoricalFeed) -> None:
        bars = feed.get_latest_bars("EURUSD", Timeframe.H1, 100)
        assert len(bars) == 10  # only 10 stored

    def test_get_latest_bars_are_most_recent(self, feed: HistoricalFeed) -> None:
        bars = feed.get_latest_bars("EURUSD", Timeframe.H1, 3)
        expected_last = T0 + timedelta(hours=9)
        assert bars[-1].timestamp_utc == expected_last

    def test_get_bars_range(self, feed: HistoricalFeed) -> None:
        bars = feed.get_bars_range(
            "EURUSD", Timeframe.H1,
            date_from=T0 + timedelta(hours=2),
            date_to=T0 + timedelta(hours=5),
        )
        assert len(bars) == 4

    def test_get_latest_bars_empty_store_raises(self) -> None:
        store = DuckDBBarStore(":memory:")
        feed = HistoricalFeed(store)
        feed.initialize()
        with pytest.raises(InsufficientHistoryError):
            feed.get_latest_bars("EURUSD", Timeframe.H1, 10)

    def test_not_initialized_raises(self) -> None:
        store = DuckDBBarStore(":memory:")
        feed = HistoricalFeed(store)
        with pytest.raises(FeedNotInitializedError):
            feed.get_latest_bars("EURUSD", Timeframe.H1, 5)
