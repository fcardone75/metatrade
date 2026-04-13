"""Tests for MockFeed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.market_data.feed.mock_feed import MockFeed

UTC = timezone.utc
T0 = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture
def feed() -> MockFeed:
    f = MockFeed(seed=42)
    f.initialize()
    return f


class TestMockFeed:
    def test_initialize(self) -> None:
        feed = MockFeed()
        assert not feed.is_connected()
        feed.initialize()
        assert feed.is_connected()

    def test_shutdown(self, feed: MockFeed) -> None:
        feed.shutdown()
        assert not feed.is_connected()

    def test_generate_bars_count(self, feed: MockFeed) -> None:
        bars = feed.generate_bars(50)
        assert len(bars) == 50

    def test_generate_bars_are_valid(self, feed: MockFeed) -> None:
        bars = feed.generate_bars(10)
        for bar in bars:
            assert isinstance(bar, Bar)
            assert bar.high >= bar.low
            assert bar.low <= bar.open <= bar.high
            assert bar.low <= bar.close <= bar.high
            assert bar.volume >= 0

    def test_generate_bars_chronological(self, feed: MockFeed) -> None:
        bars = feed.generate_bars(10, start_ts=T0)
        timestamps = [b.timestamp_utc for b in bars]
        assert timestamps == sorted(timestamps)

    def test_generate_bars_h1_step(self) -> None:
        feed = MockFeed(timeframe=Timeframe.H1, seed=1)
        feed.initialize()
        bars = feed.generate_bars(3, start_ts=T0)
        assert (bars[1].timestamp_utc - bars[0].timestamp_utc).seconds == 3600
        assert (bars[2].timestamp_utc - bars[1].timestamp_utc).seconds == 3600

    def test_deterministic_with_seed(self) -> None:
        f1 = MockFeed(seed=99)
        f2 = MockFeed(seed=99)
        f1.initialize()
        f2.initialize()
        bars1 = f1.generate_bars(5, start_ts=T0)
        bars2 = f2.generate_bars(5, start_ts=T0)
        for b1, b2 in zip(bars1, bars2):
            assert b1.open == b2.open
            assert b1.close == b2.close

    def test_different_seeds_produce_different_bars(self) -> None:
        f1 = MockFeed(seed=1)
        f2 = MockFeed(seed=2)
        f1.initialize()
        f2.initialize()
        bars1 = f1.generate_bars(5, start_ts=T0)
        bars2 = f2.generate_bars(5, start_ts=T0)
        closes1 = [b.close for b in bars1]
        closes2 = [b.close for b in bars2]
        assert closes1 != closes2

    def test_get_latest_bars(self, feed: MockFeed) -> None:
        bars = feed.get_latest_bars("EURUSD", Timeframe.H1, 20)
        assert len(bars) == 20

    def test_get_bars_range(self, feed: MockFeed) -> None:
        bars = feed.get_bars_range(
            "EURUSD", Timeframe.H1,
            date_from=T0,
            date_to=T0 + timedelta(hours=5),
        )
        assert len(bars) == 6  # T0, T0+1h, ..., T0+5h

    def test_spread_assigned(self) -> None:
        feed = MockFeed(spread=Decimal("0.00020"), seed=42)
        feed.initialize()
        bars = feed.generate_bars(5, start_ts=T0)
        for bar in bars:
            assert bar.spread == Decimal("0.00020")

    def test_symbols_are_assigned(self) -> None:
        feed = MockFeed(symbol="GBPUSD", seed=42)
        feed.initialize()
        bars = feed.generate_bars(3, start_ts=T0)
        for bar in bars:
            assert bar.symbol == "GBPUSD"
