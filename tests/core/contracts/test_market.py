"""Tests for core/contracts/market.py."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar, Instrument, Tick
from metatrade.core.enums import Timeframe

UTC = timezone.utc
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


class TestInstrument:
    def test_valid_construction(self) -> None:
        inst = Instrument(
            symbol="EURUSD",
            base_currency="EUR",
            quote_currency="USD",
            pip_size=Decimal("0.0001"),
            lot_size=Decimal("100000"),
            min_lot=Decimal("0.01"),
            max_lot=Decimal("500"),
            lot_step=Decimal("0.01"),
            digits=5,
        )
        assert inst.symbol == "EURUSD"

    def test_empty_symbol_raises(self) -> None:
        with pytest.raises(ValueError, match="symbol cannot be empty"):
            Instrument(
                symbol="",
                base_currency="EUR",
                quote_currency="USD",
                pip_size=Decimal("0.0001"),
                lot_size=Decimal("100000"),
                min_lot=Decimal("0.01"),
                max_lot=Decimal("500"),
                lot_step=Decimal("0.01"),
                digits=5,
            )

    def test_negative_pip_size_raises(self) -> None:
        with pytest.raises(ValueError, match="pip_size"):
            Instrument("EURUSD", "EUR", "USD",
                       pip_size=Decimal("-0.0001"),
                       lot_size=Decimal("100000"),
                       min_lot=Decimal("0.01"),
                       max_lot=Decimal("500"),
                       lot_step=Decimal("0.01"),
                       digits=5)

    def test_min_lot_greater_than_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min_lot.*max_lot"):
            Instrument("EURUSD", "EUR", "USD",
                       pip_size=Decimal("0.0001"),
                       lot_size=Decimal("100000"),
                       min_lot=Decimal("10"),
                       max_lot=Decimal("1"),
                       lot_step=Decimal("0.01"),
                       digits=5)

    def test_pip_value_per_lot_usd_quoted(self, eurusd: Instrument) -> None:
        pv = eurusd.pip_value_per_lot("USD")
        assert pv == Decimal("10")  # 0.0001 * 100_000

    def test_pip_value_per_lot_non_usd_raises(self, eurusd: Instrument) -> None:
        with pytest.raises(NotImplementedError):
            eurusd.pip_value_per_lot("EUR")

    def test_frozen(self, eurusd: Instrument) -> None:
        with pytest.raises(Exception):
            eurusd.symbol = "GBPUSD"  # type: ignore[misc]


class TestBar:
    def test_valid_bullish_bar(self, bullish_bar: Bar) -> None:
        assert bullish_bar.is_bullish
        assert not bullish_bar.is_bearish

    def test_valid_bearish_bar(self, bearish_bar: Bar) -> None:
        assert bearish_bar.is_bearish
        assert not bearish_bar.is_bullish

    def test_body_size(self, bullish_bar: Bar) -> None:
        expected = abs(bullish_bar.close - bullish_bar.open)
        assert bullish_bar.body_size == expected

    def test_range(self, bullish_bar: Bar) -> None:
        expected = bullish_bar.high - bullish_bar.low
        assert bullish_bar.range == expected

    def test_upper_wick(self, bullish_bar: Bar) -> None:
        expected = bullish_bar.high - max(bullish_bar.open, bullish_bar.close)
        assert bullish_bar.upper_wick == expected

    def test_lower_wick(self, bullish_bar: Bar) -> None:
        expected = min(bullish_bar.open, bullish_bar.close) - bullish_bar.low
        assert bullish_bar.lower_wick == expected

    def test_doji_detection(self) -> None:
        doji = Bar(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            timestamp_utc=T0,
            open=Decimal("1.10000"),
            high=Decimal("1.10200"),
            low=Decimal("1.09800"),
            close=Decimal("1.10001"),  # tiny body
            volume=Decimal("100"),
        )
        assert doji.is_doji

    def test_high_less_than_low_raises(self) -> None:
        with pytest.raises(ValueError, match="high.*low"):
            Bar(
                symbol="EURUSD",
                timeframe=Timeframe.H1,
                timestamp_utc=T0,
                open=Decimal("1.10000"),
                high=Decimal("1.09000"),  # high < low
                low=Decimal("1.10500"),
                close=Decimal("1.10200"),
                volume=Decimal("100"),
            )

    def test_open_outside_range_raises(self) -> None:
        with pytest.raises(ValueError, match="open.*outside"):
            Bar(
                symbol="EURUSD",
                timeframe=Timeframe.H1,
                timestamp_utc=T0,
                open=Decimal("1.12000"),  # above high
                high=Decimal("1.10500"),
                low=Decimal("1.09800"),
                close=Decimal("1.10200"),
                volume=Decimal("100"),
            )

    def test_close_outside_range_raises(self) -> None:
        with pytest.raises(ValueError, match="close.*outside"):
            Bar(
                symbol="EURUSD",
                timeframe=Timeframe.H1,
                timestamp_utc=T0,
                open=Decimal("1.10000"),
                high=Decimal("1.10500"),
                low=Decimal("1.09800"),
                close=Decimal("1.08000"),  # below low
                volume=Decimal("100"),
            )

    def test_negative_volume_raises(self) -> None:
        with pytest.raises(ValueError, match="volume"):
            Bar(
                symbol="EURUSD",
                timeframe=Timeframe.H1,
                timestamp_utc=T0,
                open=Decimal("1.10000"),
                high=Decimal("1.10500"),
                low=Decimal("1.09800"),
                close=Decimal("1.10200"),
                volume=Decimal("-1"),
            )

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            Bar(
                symbol="EURUSD",
                timeframe=Timeframe.H1,
                timestamp_utc=datetime(2024, 1, 1, 10, 0),  # naive
                open=Decimal("1.10000"),
                high=Decimal("1.10500"),
                low=Decimal("1.09800"),
                close=Decimal("1.10200"),
                volume=Decimal("100"),
            )

    def test_negative_spread_raises(self) -> None:
        with pytest.raises(ValueError, match="spread"):
            Bar(
                symbol="EURUSD",
                timeframe=Timeframe.H1,
                timestamp_utc=T0,
                open=Decimal("1.10000"),
                high=Decimal("1.10500"),
                low=Decimal("1.09800"),
                close=Decimal("1.10200"),
                volume=Decimal("100"),
                spread=Decimal("-1"),
            )

    def test_frozen(self, bullish_bar: Bar) -> None:
        with pytest.raises(Exception):
            bullish_bar.close = Decimal("1.20000")  # type: ignore[misc]


class TestTick:
    def test_valid_tick(self) -> None:
        tick = Tick(
            symbol="EURUSD",
            timestamp_utc=T0,
            bid=Decimal("1.10000"),
            ask=Decimal("1.10002"),
        )
        assert tick.spread == Decimal("0.00002")
        assert tick.mid == Decimal("1.10001")

    def test_ask_less_than_bid_raises(self) -> None:
        with pytest.raises(ValueError, match="ask.*bid"):
            Tick(
                symbol="EURUSD",
                timestamp_utc=T0,
                bid=Decimal("1.10005"),
                ask=Decimal("1.10000"),
            )

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            Tick(
                symbol="EURUSD",
                timestamp_utc=datetime(2024, 1, 1),
                bid=Decimal("1.10000"),
                ask=Decimal("1.10002"),
            )

    def test_zero_bid_raises(self) -> None:
        with pytest.raises(ValueError, match="bid must be > 0"):
            Tick(symbol="EURUSD", timestamp_utc=T0,
                 bid=Decimal("0"), ask=Decimal("0.00001"))

    def test_spread_is_zero_when_bid_equals_ask(self) -> None:
        tick = Tick(
            symbol="EURUSD",
            timestamp_utc=T0,
            bid=Decimal("1.10000"),
            ask=Decimal("1.10000"),
        )
        assert tick.spread == Decimal("0")

    def test_frozen(self) -> None:
        tick = Tick(symbol="EURUSD", timestamp_utc=T0,
                    bid=Decimal("1.1"), ask=Decimal("1.1001"))
        with pytest.raises(Exception):
            tick.bid = Decimal("1.2")  # type: ignore[misc]
