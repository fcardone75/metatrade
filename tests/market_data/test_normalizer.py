"""Tests for bar normalizer."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.market_data.models import RawBar
from metatrade.market_data.normalizer.bar_normalizer import BarNormalizer

UTC = UTC
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
TS0 = int(T0.timestamp())


def make_raw(**kwargs) -> RawBar:
    defaults = dict(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_unix=TS0,
        open=1.10000,
        high=1.10500,
        low=1.09800,
        close=1.10200,
        volume=1500.0,
    )
    defaults.update(kwargs)
    return RawBar(**defaults)


class TestBarNormalizer:
    def test_valid_normalization(self) -> None:
        n = BarNormalizer()
        bar = n.normalize_raw(make_raw())
        assert isinstance(bar, Bar)
        assert bar.symbol == "EURUSD"
        assert bar.timeframe == Timeframe.H1
        assert bar.timestamp_utc.tzinfo is not None

    def test_timestamp_is_utc(self) -> None:
        n = BarNormalizer()
        bar = n.normalize_raw(make_raw(timestamp_unix=TS0))
        assert bar.timestamp_utc == T0

    def test_prices_are_decimal(self) -> None:
        n = BarNormalizer()
        bar = n.normalize_raw(make_raw())
        assert isinstance(bar.open, Decimal)
        assert isinstance(bar.high, Decimal)
        assert isinstance(bar.low, Decimal)
        assert isinstance(bar.close, Decimal)

    def test_price_quantization_5_digits(self) -> None:
        n = BarNormalizer(price_digits=5)
        bar = n.normalize_raw(make_raw(close=1.1023456789))
        assert bar.close == Decimal("1.10235")

    def test_price_quantization_3_digits(self) -> None:
        n = BarNormalizer(price_digits=3)
        bar = n.normalize_raw(make_raw(open=150.123456, high=150.567, low=149.987, close=150.234))
        assert bar.open == Decimal("150.123")

    def test_spread_normalization(self) -> None:
        n = BarNormalizer(spread_divisor=10)
        bar = n.normalize_raw(make_raw(spread=20.0))  # 20 points → 0.00200
        assert bar.spread == Decimal("0.00200")

    def test_spread_none_when_zero(self) -> None:
        n = BarNormalizer()
        # spread=0.0 should yield None (no spread info)
        raw = make_raw(spread=0.0)
        bar = n.normalize_raw(raw)
        # spread=None because we only set it when > 0
        # The raw.spread=0.0 path gives None in normalize_raw via the None default
        assert bar.spread is None

    def test_volume_rounded_to_int(self) -> None:
        n = BarNormalizer()
        bar = n.normalize_raw(make_raw(volume=1234.7))
        assert bar.volume == Decimal("1235")

    def test_negative_volume_clamped_to_zero(self) -> None:
        n = BarNormalizer()
        bar = n.normalize_raw(make_raw(volume=-10.0))
        assert bar.volume == Decimal("0")

    def test_corrupt_row_raises_when_skip_false(self) -> None:
        n = BarNormalizer(skip_corrupt=False)
        raw = make_raw(high=1.09000, low=1.10500)  # high < low → invalid
        with pytest.raises(Exception):
            n.normalize_raw(raw)

    def test_normalize_csv_row(self) -> None:
        n = BarNormalizer()
        bar = n.normalize_csv_row(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            timestamp_unix=TS0,
            open_=1.10000,
            high=1.10500,
            low=1.09800,
            close=1.10200,
            volume=1000.0,
        )
        assert isinstance(bar, Bar)
        assert bar.spread is None

    def test_invalid_price_digits_raises(self) -> None:
        with pytest.raises(ValueError, match="price_digits"):
            BarNormalizer(price_digits=0)

    def test_invalid_price_digits_too_high(self) -> None:
        with pytest.raises(ValueError, match="price_digits"):
            BarNormalizer(price_digits=9)

    def test_normalize_mt5_rates_valid(self) -> None:
        """Test with a mock numpy-like structured array."""
        import numpy as np
        dtype = [
            ("time", "<i8"), ("open", "<f8"), ("high", "<f8"),
            ("low", "<f8"), ("close", "<f8"), ("tick_volume", "<u8"),
            ("spread", "<i4"), ("real_volume", "<u8"),
        ]
        rates = np.array(
            [(TS0, 1.10000, 1.10500, 1.09800, 1.10200, 1500, 20, 0)],
            dtype=dtype,
        )
        n = BarNormalizer()
        bars, skipped = n.normalize_mt5_rates("EURUSD", Timeframe.H1, rates)
        assert len(bars) == 1
        assert skipped == 0
        assert bars[0].symbol == "EURUSD"

    def test_normalize_mt5_rates_skips_corrupt(self) -> None:
        import numpy as np
        dtype = [
            ("time", "<i8"), ("open", "<f8"), ("high", "<f8"),
            ("low", "<f8"), ("close", "<f8"), ("tick_volume", "<u8"),
            ("spread", "<i4"), ("real_volume", "<u8"),
        ]
        # Second row has high < low — corrupt
        rates = np.array(
            [
                (TS0, 1.10000, 1.10500, 1.09800, 1.10200, 1500, 20, 0),
                (TS0 + 3600, 1.10200, 1.09000, 1.11000, 1.10500, 1000, 20, 0),
            ],
            dtype=dtype,
        )
        n = BarNormalizer(skip_corrupt=True)
        bars, skipped = n.normalize_mt5_rates("EURUSD", Timeframe.H1, rates)
        assert skipped == 1
        assert len(bars) == 1
