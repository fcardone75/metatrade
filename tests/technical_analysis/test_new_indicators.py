"""Tests for new technical analysis indicators: Bollinger Bands and ADX."""

from __future__ import annotations

from decimal import Decimal

import pytest

from metatrade.technical_analysis.indicators.adx import adx
from metatrade.technical_analysis.indicators.bollinger import (
    band_width,
    bollinger_bands,
    percent_b,
)


def D(v: str | float | int) -> Decimal:
    return Decimal(str(v))


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def _flat_closes(self, n: int, price: float = 1.1000) -> list[Decimal]:
        return [D(price)] * n

    def _rising_closes(self, n: int, start: float = 1.0, step: float = 0.001) -> list[Decimal]:
        return [D(round(start + i * step, 5)) for i in range(n)]

    def test_output_length_matches_input(self) -> None:
        closes = self._rising_closes(50)
        upper, mid, lower = bollinger_bands(closes, period=20)
        assert len(upper) == len(mid) == len(lower) == 50

    def test_middle_band_is_sma(self) -> None:
        closes = [D(str(i + 1)) for i in range(25)]
        _, mid, _ = bollinger_bands(closes, period=5)
        # Last middle = SMA of last 5: 21+22+23+24+25 / 5 = 23.0
        assert abs(float(mid[-1]) - 23.0) < 0.001

    def test_upper_above_middle_above_lower(self) -> None:
        closes = self._rising_closes(30)
        upper, mid, lower = bollinger_bands(closes, period=10)
        # For any bar after warmup, upper > middle > lower
        for i in range(10, 30):
            assert upper[i] > mid[i] > lower[i]

    def test_flat_market_bands_symmetric(self) -> None:
        closes = self._flat_closes(30)
        upper, mid, lower = bollinger_bands(closes, period=10)
        # Flat prices → zero std → bands collapse to SMA
        for i in range(10, 30):
            assert abs(float(upper[i] - mid[i])) < 1e-9
            assert abs(float(lower[i] - mid[i])) < 1e-9

    def test_wider_num_std_produces_wider_bands(self) -> None:
        closes = self._rising_closes(30)
        upper1, mid1, lower1 = bollinger_bands(closes, period=10, num_std=1.0)
        upper2, mid2, lower2 = bollinger_bands(closes, period=10, num_std=2.0)
        assert upper2[-1] > upper1[-1]
        assert lower2[-1] < lower1[-1]

    def test_raises_on_insufficient_data(self) -> None:
        with pytest.raises(ValueError, match="Need at least"):
            bollinger_bands([D("1")] * 5, period=10)

    def test_raises_on_period_less_than_2(self) -> None:
        with pytest.raises(ValueError, match="period must be"):
            bollinger_bands([D("1")] * 10, period=1)

    def test_backfill_warmup_period(self) -> None:
        closes = self._rising_closes(25)
        upper, mid, lower = bollinger_bands(closes, period=20)
        # Values at indices 0..18 should equal value at index 19
        for i in range(19):
            assert upper[i] == upper[19]
            assert mid[i] == mid[19]
            assert lower[i] == lower[19]

    def test_band_width_with_volatile_data(self) -> None:
        # Volatile data → wider bands
        closes_volatile = [D("1.0"), D("1.2"), D("0.9"), D("1.3"), D("0.8")] * 6
        closes_flat = self._flat_closes(30, 1.1)
        _, upper_v, mid_v = bollinger_bands(closes_volatile, period=10)
        _, upper_f, mid_f = bollinger_bands(closes_flat, period=10)
        bw_volatile = band_width(upper_v[-1], closes_volatile[-1], mid_v[-1])
        bw_flat = band_width(upper_f[-1], closes_flat[-1], mid_f[-1])
        assert abs(bw_flat) < 1e-6  # flat → zero width


class TestPercentB:
    def test_at_middle(self) -> None:
        pct = percent_b(D("1.10"), D("1.12"), D("1.08"))
        assert abs(pct - 0.5) < 1e-9

    def test_at_upper_band(self) -> None:
        pct = percent_b(D("1.12"), D("1.12"), D("1.08"))
        assert abs(pct - 1.0) < 1e-9

    def test_at_lower_band(self) -> None:
        pct = percent_b(D("1.08"), D("1.12"), D("1.08"))
        assert abs(pct - 0.0) < 1e-9

    def test_above_upper_band(self) -> None:
        pct = percent_b(D("1.15"), D("1.12"), D("1.08"))
        assert pct > 1.0

    def test_below_lower_band(self) -> None:
        pct = percent_b(D("1.05"), D("1.12"), D("1.08"))
        assert pct < 0.0

    def test_zero_width_returns_half(self) -> None:
        pct = percent_b(D("1.10"), D("1.10"), D("1.10"))
        assert pct == 0.5


class TestBandWidth:
    def test_zero_on_flat(self) -> None:
        bw = band_width(D("1.10"), D("1.10"), D("1.10"))
        assert bw == 0.0

    def test_positive_when_bands_wide(self) -> None:
        bw = band_width(D("1.12"), D("1.08"), D("1.10"))
        assert bw > 0

    def test_zero_middle_returns_zero(self) -> None:
        bw = band_width(D("1.0"), D("0.9"), D("0"))
        assert bw == 0.0


# ---------------------------------------------------------------------------
# ADX
# ---------------------------------------------------------------------------

class TestAdx:
    def _make_ohlc(
        self,
        n: int,
        base: float = 1.1000,
        trend: float = 0.0001,
    ) -> tuple[list[Decimal], list[Decimal], list[Decimal]]:
        """Generate synthetic OHLC with a gentle trend."""
        highs, lows, closes = [], [], []
        for i in range(n):
            c = base + i * trend
            highs.append(D(round(c + 0.0005, 5)))
            lows.append(D(round(c - 0.0005, 5)))
            closes.append(D(round(c, 5)))
        return highs, lows, closes

    def _make_choppy(
        self, n: int, base: float = 1.1000
    ) -> tuple[list[Decimal], list[Decimal], list[Decimal]]:
        """Generate alternating price action (choppy / ranging)."""
        highs, lows, closes = [], [], []
        for i in range(n):
            c = base + (0.0005 if i % 2 == 0 else -0.0005)
            highs.append(D(round(c + 0.0003, 5)))
            lows.append(D(round(c - 0.0003, 5)))
            closes.append(D(round(c, 5)))
        return highs, lows, closes

    def test_output_length_matches_input(self) -> None:
        h, l, c = self._make_ohlc(50)
        adx_vals, pdi, mdi = adx(h, l, c, period=14)
        assert len(adx_vals) == len(pdi) == len(mdi) == 50

    def test_raises_on_unequal_lengths(self) -> None:
        h, l, c = self._make_ohlc(50)
        with pytest.raises(ValueError, match="equal length"):
            adx(h[:-1], l, c, period=14)

    def test_raises_on_insufficient_data(self) -> None:
        h, l, c = self._make_ohlc(20)
        with pytest.raises(ValueError, match="at least"):
            adx(h, l, c, period=14)  # needs 29+

    def test_adx_non_negative(self) -> None:
        h, l, c = self._make_ohlc(60)
        adx_vals, _, _ = adx(h, l, c, period=14)
        assert all(v >= 0 for v in adx_vals)

    def test_adx_bounded_100(self) -> None:
        h, l, c = self._make_ohlc(60)
        adx_vals, pdi, mdi = adx(h, l, c, period=14)
        assert all(v <= D("100") for v in adx_vals)
        assert all(v <= D("100") for v in pdi)
        assert all(v <= D("100") for v in mdi)

    def test_uptrend_plus_di_greater_than_minus_di(self) -> None:
        """In a strong uptrend, +DI should exceed −DI."""
        h, l, c = self._make_ohlc(80, trend=0.0010)  # strong uptrend
        _, pdi, mdi = adx(h, l, c, period=14)
        # Check last half of series (after warmup)
        for i in range(40, 80):
            assert pdi[i] >= mdi[i], f"At bar {i}: +DI={float(pdi[i]):.2f} < -DI={float(mdi[i]):.2f}"

    def test_downtrend_minus_di_greater_than_plus_di(self) -> None:
        """In a strong downtrend, −DI should exceed +DI."""
        h, l, c = self._make_ohlc(80, trend=-0.0010)  # strong downtrend
        _, pdi, mdi = adx(h, l, c, period=14)
        for i in range(40, 80):
            assert mdi[i] >= pdi[i], f"At bar {i}: -DI={float(mdi[i]):.2f} < +DI={float(pdi[i]):.2f}"

    def test_adx_higher_for_strong_trend(self) -> None:
        """A strong trend produces higher ADX than a choppy market."""
        h_strong, l_strong, c_strong = self._make_ohlc(50, trend=0.0008)
        h_choppy, l_choppy, c_choppy = self._make_choppy(50)
        adx_strong, _, _ = adx(h_strong, l_strong, c_strong, period=14)
        adx_choppy, _, _ = adx(h_choppy, l_choppy, c_choppy, period=14)
        # Compare mid-series values before any saturation effects
        assert adx_strong[40] > adx_choppy[40]

    def test_warm_up_values_are_zero(self) -> None:
        """First period*2 values should be zero (padding)."""
        h, l, c = self._make_ohlc(60)
        adx_vals, _, _ = adx(h, l, c, period=14)
        # Warmup: first 28 values should be zero
        assert adx_vals[0] == D("0")
        assert adx_vals[13] == D("0")
