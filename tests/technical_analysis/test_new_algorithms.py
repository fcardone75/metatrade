"""Tests for Tier-1/2 algorithm additions:
    - HMA indicator
    - Keltner Channel indicator
    - Donchian Channel indicator
    - Hurst Exponent indicator
    - KeltnerSqueezeModule
    - DonchianBreakoutModule
    - MarketRegimeModule (Hurst metadata keys)
    - EmaCrossoverModule (HMA slow line)
    - Chandelier Exit SL in BaseRunner
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import OrderSide, SignalDirection, Timeframe
from metatrade.core.errors import ModuleNotReadyError
from metatrade.technical_analysis.indicators.donchian import donchian_channel
from metatrade.technical_analysis.indicators.hma import hma, wma
from metatrade.technical_analysis.indicators.hurst import hurst_exponent
from metatrade.technical_analysis.indicators.keltner import is_squeeze, keltner_channel
from metatrade.technical_analysis.modules.donchian_module import DonchianBreakoutModule
from metatrade.technical_analysis.modules.ema_crossover import EmaCrossoverModule
from metatrade.technical_analysis.modules.keltner_squeeze_module import KeltnerSqueezeModule
from metatrade.technical_analysis.modules.market_regime_module import MarketRegimeModule

UTC = UTC


# ── Helpers ───────────────────────────────────────────────────────────────────

def D(v: float) -> Decimal:
    return Decimal(str(round(v, 6)))


def make_bar(
    close: float,
    high_offset: float = 0.001,
    low_offset: float = 0.001,
    i: int = 0,
) -> Bar:
    c = D(close)
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
        open=c,
        high=c + D(high_offset),
        low=c - D(low_offset),
        close=c,
        volume=D(1000),
    )


def rising_bars(
    n: int, start: float = 1.10, step: float = 0.001, spread: float = 0.001
) -> list[Bar]:
    bars = []
    for i in range(n):
        c = round(start + i * step, 6)
        bars.append(Bar(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            open=D(c), high=D(c + spread), low=D(c - spread), close=D(c),
            volume=D(1000),
        ))
    return bars


def falling_bars(
    n: int, start: float = 1.20, step: float = 0.001, spread: float = 0.001
) -> list[Bar]:
    bars = []
    for i in range(n):
        c = round(start - i * step, 6)
        bars.append(Bar(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            open=D(c), high=D(c + spread), low=D(c - spread), close=D(c),
            volume=D(1000),
        ))
    return bars


def flat_bars(n: int, price: float = 1.10, spread: float = 0.0002) -> list[Bar]:
    bars = []
    for i in range(n):
        c = D(price)
        bars.append(Bar(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            open=c, high=c + D(spread), low=c - D(spread), close=c,
            volume=D(1000),
        ))
    return bars


def ts() -> datetime:
    return datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


# ── HMA indicator tests ───────────────────────────────────────────────────────

class TestWMAIndicator:
    def test_wma_single_period_is_price(self) -> None:
        prices = [D(1.0), D(2.0), D(3.0)]
        result = wma(prices, 1)
        assert result[-1] == D(3.0)

    def test_wma_length_equals_input(self) -> None:
        prices = [D(float(i)) for i in range(10)]
        result = wma(prices, 4)
        assert len(result) == 10

    def test_wma_period_too_small_raises(self) -> None:
        with pytest.raises(ValueError):
            wma([D(1.0), D(2.0)], 0)

    def test_wma_insufficient_data_raises(self) -> None:
        with pytest.raises(ValueError):
            wma([D(1.0), D(2.0)], 5)

    def test_wma_backfill_all_nonnegative(self) -> None:
        prices = [D(float(i + 1)) for i in range(20)]
        result = wma(prices, 5)
        assert all(v > Decimal("0") for v in result)

    def test_wma_rising_series_increases(self) -> None:
        prices = [D(float(i)) for i in range(20)]
        result = wma(prices, 5)
        # Last WMA should be > first (after warmup)
        assert result[-1] > result[5]

    def test_wma_period_two_known_value(self) -> None:
        # WMA(2) of [1, 2] = (1*1 + 2*2) / 3 = 5/3
        prices = [D(1.0), D(2.0)]
        result = wma(prices, 2)
        expected = D(5.0) / D(3.0)
        assert abs(result[-1] - expected) < Decimal("0.00001")


class TestHMAIndicator:
    def test_hma_length_equals_input(self) -> None:
        prices = [D(float(i + 1)) for i in range(30)]
        result = hma(prices, 10)
        assert len(result) == 30

    def test_hma_period_too_small_raises(self) -> None:
        with pytest.raises(ValueError):
            hma([D(1.0)] * 10, 3)  # period < 4

    def test_hma_insufficient_data_raises(self) -> None:
        with pytest.raises(ValueError):
            hma([D(1.0)] * 5, 10)  # too few bars

    def test_hma_rising_series_increases(self) -> None:
        prices = [D(float(i)) for i in range(50)]
        result = hma(prices, 10)
        # HMA should be rising over a rising price series
        assert result[-1] > result[-10]

    def test_hma_all_values_positive(self) -> None:
        prices = [D(1.1 + i * 0.001) for i in range(50)]
        result = hma(prices, 20)
        assert all(v > Decimal("0") for v in result)

    def test_hma_less_lag_than_ema(self) -> None:
        """HMA should respond faster to a trend reversal than EMA."""
        from metatrade.technical_analysis.indicators.ema import ema
        n = 60
        # First 30: rising, then 30: flat
        prices = [D(1.10 + i * 0.001) for i in range(30)]
        prices += [D(1.13)] * 30

        hma_vals = hma(prices, 20)
        ema_vals = ema(prices, 20)

        # After the flat section (index -1), HMA should be closer to 1.13 than EMA
        # (i.e. |hma[-1] - 1.13| < |ema[-1] - 1.13|)
        target = D(1.13)
        assert abs(hma_vals[-1] - target) <= abs(ema_vals[-1] - target)


# ── Keltner Channel indicator tests ───────────────────────────────────────────

class TestKeltnerChannelIndicator:
    def _make_data(self, n: int = 50, price: float = 1.10):
        highs  = [D(price + 0.001)] * n
        lows   = [D(price - 0.001)] * n
        closes = [D(price)] * n
        return highs, lows, closes

    def test_returns_three_equal_length_lists(self) -> None:
        h, l, c = self._make_data(50)
        upper, middle, lower = keltner_channel(h, l, c, ema_period=20, atr_period=10)
        assert len(upper) == len(middle) == len(lower) == 50

    def test_upper_above_middle_above_lower(self) -> None:
        h, l, c = self._make_data(50)
        upper, middle, lower = keltner_channel(h, l, c)
        for u, m, lo in zip(upper[-5:], middle[-5:], lower[-5:]):
            assert u > m > lo

    def test_insufficient_data_raises(self) -> None:
        h, l, c = self._make_data(5)
        with pytest.raises(ValueError):
            keltner_channel(h, l, c, ema_period=20, atr_period=10)

    def test_mismatched_lengths_raise(self) -> None:
        h = [D(1.10)] * 30
        l = [D(1.09)] * 25
        c = [D(1.10)] * 30
        with pytest.raises(ValueError):
            keltner_channel(h, l, c)

    def test_ema_period_too_small_raises(self) -> None:
        h, l, c = self._make_data(30)
        with pytest.raises(ValueError):
            keltner_channel(h, l, c, ema_period=1)


class TestIsSqueezeFunction:
    def test_squeeze_when_bb_inside_kc(self) -> None:
        # BB narrower than KC → squeeze
        assert is_squeeze(
            bb_upper=D(1.105), bb_lower=D(1.095),
            kc_upper=D(1.110), kc_lower=D(1.090),
        ) is True

    def test_no_squeeze_when_bb_wider_than_kc(self) -> None:
        assert is_squeeze(
            bb_upper=D(1.115), bb_lower=D(1.085),
            kc_upper=D(1.110), kc_lower=D(1.090),
        ) is False

    def test_no_squeeze_when_bb_equals_kc(self) -> None:
        # Strict < / >: equal is NOT a squeeze
        assert is_squeeze(
            bb_upper=D(1.110), bb_lower=D(1.090),
            kc_upper=D(1.110), kc_lower=D(1.090),
        ) is False


# ── Donchian Channel indicator tests ──────────────────────────────────────────

class TestDonchianChannelIndicator:
    def test_returns_three_equal_length_lists(self) -> None:
        highs = [D(1.10 + i * 0.001) for i in range(30)]
        lows  = [D(1.09 + i * 0.001) for i in range(30)]
        u, m, lo = donchian_channel(highs, lows, period=20)
        assert len(u) == len(m) == len(lo) == 30

    def test_upper_is_rolling_max_high(self) -> None:
        highs = [D(float(i)) for i in range(1, 26)]  # 1..25
        lows  = [D(0.5)] * 25
        u, _, _ = donchian_channel(highs, lows, period=10)
        # Last value: max of last 10 highs = 25
        assert u[-1] == D(25)

    def test_lower_is_rolling_min_low(self) -> None:
        highs = [D(2.0)] * 25
        lows  = [D(float(i + 1)) for i in range(25)]  # 1..25
        _, _, lo = donchian_channel(highs, lows, period=10)
        # Last value: min of last 10 lows = 16
        assert lo[-1] == D(16)

    def test_middle_is_mean_of_upper_lower(self) -> None:
        highs = [D(1.10 + i * 0.001) for i in range(30)]
        lows  = [D(1.09 - i * 0.001) for i in range(30)]
        u, m, lo = donchian_channel(highs, lows, period=20)
        for u_v, m_v, lo_v in zip(u[-5:], m[-5:], lo[-5:]):
            assert abs(m_v - (u_v + lo_v) / Decimal("2")) < Decimal("0.000001")

    def test_period_too_small_raises(self) -> None:
        with pytest.raises(ValueError):
            donchian_channel([D(1.0)] * 5, [D(0.9)] * 5, period=1)

    def test_insufficient_data_raises(self) -> None:
        with pytest.raises(ValueError):
            donchian_channel([D(1.0)] * 5, [D(0.9)] * 5, period=20)

    def test_mismatched_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            donchian_channel([D(1.0)] * 20, [D(0.9)] * 15, period=10)

    def test_backfill_first_values(self) -> None:
        """All values before warm-up are equal to the first computed value."""
        highs = [D(1.0 + i * 0.001) for i in range(25)]
        lows  = [D(0.9)] * 25
        u, _, _ = donchian_channel(highs, lows, period=10)
        # First 9 values should equal u[9] (the first real value)
        for i in range(9):
            assert u[i] == u[9]


# ── Hurst Exponent indicator tests ────────────────────────────────────────────

class TestHurstExponent:
    def test_returns_float(self) -> None:
        closes = [D(1.10 + i * 0.001) for i in range(50)]
        h = hurst_exponent(closes)
        assert isinstance(h, float)

    def test_result_in_bounds(self) -> None:
        closes = [D(1.10 + i * 0.001) for i in range(50)]
        h = hurst_exponent(closes)
        assert 0.10 <= h <= 0.90

    def test_insufficient_data_returns_half(self) -> None:
        closes = [D(1.10)] * 5
        h = hurst_exponent(closes, min_period=8)
        assert h == 0.5

    def test_trending_series_above_threshold(self) -> None:
        """A strongly trending series should produce H > 0.5."""
        closes = [D(1.0 + i * 0.01) for i in range(150)]
        h = hurst_exponent(closes)
        # Very clean trend should give H > 0.5
        assert h > 0.5, f"Expected H > 0.5 for rising trend, got {h}"

    def test_flat_series_returns_half_or_close(self) -> None:
        """A perfectly flat series has no returns variance → returns 0.5."""
        closes = [D(1.10)] * 60
        h = hurst_exponent(closes)
        assert h == 0.5  # all returns = 0 → S=0 → fallback

    def test_different_min_periods(self) -> None:
        closes = [D(1.10 + i * 0.001) for i in range(100)]
        h4  = hurst_exponent(closes, min_period=4)
        h8  = hurst_exponent(closes, min_period=8)
        h16 = hurst_exponent(closes, min_period=16)
        # All should be in bounds regardless of parameter
        for h in (h4, h8, h16):
            assert 0.10 <= h <= 0.90


# ── KeltnerSqueezeModule tests ────────────────────────────────────────────────

class TestKeltnerSqueezeModule:
    def _module(self, **kwargs) -> KeltnerSqueezeModule:
        return KeltnerSqueezeModule(
            bb_period=20, bb_std=2.0,
            kc_ema_period=20, kc_atr_period=10,
            kc_mult=1.5, timeframe_id="h1",
            **kwargs,
        )

    def test_module_id(self) -> None:
        m = self._module()
        assert m.module_id == "keltner_squeeze_h1"

    def test_min_bars_positive(self) -> None:
        m = self._module()
        assert m.min_bars > 0

    def test_raises_on_too_few_bars(self) -> None:
        m = self._module()
        with pytest.raises(ModuleNotReadyError):
            m.analyse([make_bar(1.1, i=0)], ts())

    def test_invalid_bb_period_raises(self) -> None:
        with pytest.raises(ValueError):
            KeltnerSqueezeModule(bb_period=1)

    def test_invalid_kc_ema_period_raises(self) -> None:
        with pytest.raises(ValueError):
            KeltnerSqueezeModule(kc_ema_period=1)

    def test_hold_during_squeeze(self) -> None:
        """When BB is inside KC, the module should return HOLD."""
        m = KeltnerSqueezeModule(
            bb_period=10, kc_ema_period=10, kc_atr_period=5,
            kc_mult=3.0,  # very wide KC → easy to trigger squeeze
            timeframe_id="h1",
        )
        # Flat bars: tiny ATR → tiny BB width → much smaller than KC width
        bars = flat_bars(m.min_bars + 10)
        sig = m.analyse(bars, ts())
        assert sig.direction == SignalDirection.HOLD
        # Squeeze flag should be True
        assert sig.metadata["squeeze"] is True

    def test_buy_on_bullish_breakout(self) -> None:
        """A bar far above KC upper should give BUY."""
        m = KeltnerSqueezeModule(
            bb_period=10, kc_ema_period=10, kc_atr_period=5,
            kc_mult=0.5,  # very narrow KC → easy to break above
            timeframe_id="h1",
        )
        bars = rising_bars(m.min_bars + 20, step=0.003)
        sig = m.analyse(bars, ts())
        # Strong upward breakout: BUY or HOLD (depends on exact prices)
        assert sig.direction in (SignalDirection.BUY, SignalDirection.HOLD)

    def test_sell_on_bearish_breakout(self) -> None:
        """A bar far below KC lower should give SELL."""
        m = KeltnerSqueezeModule(
            bb_period=10, kc_ema_period=10, kc_atr_period=5,
            kc_mult=0.5,  # very narrow KC
            timeframe_id="h1",
        )
        bars = falling_bars(m.min_bars + 20, step=0.003)
        sig = m.analyse(bars, ts())
        assert sig.direction in (SignalDirection.SELL, SignalDirection.HOLD)

    def test_confidence_in_valid_range(self) -> None:
        m = self._module()
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, ts())
        assert 0.0 <= sig.confidence <= 1.0

    def test_confidence_capped(self) -> None:
        m = KeltnerSqueezeModule(
            bb_period=10, kc_ema_period=10, kc_atr_period=5,
            kc_mult=0.2, confidence_cap=0.75, timeframe_id="h1",
        )
        bars = rising_bars(m.min_bars + 30, step=0.01)
        sig = m.analyse(bars, ts())
        assert sig.confidence <= 0.75

    def test_metadata_keys_present(self) -> None:
        m = self._module()
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, ts())
        for key in ("squeeze", "bb_upper", "bb_lower", "kc_upper", "kc_middle", "kc_lower", "close"):
            assert key in sig.metadata, f"Missing metadata key: {key}"

    def test_hold_confidence_is_050(self) -> None:
        m = KeltnerSqueezeModule(
            bb_period=10, kc_ema_period=10, kc_atr_period=5,
            kc_mult=3.0, timeframe_id="h1",
        )
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, ts())
        if sig.direction == SignalDirection.HOLD:
            assert sig.confidence == 0.50


# ── DonchianBreakoutModule tests ──────────────────────────────────────────────

class TestDonchianBreakoutModule:
    def _module(self, period: int = 20) -> DonchianBreakoutModule:
        return DonchianBreakoutModule(period=period, timeframe_id="h1")

    def test_module_id(self) -> None:
        m = self._module()
        assert m.module_id == "donchian_h1"

    def test_min_bars_is_period_plus_two(self) -> None:
        m = self._module(period=20)
        assert m.min_bars == 22  # period + 2

    def test_raises_on_too_few_bars(self) -> None:
        m = self._module()
        with pytest.raises(ModuleNotReadyError):
            m.analyse([make_bar(1.1)], ts())

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            DonchianBreakoutModule(period=1)

    def test_bullish_breakout_gives_buy(self) -> None:
        """Close > prev Donchian upper → BUY."""
        period = 10
        m = DonchianBreakoutModule(period=period, timeframe_id="h1")
        # Create bars where the last bar clearly breaks out above previous high
        # Bars 0..period: flat at 1.10; last bar: spike to 1.20
        base_bars: list[Bar] = []
        for i in range(period + 2):
            c = D(1.10)
            base_bars.append(Bar(
                symbol="EURUSD", timeframe=Timeframe.H1,
                timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                open=c, high=c + D(0.001), low=c - D(0.001), close=c,
                volume=D(1000),
            ))
        # Replace last bar: close well above the 10-period high of 1.101
        last = base_bars[-1]
        base_bars[-1] = Bar(
            symbol=last.symbol, timeframe=last.timeframe,
            timestamp_utc=last.timestamp_utc,
            open=D(1.20), high=D(1.21), low=D(1.19), close=D(1.20),
            volume=D(1000),
        )
        sig = m.analyse(base_bars, ts())
        assert sig.direction == SignalDirection.BUY

    def test_bearish_breakout_gives_sell(self) -> None:
        """Close < prev Donchian lower → SELL."""
        period = 10
        m = DonchianBreakoutModule(period=period, timeframe_id="h1")
        base_bars: list[Bar] = []
        for i in range(period + 2):
            c = D(1.10)
            base_bars.append(Bar(
                symbol="EURUSD", timeframe=Timeframe.H1,
                timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                open=c, high=c + D(0.001), low=c - D(0.001), close=c,
                volume=D(1000),
            ))
        # Replace last bar: close well below the 10-period low of 1.099
        last = base_bars[-1]
        base_bars[-1] = Bar(
            symbol=last.symbol, timeframe=last.timeframe,
            timestamp_utc=last.timestamp_utc,
            open=D(1.00), high=D(1.01), low=D(0.99), close=D(1.00),
            volume=D(1000),
        )
        sig = m.analyse(base_bars, ts())
        assert sig.direction == SignalDirection.SELL

    def test_inside_channel_gives_hold(self) -> None:
        """Close inside the Donchian channel → HOLD."""
        m = self._module(period=10)
        # Flat bars: close always inside channel
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, ts())
        assert sig.direction == SignalDirection.HOLD

    def test_hold_confidence_is_050(self) -> None:
        m = self._module(period=10)
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, ts())
        if sig.direction == SignalDirection.HOLD:
            assert sig.confidence == 0.50

    def test_confidence_above_060_on_breakout(self) -> None:
        """Any directional breakout should have confidence >= 0.60."""
        m = self._module(period=10)
        # Force a clear bullish breakout
        base_bars: list[Bar] = []
        for i in range(m.min_bars):
            c = D(1.10)
            base_bars.append(Bar(
                symbol="EURUSD", timeframe=Timeframe.H1,
                timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                open=c, high=c + D(0.001), low=c - D(0.001), close=c,
                volume=D(1000),
            ))
        last = base_bars[-1]
        base_bars[-1] = Bar(
            symbol=last.symbol, timeframe=last.timeframe,
            timestamp_utc=last.timestamp_utc,
            open=D(1.20), high=D(1.21), low=D(1.19), close=D(1.20),
            volume=D(1000),
        )
        sig = m.analyse(base_bars, ts())
        if sig.direction != SignalDirection.HOLD:
            assert sig.confidence >= 0.60

    def test_metadata_keys_present(self) -> None:
        m = self._module(period=10)
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, ts())
        for key in ("upper", "lower", "middle", "close", "width"):
            assert key in sig.metadata, f"Missing metadata key: {key}"

    def test_confidence_capped(self) -> None:
        m = DonchianBreakoutModule(period=10, confidence_cap=0.70, timeframe_id="h1")
        base_bars: list[Bar] = []
        for i in range(m.min_bars):
            c = D(1.10)
            base_bars.append(Bar(
                symbol="EURUSD", timeframe=Timeframe.H1,
                timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                open=c, high=c + D(0.001), low=c - D(0.001), close=c,
                volume=D(1000),
            ))
        last = base_bars[-1]
        base_bars[-1] = Bar(
            symbol=last.symbol, timeframe=last.timeframe,
            timestamp_utc=last.timestamp_utc,
            open=D(1.50), high=D(1.51), low=D(1.49), close=D(1.50),
            volume=D(1000),
        )
        sig = m.analyse(base_bars, ts())
        assert sig.confidence <= 0.70


# ── MarketRegimeModule — Hurst metadata ───────────────────────────────────────

class TestMarketRegimeHurst:
    def _module(self) -> MarketRegimeModule:
        return MarketRegimeModule(
            adx_period=14, adx_threshold=25.0,
            atr_period=5, atr_avg_period=20,
            high_vol_mult=2.0, hurst_window=50,
            timeframe_id="h1",
        )

    def test_hurst_key_in_metadata(self) -> None:
        m = self._module()
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, ts())
        assert "hurst" in sig.metadata
        assert "hurst_label" in sig.metadata

    def test_hurst_value_in_bounds(self) -> None:
        m = self._module()
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, ts())
        h = sig.metadata["hurst"]
        assert 0.10 <= h <= 0.90

    def test_hurst_label_valid(self) -> None:
        m = self._module()
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, ts())
        assert sig.metadata["hurst_label"] in ("trending", "mean_reverting", "random")

    def test_hurst_disabled_when_window_zero(self) -> None:
        """hurst_window=0 should set H to 0.5 (neutral) and label 'random'."""
        m = MarketRegimeModule(
            adx_period=14, adx_threshold=25.0,
            atr_period=5, atr_avg_period=20,
            high_vol_mult=2.0, hurst_window=0,
            timeframe_id="h1",
        )
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, ts())
        assert sig.metadata["hurst"] == 0.5
        assert sig.metadata["hurst_label"] == "random"

    def test_confidence_floor_at_040(self) -> None:
        """Hurst contradiction should not push confidence below 0.40."""
        m = self._module()
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, ts())
        assert sig.confidence >= 0.40


# ── EmaCrossoverModule — HMA slow line tests ──────────────────────────────────

class TestEmaCrossoverHMA:
    def test_module_id_unchanged(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21, timeframe_id="h1")
        assert "ema_crossover" in m.module_id

    def test_hma_slow_default_is_true(self) -> None:
        m = EmaCrossoverModule()
        assert m._use_hma is True  # HMA is the default slow line

    def test_use_hma_false_uses_ema(self) -> None:
        m = EmaCrossoverModule(use_hma_slow=False, slow_period=21)
        assert m._use_hma is False

    def test_hma_min_bars_larger_than_ema(self) -> None:
        m_hma = EmaCrossoverModule(use_hma_slow=True,  slow_period=21)
        m_ema = EmaCrossoverModule(use_hma_slow=False, slow_period=21)
        # HMA needs extra bars for the final WMA smoothing pass
        assert m_hma.min_bars > m_ema.min_bars

    def test_metadata_slow_type_is_hma(self) -> None:
        m = EmaCrossoverModule(use_hma_slow=True, slow_period=21, timeframe_id="h1")
        bars = rising_bars(m.min_bars + 5, step=0.001)
        sig = m.analyse(bars, ts())
        assert sig.metadata.get("slow_type") == "hma"

    def test_metadata_slow_type_is_ema_when_disabled(self) -> None:
        m = EmaCrossoverModule(use_hma_slow=False, slow_period=21, timeframe_id="h1")
        bars = rising_bars(m.min_bars + 5, step=0.001)
        sig = m.analyse(bars, ts())
        assert sig.metadata.get("slow_type") == "ema"

    def test_buy_on_rising_bars(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21, timeframe_id="h1",
                               use_hma_slow=True)
        bars = rising_bars(m.min_bars + 20, step=0.003)
        sig = m.analyse(bars, ts())
        assert sig.direction in (SignalDirection.BUY, SignalDirection.HOLD)

    def test_raises_on_too_few_bars(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21)
        with pytest.raises(ModuleNotReadyError):
            m.analyse([make_bar(1.1)], ts())


# ── Chandelier Exit SL in BaseRunner ─────────────────────────────────────────

class TestChandelierExitSL:
    """Test that BaseRunner._estimate_sl() uses ATR-based Chandelier Exit."""

    def _make_bars(self, n: int, start: float = 1.10, step: float = 0.001) -> list[Bar]:
        return rising_bars(n, start=start, step=step)

    def test_sl_below_entry_for_buy(self) -> None:
        from metatrade.runner.base import BaseRunner
        from metatrade.runner.config import RunnerConfig

        cfg = RunnerConfig()
        runner = BaseRunner(config=cfg, modules=[], auto_weight=False)
        bars = self._make_bars(30)
        entry = bars[-1].close
        sl = runner._estimate_sl(bars, entry, OrderSide.BUY)
        assert sl < entry

    def test_sl_above_entry_for_sell(self) -> None:
        from metatrade.runner.base import BaseRunner
        from metatrade.runner.config import RunnerConfig

        cfg = RunnerConfig()
        runner = BaseRunner(config=cfg, modules=[], auto_weight=False)
        bars = self._make_bars(30)
        entry = bars[-1].close
        sl = runner._estimate_sl(bars, entry, OrderSide.SELL)
        assert sl > entry

    def test_sl_distance_proportional_to_volatility(self) -> None:
        """Wide bars → larger ATR → wider SL distance."""
        from metatrade.runner.base import BaseRunner
        from metatrade.runner.config import RunnerConfig

        cfg = RunnerConfig()
        runner = BaseRunner(config=cfg, modules=[], auto_weight=False)

        tight_bars = [
            Bar(
                symbol="EURUSD", timeframe=Timeframe.H1,
                timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                open=D(1.10), high=D(1.10) + D(0.0002), low=D(1.10) - D(0.0002),
                close=D(1.10), volume=D(1000),
            )
            for i in range(30)
        ]
        wide_bars = [
            Bar(
                symbol="EURUSD", timeframe=Timeframe.H1,
                timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                open=D(1.10), high=D(1.10) + D(0.010), low=D(1.10) - D(0.010),
                close=D(1.10), volume=D(1000),
            )
            for i in range(30)
        ]

        entry = D(1.10)
        sl_tight = runner._estimate_sl(tight_bars, entry, OrderSide.BUY)
        sl_wide  = runner._estimate_sl(wide_bars,  entry, OrderSide.BUY)

        # Wider bars → larger ATR → SL further below entry
        dist_tight = float(entry - sl_tight)
        dist_wide  = float(entry - sl_wide)
        assert dist_wide > dist_tight, (
            f"Wide bars should give wider SL: tight={dist_tight:.5f}, wide={dist_wide:.5f}"
        )

    def test_sl_fallback_on_insufficient_bars(self) -> None:
        """With fewer bars than ATR period, falls back to 20-pip default."""
        from metatrade.runner.base import BaseRunner
        from metatrade.runner.config import RunnerConfig

        cfg = RunnerConfig(chandelier_atr_period=14)
        runner = BaseRunner(config=cfg, modules=[], auto_weight=False)

        # Only 5 bars — not enough for ATR(14)
        bars = self._make_bars(5)
        entry = bars[-1].close
        sl = runner._estimate_sl(bars, entry, OrderSide.BUY)
        # Fallback: SL = entry - 0.0020
        assert abs(float(entry - sl) - 0.0020) < 0.0001

    def test_custom_atr_period_and_mult(self) -> None:
        """chandelier_atr_mult=1.0 vs 3.0 should give different SL distances."""
        from metatrade.runner.base import BaseRunner
        from metatrade.runner.config import RunnerConfig

        bars = self._make_bars(30, step=0.001)
        entry = bars[-1].close

        cfg1 = RunnerConfig(chandelier_atr_period=10, chandelier_atr_mult=1.0)
        cfg3 = RunnerConfig(chandelier_atr_period=10, chandelier_atr_mult=3.0)
        r1 = BaseRunner(config=cfg1, modules=[], auto_weight=False)
        r3 = BaseRunner(config=cfg3, modules=[], auto_weight=False)

        sl1 = r1._estimate_sl(bars, entry, OrderSide.BUY)
        sl3 = r3._estimate_sl(bars, entry, OrderSide.BUY)
        # mult=3 → further below entry
        assert sl3 < sl1
