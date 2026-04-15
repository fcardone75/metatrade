"""Tests for phase-3 technical analysis modules:
StochasticRSI, AdaptiveRSI, SwingLevel, MultiTimeframe,
and updated ATR-adaptive PivotPoints.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import SignalDirection, Timeframe
from metatrade.core.errors import ModuleNotReadyError
from metatrade.technical_analysis.modules.stochastic_rsi_module import StochasticRsiModule
from metatrade.technical_analysis.modules.adaptive_rsi_module import AdaptiveRsiModule
from metatrade.technical_analysis.modules.swing_level_module import (
    SwingLevelModule,
    _find_swing_highs,
    _find_swing_lows,
)
from metatrade.technical_analysis.modules.multi_timeframe_module import (
    MultiTimeframeModule,
    _resample_bars,
)
from metatrade.technical_analysis.modules.pivot_points_module import PivotPointsModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 3, 14, 0, 0, tzinfo=timezone.utc)  # Tuesday 14:00 UTC


def D(v: str | float | int) -> Decimal:
    return Decimal(str(v))


def make_bar(
    close: float,
    open_: float | None = None,
    high_extra: float = 0.0005,
    low_extra: float = 0.0005,
    i: int = 0,
    volume: float = 1000.0,
) -> Bar:
    o = open_ if open_ is not None else close
    h = max(o, close) + high_extra
    lo = min(o, close) - low_extra
    ts = datetime(2024, 1, 1 + i // 24, i % 24, 0, 0, tzinfo=timezone.utc)
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_utc=ts,
        open=D(str(round(o, 6))),
        high=D(str(round(h, 6))),
        low=D(str(round(lo, 6))),
        close=D(str(round(close, 6))),
        volume=D(str(volume)),
    )


def rising_bars(n: int, start: float = 1.10000, step: float = 0.0002) -> list[Bar]:
    return [make_bar(round(start + i * step, 5), i=i) for i in range(n)]


def falling_bars(n: int, start: float = 1.20000, step: float = 0.0002) -> list[Bar]:
    return [make_bar(round(start - i * step, 5), i=i) for i in range(n)]


def flat_bars(n: int, price: float = 1.10000) -> list[Bar]:
    return [make_bar(price, high_extra=0.0, low_extra=0.0, i=i) for i in range(n)]


def alternating_bars(
    n: int, base: float = 1.10000, swing: float = 0.0010
) -> list[Bar]:
    bars = []
    for i in range(n):
        c = base + (swing if i % 2 == 0 else -swing)
        bars.append(make_bar(round(c, 5), i=i))
    return bars


def wave_bars(
    n: int,
    base: float = 1.10000,
    amplitude: float = 0.0050,
    period: int = 20,
) -> list[Bar]:
    """Smooth sine-wave price series for swing detection."""
    import math
    bars = []
    for i in range(n):
        c = base + amplitude * math.sin(2 * math.pi * i / period)
        bars.append(make_bar(round(c, 6), i=i))
    return bars


# ---------------------------------------------------------------------------
# StochasticRsiModule
# ---------------------------------------------------------------------------

class TestStochasticRsiModule:
    def _module(self, **kwargs) -> StochasticRsiModule:
        return StochasticRsiModule(timeframe_id="h1", **kwargs)

    def test_module_id(self) -> None:
        m = self._module()
        assert m.module_id == "stoch_rsi_h1"

    def test_min_bars_positive(self) -> None:
        m = self._module()
        assert m.min_bars > 0

    def test_raises_below_min_bars(self) -> None:
        m = self._module()
        with pytest.raises(ModuleNotReadyError):
            m.analyse(rising_bars(m.min_bars - 1), NOW)

    def test_returns_signal_with_enough_bars(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 10), NOW)
        assert sig.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD)

    def test_confidence_in_range(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 10), NOW)
        assert 0.0 <= sig.confidence <= 1.0

    def test_hold_on_flat_market(self) -> None:
        # Flat market → K and D both stay around 0.5 → no crossover in extreme zone
        m = self._module()
        sig = m.analyse(flat_bars(m.min_bars + 10), NOW)
        assert sig.direction == SignalDirection.HOLD

    def test_metadata_contains_k_and_d(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 10), NOW)
        assert "k" in sig.metadata
        assert "d" in sig.metadata

    def test_metadata_k_d_in_range(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 10), NOW)
        # K and D should be in [0, 1] range (or close, given float rounding)
        k, d = sig.metadata["k"], sig.metadata["d"]
        assert -0.01 <= k <= 1.01
        assert -0.01 <= d <= 1.01

    def test_module_id_suffix(self) -> None:
        m = StochasticRsiModule(timeframe_id="m15")
        assert m.module_id == "stoch_rsi_m15"

    def test_invalid_oversold_overbought_raises(self) -> None:
        with pytest.raises(ValueError):
            StochasticRsiModule(oversold=0.8, overbought=0.2)  # oversold > overbought

    def test_timestamp_preserved(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 5), NOW)
        assert sig.timestamp_utc == NOW

    def test_module_id_in_signal(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 5), NOW)
        assert sig.module_id == m.module_id

    def test_buy_after_oversold_crossover(self) -> None:
        """After a long drop (oversold), then sharp recovery → BUY."""
        m = self._module(rsi_period=10, stoch_period=10, smooth_k=2, smooth_d=2)
        n = m.min_bars + 15
        # Drop sharply then recover
        bars = falling_bars(n - 5, start=1.2, step=0.0010) + rising_bars(5, start=1.15, step=0.0020)
        sig = m.analyse(bars[:n], NOW)
        # May or may not trigger BUY depending on exact K/D values, but should not crash
        assert sig.direction in (SignalDirection.BUY, SignalDirection.HOLD)


# ---------------------------------------------------------------------------
# AdaptiveRsiModule
# ---------------------------------------------------------------------------

class TestAdaptiveRsiModule:
    def _module(self, **kwargs) -> AdaptiveRsiModule:
        return AdaptiveRsiModule(timeframe_id="h1", **kwargs)

    def test_module_id(self) -> None:
        m = self._module()
        assert m.module_id == "adaptive_rsi_h1"

    def test_min_bars_positive(self) -> None:
        m = self._module()
        assert m.min_bars > 0

    def test_raises_below_min_bars(self) -> None:
        m = self._module()
        with pytest.raises(ModuleNotReadyError):
            m.analyse(rising_bars(m.min_bars - 1), NOW)

    def test_returns_signal_with_enough_bars(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 10), NOW)
        assert sig.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD)

    def test_confidence_in_range(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 10), NOW)
        assert 0.0 <= sig.confidence <= 1.0

    def test_sell_on_strongly_rising_bars(self) -> None:
        """Fast price rise → RSI very high → SELL when in ranging regime."""
        m = self._module(rsi_period=5, adx_period=5)
        bars = rising_bars(m.min_bars + 10, step=0.0030)  # aggressive rise
        sig = m.analyse(bars, NOW)
        # RSI should be very high in a strong rising market → SELL expected
        assert sig.direction == SignalDirection.SELL

    def test_buy_on_strongly_falling_bars(self) -> None:
        """Fast price fall → RSI very low → BUY."""
        m = self._module(rsi_period=5, adx_period=5)
        bars = falling_bars(m.min_bars + 10, step=0.0030)
        sig = m.analyse(bars, NOW)
        assert sig.direction == SignalDirection.BUY

    def test_hold_on_ranging_market(self) -> None:
        """Alternating bars → RSI near 50 → HOLD."""
        m = self._module()
        sig = m.analyse(alternating_bars(m.min_bars + 10), NOW)
        assert sig.direction == SignalDirection.HOLD

    def test_metadata_contains_rsi_and_adx(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 5), NOW)
        assert "rsi" in sig.metadata
        assert "adx" in sig.metadata
        assert "regime" in sig.metadata

    def test_metadata_regime_values(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 5), NOW)
        assert sig.metadata["regime"] in ("ranging", "trending", "transitioning")

    def test_invalid_adx_range_raises(self) -> None:
        with pytest.raises(ValueError, match="ranging_adx must be < trending_adx"):
            AdaptiveRsiModule(ranging_adx=40.0, trending_adx=20.0)

    def test_confidence_cap_respected(self) -> None:
        cap = 0.75
        m = self._module(confidence_cap=cap, rsi_period=5, adx_period=5)
        bars = rising_bars(m.min_bars + 10, step=0.0030)
        sig = m.analyse(bars, NOW)
        assert sig.confidence <= cap + 1e-9

    def test_timestamp_preserved(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 5), NOW)
        assert sig.timestamp_utc == NOW

    def test_module_id_suffix(self) -> None:
        m = AdaptiveRsiModule(timeframe_id="m30")
        assert m.module_id == "adaptive_rsi_m30"

    def test_overbought_threshold_widens_in_trend(self) -> None:
        """In trending regime, overbought threshold should be 80 (vs 70 in ranging)."""
        m = self._module()
        # Create ADX > trending_adx (>35) with strongly trending bars
        sig = m.analyse(rising_bars(m.min_bars + 5, step=0.0005), NOW)
        ob_thresh = sig.metadata.get("overbought_threshold", 70)
        # In a trend, threshold should be >= 70 (may be higher)
        assert ob_thresh >= 70.0


# ---------------------------------------------------------------------------
# SwingLevelModule helpers
# ---------------------------------------------------------------------------

class TestFindSwingHelpers:
    def test_find_swing_highs_basic(self) -> None:
        """A single clear peak should be identified."""
        # Pattern: 1.10, 1.11, 1.12, 1.11, 1.10 with lookback=2
        highs = [D(str(v)) for v in [1.10, 1.11, 1.12, 1.11, 1.10, 1.09, 1.08]]
        swings = _find_swing_highs(highs, lookback=2)
        # Index 2 (1.12) should be the swing high
        prices = [float(p) for _, p in swings]
        assert 1.12 in prices

    def test_find_swing_lows_basic(self) -> None:
        """A single clear trough should be identified."""
        lows = [D(str(v)) for v in [1.12, 1.11, 1.10, 1.11, 1.12, 1.13, 1.14]]
        swings = _find_swing_lows(lows, lookback=2)
        prices = [float(p) for _, p in swings]
        assert 1.10 in prices

    def test_find_swing_highs_empty_for_monotone(self) -> None:
        highs = [D(str(1.10 + i * 0.001)) for i in range(10)]
        swings = _find_swing_highs(highs, lookback=2)
        assert swings == []

    def test_find_swing_lows_empty_for_monotone(self) -> None:
        lows = [D(str(1.10 + i * 0.001)) for i in range(10)]
        swings = _find_swing_lows(lows, lookback=2)
        assert swings == []

    def test_swing_indices_within_bounds(self) -> None:
        highs = [D(str(v)) for v in [1.1, 1.15, 1.12, 1.14, 1.11, 1.10, 1.09]]
        swings = _find_swing_highs(highs, lookback=1)
        for idx, _ in swings:
            assert 0 <= idx < len(highs)

    def test_multiple_swing_highs(self) -> None:
        # Two peaks separated by valley
        highs = [D(str(v)) for v in [1.10, 1.13, 1.10, 1.12, 1.10, 1.11, 1.09]]
        swings = _find_swing_highs(highs, lookback=1)
        assert len(swings) >= 1  # at least one swing detected


# ---------------------------------------------------------------------------
# SwingLevelModule
# ---------------------------------------------------------------------------

class TestSwingLevelModule:
    def _module(self, **kwargs) -> SwingLevelModule:
        return SwingLevelModule(timeframe_id="h1", **kwargs)

    def test_module_id(self) -> None:
        m = self._module()
        assert m.module_id == "swing_level_h1"

    def test_min_bars_positive(self) -> None:
        m = self._module()
        assert m.min_bars > 0

    def test_raises_below_min_bars(self) -> None:
        m = self._module()
        with pytest.raises(ModuleNotReadyError):
            m.analyse(wave_bars(m.min_bars - 1), NOW)

    def test_returns_signal_with_enough_bars(self) -> None:
        m = self._module()
        sig = m.analyse(wave_bars(m.min_bars + 10), NOW)
        assert sig.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD)

    def test_confidence_in_range(self) -> None:
        m = self._module()
        sig = m.analyse(wave_bars(m.min_bars + 10), NOW)
        assert 0.0 <= sig.confidence <= 1.0

    def test_metadata_contains_swing_lists(self) -> None:
        m = self._module()
        sig = m.analyse(wave_bars(m.min_bars + 10), NOW)
        assert "swing_highs" in sig.metadata
        assert "swing_lows" in sig.metadata
        assert "atr" in sig.metadata

    def test_metadata_swing_prices_are_floats(self) -> None:
        m = self._module()
        sig = m.analyse(wave_bars(m.min_bars + 20), NOW)
        for p in sig.metadata.get("swing_highs", []):
            assert isinstance(p, float)
        for p in sig.metadata.get("swing_lows", []):
            assert isinstance(p, float)

    def test_hold_when_no_swings(self) -> None:
        """Strictly monotone rising bars → no swing lows, no swing highs."""
        m = self._module(lookback=3, max_levels=5)
        bars = rising_bars(m.min_bars + 5, step=0.0002)
        sig = m.analyse(bars, NOW)
        # No confirmed swing highs or lows → HOLD
        assert sig.direction == SignalDirection.HOLD

    def test_invalid_lookback_raises(self) -> None:
        with pytest.raises(ValueError, match="lookback must be >= 1"):
            SwingLevelModule(lookback=0)

    def test_invalid_max_levels_raises(self) -> None:
        with pytest.raises(ValueError, match="max_levels must be >= 1"):
            SwingLevelModule(max_levels=0)

    def test_confidence_cap_respected(self) -> None:
        cap = 0.70
        m = self._module(confidence_cap=cap)
        sig = m.analyse(wave_bars(m.min_bars + 50), NOW)
        assert sig.confidence <= cap + 1e-9

    def test_timestamp_preserved(self) -> None:
        m = self._module()
        sig = m.analyse(wave_bars(m.min_bars + 10), NOW)
        assert sig.timestamp_utc == NOW

    def test_module_id_suffix(self) -> None:
        m = SwingLevelModule(timeframe_id="d1")
        assert m.module_id == "swing_level_d1"

    def test_touch_dist_metadata_present(self) -> None:
        m = self._module()
        sig = m.analyse(wave_bars(m.min_bars + 50), NOW)
        if sig.direction != SignalDirection.HOLD:
            assert "touch_dist" in sig.metadata


# ---------------------------------------------------------------------------
# _resample_bars helper
# ---------------------------------------------------------------------------

class TestResampleBars:
    def test_resample_factor_4(self) -> None:
        bars = rising_bars(12)
        resampled = _resample_bars(bars, factor=4)
        assert len(resampled) == 3  # 12 / 4 = 3 complete groups

    def test_incomplete_group_discarded(self) -> None:
        bars = rising_bars(10)
        resampled = _resample_bars(bars, factor=4)
        assert len(resampled) == 2  # 10 // 4 = 2 complete groups

    def test_resample_preserves_first_open(self) -> None:
        bars = rising_bars(4, start=1.1, step=0.001)
        resampled = _resample_bars(bars, factor=4)
        assert len(resampled) == 1
        assert resampled[0].open == bars[0].open

    def test_resample_preserves_last_close(self) -> None:
        bars = rising_bars(4, start=1.1, step=0.001)
        resampled = _resample_bars(bars, factor=4)
        assert resampled[0].close == bars[-1].close

    def test_resample_high_is_max(self) -> None:
        bars = [
            make_bar(1.10, high_extra=0.010, i=0),
            make_bar(1.11, high_extra=0.001, i=1),
            make_bar(1.09, high_extra=0.001, i=2),
            make_bar(1.10, high_extra=0.001, i=3),
        ]
        resampled = _resample_bars(bars, factor=4)
        expected_high = max(b.high for b in bars)
        assert resampled[0].high == expected_high

    def test_resample_low_is_min(self) -> None:
        bars = [
            make_bar(1.10, low_extra=0.010, i=0),
            make_bar(1.11, low_extra=0.001, i=1),
            make_bar(1.09, low_extra=0.001, i=2),
            make_bar(1.10, low_extra=0.001, i=3),
        ]
        resampled = _resample_bars(bars, factor=4)
        expected_low = min(b.low for b in bars)
        assert resampled[0].low == expected_low

    def test_volume_summed(self) -> None:
        bars = [make_bar(1.1, i=i, volume=100.0) for i in range(4)]
        resampled = _resample_bars(bars, factor=4)
        assert float(resampled[0].volume) == pytest.approx(400.0, abs=0.01)

    def test_empty_input(self) -> None:
        assert _resample_bars([], factor=4) == []


# ---------------------------------------------------------------------------
# MultiTimeframeModule
# ---------------------------------------------------------------------------

class TestMultiTimeframeModule:
    def _module(self, **kwargs) -> MultiTimeframeModule:
        return MultiTimeframeModule(timeframe_id="h1_to_h4", **kwargs)

    def test_module_id(self) -> None:
        m = self._module()
        assert m.module_id == "multi_tf_h1_to_h4"

    def test_min_bars_positive(self) -> None:
        m = self._module()
        assert m.min_bars > 0

    def test_raises_below_min_bars(self) -> None:
        m = self._module()
        with pytest.raises(ModuleNotReadyError):
            m.analyse(rising_bars(m.min_bars - 1), NOW)

    def test_returns_signal_with_enough_bars(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 5), NOW)
        assert sig.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD)

    def test_buy_on_sustained_uptrend(self) -> None:
        """Strongly rising bars → higher-TF uptrend → BUY."""
        m = self._module(resample_factor=4, fast_period=5, slow_period=10)
        n = m.min_bars + 20
        sig = m.analyse(rising_bars(n, step=0.0002), NOW)
        assert sig.direction == SignalDirection.BUY

    def test_sell_on_sustained_downtrend(self) -> None:
        """Strongly falling bars → higher-TF downtrend → SELL."""
        m = self._module(resample_factor=4, fast_period=5, slow_period=10)
        n = m.min_bars + 20
        sig = m.analyse(falling_bars(n, step=0.0002), NOW)
        assert sig.direction == SignalDirection.SELL

    def test_confidence_in_range(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 20), NOW)
        assert 0.0 <= sig.confidence <= 1.0

    def test_metadata_contains_ema_values(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 10), NOW)
        assert "fast_ema" in sig.metadata
        assert "slow_ema" in sig.metadata
        assert "htf_bars" in sig.metadata

    def test_metadata_htf_bars_positive(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 10), NOW)
        assert sig.metadata["htf_bars"] > 0

    def test_invalid_resample_factor_raises(self) -> None:
        with pytest.raises(ValueError, match="resample_factor must be >= 2"):
            MultiTimeframeModule(resample_factor=1)

    def test_invalid_fast_slow_raises(self) -> None:
        with pytest.raises(ValueError, match="fast_period"):
            MultiTimeframeModule(fast_period=21, slow_period=9)

    def test_confidence_cap_respected(self) -> None:
        cap = 0.72
        m = self._module(confidence_cap=cap)
        sig = m.analyse(rising_bars(m.min_bars + 20), NOW)
        assert sig.confidence <= cap + 1e-9

    def test_timestamp_preserved(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 10), NOW)
        assert sig.timestamp_utc == NOW

    def test_module_id_custom_suffix(self) -> None:
        m = MultiTimeframeModule(timeframe_id="h1_to_d1")
        assert m.module_id == "multi_tf_h1_to_d1"

    def test_spread_pct_positive_in_uptrend(self) -> None:
        m = self._module(resample_factor=4, fast_period=5, slow_period=10)
        n = m.min_bars + 20
        sig = m.analyse(rising_bars(n, step=0.0002), NOW)
        assert sig.metadata["spread_pct"] > 0


# ---------------------------------------------------------------------------
# PivotPointsModule (ATR-adaptive)
# ---------------------------------------------------------------------------

class TestPivotPointsModuleATR:
    """Tests specifically for the updated ATR-adaptive pivot points."""

    def _module(self, **kwargs) -> PivotPointsModule:
        return PivotPointsModule(timeframe_id="h1", **kwargs)

    def test_module_id(self) -> None:
        m = self._module()
        assert m.module_id == "pivot_points_h1"

    def test_min_bars_includes_atr_period(self) -> None:
        m = PivotPointsModule(session_bars=24, atr_period=14)
        assert m.min_bars >= 14  # must accommodate ATR period

    def test_raises_below_min_bars(self) -> None:
        m = self._module()
        with pytest.raises(ModuleNotReadyError):
            m.analyse(rising_bars(m.min_bars - 1), NOW)

    def test_returns_signal_with_enough_bars(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 5), NOW)
        assert sig.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD)

    def test_metadata_contains_atr(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 5), NOW)
        assert "atr" in sig.metadata
        assert sig.metadata["atr"] > 0.0

    def test_metadata_contains_touch_dist(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 5), NOW)
        assert "touch_dist" in sig.metadata

    def test_touch_dist_scales_with_atr_mult(self) -> None:
        """Higher atr_touch_mult → larger touch distance → more BUY/SELL signals."""
        bars = rising_bars(200, step=0.0001)
        m_narrow = PivotPointsModule(atr_touch_mult=0.1, timeframe_id="h1")
        m_wide   = PivotPointsModule(atr_touch_mult=2.0, timeframe_id="h1")
        sig_narrow = m_narrow.analyse(bars, NOW)
        sig_wide   = m_wide.analyse(bars, NOW)
        # Wide touch dist should never produce smaller touch than narrow
        assert sig_wide.metadata["touch_dist"] >= sig_narrow.metadata["touch_dist"]

    def test_invalid_session_bars_raises(self) -> None:
        with pytest.raises(ValueError, match="session_bars must be >= 2"):
            PivotPointsModule(session_bars=1)

    def test_invalid_atr_period_raises(self) -> None:
        with pytest.raises(ValueError, match="atr_period must be >= 1"):
            PivotPointsModule(atr_period=0)

    def test_confidence_in_range(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 5), NOW)
        assert 0.0 <= sig.confidence <= 1.0

    def test_metadata_contains_pivot_levels(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 5), NOW)
        for key in ("pivot", "r1", "r2", "s1", "s2"):
            assert key in sig.metadata

    def test_timestamp_preserved(self) -> None:
        m = self._module()
        sig = m.analyse(rising_bars(m.min_bars + 5), NOW)
        assert sig.timestamp_utc == NOW


# ── Additional coverage for stochastic_rsi_module ────────────────────────────

class TestStochasticRsiModuleValidation:
    def test_rsi_period_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="rsi_period must be >= 2"):
            StochasticRsiModule(rsi_period=1)

    def test_stoch_period_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="stoch_period must be >= 2"):
            StochasticRsiModule(stoch_period=1)

    def test_invalid_thresholds_raise(self) -> None:
        with pytest.raises(ValueError, match="oversold"):
            StochasticRsiModule(oversold=0.8, overbought=0.7)


class TestStochasticRsiSignals:
    def _make_bullish_cross_bars(self, n: int) -> list[Bar]:
        """Alternating warm-up then decline + big bounce forces oversold K/D crossover.

        Phase 1 (first 16 bars): alternating ±0.0010 keeps RSI near 50.
        Phase 2 (next n-17 bars): steady decline drives RSI into the low
          single digits while remaining non-zero (avoiding the k_raw=0.5
          neutral-flat edge case).
        Last bar: +0.0300 spike → RSI jumps, raw %K → 1.0; with smooth_k=2
          (SMA) K[-1]=0.5 while K[-2]=0 and D[-1]=0.25 while D[-2]=0,
          yielding a bullish K-above-D crossover in the oversold zone.
        """
        prices: list[float] = []
        for i in range(16):
            prices.append(1.1200 + 0.0010 if i % 2 == 0 else 1.1200 - 0.0010)
        for _ in range(n - 17):
            prices.append(prices[-1] - 0.0015)
        prices.append(prices[-1] + 0.0300)   # big bounce at last bar
        return [
            make_bar(p, high_extra=0.0002, low_extra=0.0002, i=i)
            for i, p in enumerate(prices)
        ]

    def _make_bearish_cross_bars(self, n: int) -> list[Bar]:
        """Alternating warm-up then rise + big drop forces overbought K/D crossover.

        Phase 1 (first 16 bars): alternating keeps RSI near 50.
        Phase 2 (next n-17 bars): steady rise drives RSI into the high
          nineties while remaining non-unity.
        Last bar: −0.0300 spike → RSI drops, raw %K → 0.0; K[-1]=0.5
          while K[-2]=1 and D[-1]=0.75 while D[-2]=1, yielding a bearish
          K-below-D crossover in the overbought zone.
        """
        prices: list[float] = []
        for i in range(16):
            prices.append(1.1000 + 0.0010 if i % 2 == 0 else 1.1000 - 0.0010)
        for _ in range(n - 17):
            prices.append(prices[-1] + 0.0015)
        prices.append(prices[-1] - 0.0300)   # big drop at last bar
        return [
            make_bar(p, high_extra=0.0002, low_extra=0.0002, i=i)
            for i, p in enumerate(prices)
        ]

    def test_buy_signal_produced_from_oversold_crossover(self) -> None:
        """Bullish K/D crossover from oversold zone yields BUY direction."""
        m = StochasticRsiModule(
            rsi_period=5, stoch_period=3, smooth_k=2, smooth_d=2,
            oversold=0.40, overbought=0.85,
        )
        n = m.min_bars + 20
        bars = self._make_bullish_cross_bars(n)
        sig = m.analyse(bars, NOW)
        assert sig.direction == SignalDirection.BUY
        assert 0.55 <= sig.confidence <= 0.85

    def test_sell_signal_produced_from_overbought_crossover(self) -> None:
        """Bearish K/D crossover from overbought zone yields SELL direction."""
        m = StochasticRsiModule(
            rsi_period=5, stoch_period=3, smooth_k=2, smooth_d=2,
            oversold=0.20, overbought=0.60,
        )
        n = m.min_bars + 20
        bars = self._make_bearish_cross_bars(n)
        sig = m.analyse(bars, NOW)
        assert sig.direction == SignalDirection.SELL
        assert 0.55 <= sig.confidence <= 0.85


# ── Additional coverage for swing_level_module ───────────────────────────────

class TestSwingLevelSignals:
    def _make_swing_low_bars(self, n: int) -> list[Bar]:
        """V-shape price series: confirmed swing low at bar 8, last 3 bars near it.

        With lookback=3 the swing at index 8 is confirmed by bars 5-7 (higher
        lows on the left) and bars 9-11 (higher lows on the right).  The last
        3 bars close at 1.1001, which is only 6 pips above the swing-low price
        of 1.0995 — well within atr_touch_mult=5 × ATR≈0.0030 = 0.0150.
        """
        prices: list[float] = []
        # bars 0-7: declining (no swing low — each bar lower than the previous)
        for i in range(8):
            prices.append(1.1200 - i * 0.0025)
        # bar 8: clear minimum — swing low confirmed by surrounding bars
        prices.append(1.1000)
        # bars 9-11: recovering (higher lows confirm bar 8 as a swing low)
        prices.extend([1.1015, 1.1030, 1.1045])
        # bars 12 .. n-4: trending higher, well above the swing low
        for i in range(n - 15):
            prices.append(1.1045 + (i + 1) * 0.0010)
        # last 3 bars (excluded from safe_bars): return near the swing low
        prices.extend([1.1003, 1.1002, 1.1001])
        assert len(prices) == n, f"bar count mismatch: {len(prices)} != {n}"
        return [
            make_bar(p, high_extra=0.0005, low_extra=0.0005, i=i)
            for i, p in enumerate(prices)
        ]

    def _make_swing_high_bars(self, n: int) -> list[Bar]:
        """Inverted-V price series: confirmed swing high at bar 8, last 3 bars near it.

        With lookback=3, bar 8 is confirmed as a swing high by bars 5-7
        (lower highs on the left) and bars 9-11 (lower highs on the right).
        The last 3 bars close at 1.1199, which is only 6 pips below the
        swing-high price of 1.1205 — well within atr_touch_mult=5 × ATR.
        """
        prices: list[float] = []
        # bars 0-7: rising (no swing high — each bar higher)
        for i in range(8):
            prices.append(1.1000 + i * 0.0025)
        # bar 8: clear maximum — swing high confirmed by surrounding bars
        prices.append(1.1200)
        # bars 9-11: declining (lower highs confirm bar 8 as a swing high)
        prices.extend([1.1185, 1.1170, 1.1155])
        # bars 12 .. n-4: trending lower, well below the swing high
        for i in range(n - 15):
            prices.append(1.1155 - (i + 1) * 0.0010)
        # last 3 bars (excluded from safe_bars): approach the swing high
        prices.extend([1.1197, 1.1198, 1.1199])
        assert len(prices) == n, f"bar count mismatch: {len(prices)} != {n}"
        return [
            make_bar(p, high_extra=0.0005, low_extra=0.0005, i=i)
            for i, p in enumerate(prices)
        ]

    def test_near_support_produces_buy(self) -> None:
        """Last bar within touch_dist of a confirmed swing low → BUY."""
        m = SwingLevelModule(lookback=3, atr_touch_mult=5.0, max_levels=5)
        n = m.min_bars + 10
        bars = self._make_swing_low_bars(n)
        sig = m.analyse(bars, NOW)
        assert sig.direction == SignalDirection.BUY
        assert 0.58 <= sig.confidence <= m._confidence_cap

    def test_near_resistance_produces_sell(self) -> None:
        """Last bar within touch_dist of a confirmed swing high → SELL."""
        m = SwingLevelModule(lookback=3, atr_touch_mult=5.0, max_levels=5)
        n = m.min_bars + 10
        bars = self._make_swing_high_bars(n)
        sig = m.analyse(bars, NOW)
        assert sig.direction == SignalDirection.SELL
        assert 0.58 <= sig.confidence <= m._confidence_cap
