"""Tests for the 5 new technical analysis modules:
Bollinger Bands, ADX, Pivot Points, Volatility Regime, Seasonality.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import SignalDirection, Timeframe
from metatrade.core.errors import ModuleNotReadyError
from metatrade.technical_analysis.modules.adx_module import AdxModule
from metatrade.technical_analysis.modules.bollinger_module import BollingerBandsModule
from metatrade.technical_analysis.modules.pivot_points_module import (
    PivotPointsModule,
    _compute_pivot_levels,
)
from metatrade.technical_analysis.modules.seasonality_module import (
    SeasonalityModule,
    _classify_session,
    _SessionQuality,
)
from metatrade.technical_analysis.modules.volatility_regime_module import (
    VolatilityRegimeModule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def D(v: str | float | int) -> Decimal:
    return Decimal(str(v))


def make_bar(
    close: Decimal,
    open_: Decimal | None = None,
    high_extra: Decimal = D("0.0005"),
    low_extra: Decimal = D("0.0005"),
    i: int = 0,
) -> Bar:
    o = open_ if open_ is not None else close
    h = max(o, close) + high_extra
    lo = min(o, close) - low_extra
    ts = datetime(2024, 1, 1 + i // 24, i % 24, 0, 0, tzinfo=UTC)
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_utc=ts,
        open=o,
        high=h,
        low=lo,
        close=close,
        volume=D("1000"),
    )


def rising_bars(n: int, start: float = 1.10000, step: float = 0.0002) -> list[Bar]:
    return [make_bar(D(round(start + i * step, 5)), i=i) for i in range(n)]


def falling_bars(n: int, start: float = 1.20000, step: float = 0.0002) -> list[Bar]:
    return [make_bar(D(round(start - i * step, 5)), i=i) for i in range(n)]


def flat_bars(n: int, price: float = 1.10000) -> list[Bar]:
    return [make_bar(D(str(price)), i=i) for i in range(n)]


def alternating_bars(n: int, base: float = 1.10000, swing: float = 0.0010) -> list[Bar]:
    """Alternating up/down prices (choppy, ranging market)."""
    bars = []
    for i in range(n):
        c = base + (swing if i % 2 == 0 else -swing)
        bars.append(make_bar(D(round(c, 5)), i=i))
    return bars


NOW_LONDON_OVERLAP = datetime(2024, 6, 4, 14, 0, 0, tzinfo=UTC)  # Tue 14:00 UTC
NOW_ASIAN = datetime(2024, 6, 4, 3, 0, 0, tzinfo=UTC)            # Tue 03:00 UTC
NOW_FRIDAY_PM = datetime(2024, 6, 7, 16, 0, 0, tzinfo=UTC)       # Fri 16:00 UTC


# ---------------------------------------------------------------------------
# BollingerBandsModule
# ---------------------------------------------------------------------------

class TestBollingerBandsModule:
    def test_module_id(self) -> None:
        m = BollingerBandsModule(timeframe_id="h1")
        assert m.module_id == "bollinger_h1"

    def test_min_bars(self) -> None:
        m = BollingerBandsModule(period=20)
        assert m.min_bars == 21  # period + 1

    def test_raises_on_too_few_bars(self) -> None:
        m = BollingerBandsModule(period=20)
        with pytest.raises(ModuleNotReadyError):
            m.analyse(flat_bars(10), NOW_LONDON_OVERLAP)

    def test_raises_on_period_less_than_2(self) -> None:
        with pytest.raises(ValueError):
            BollingerBandsModule(period=1)

    def test_hold_on_flat_price_within_bands(self) -> None:
        # Flat price → never touches band extremes → HOLD
        m = BollingerBandsModule(period=20)
        bars = flat_bars(30)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.HOLD

    def test_buy_on_oversold_bounce(self) -> None:
        """Price crashes far below the lower band then recovers."""
        m = BollingerBandsModule(period=10, num_std=2.0)
        bars = flat_bars(20, price=1.1000)
        # Push the second-to-last bar far below the band
        crash_close = D("1.0500")  # well below ~1.0997 lower band
        bars[-2] = make_bar(crash_close, i=18)
        # Last bar recovers toward previous flat level
        bars[-1] = make_bar(D("1.0999"), i=19)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.BUY

    def test_sell_on_overbought_rejection(self) -> None:
        """Price spikes far above the upper band then drops back."""
        m = BollingerBandsModule(period=10, num_std=2.0)
        bars = flat_bars(20, price=1.1000)
        spike_close = D("1.1500")  # well above ~1.1003 upper band
        bars[-2] = make_bar(spike_close, i=18)
        bars[-1] = make_bar(D("1.1001"), i=19)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.SELL

    def test_confidence_in_valid_range(self) -> None:
        m = BollingerBandsModule(period=10)
        bars = flat_bars(30, price=1.1000)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert 0.0 <= sig.confidence <= 1.0

    def test_metadata_keys_present(self) -> None:
        m = BollingerBandsModule(period=10)
        bars = flat_bars(30)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert "upper" in sig.metadata
        assert "middle" in sig.metadata
        assert "lower" in sig.metadata
        assert "percent_b" in sig.metadata
        assert "band_width" in sig.metadata

    def test_metadata_bands_ordered(self) -> None:
        m = BollingerBandsModule(period=10)
        bars = rising_bars(30)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.metadata["upper"] >= sig.metadata["lower"]

    def test_confidence_cap_respected(self) -> None:
        m = BollingerBandsModule(period=10, confidence_cap=0.70)
        bars = flat_bars(20, price=1.1000)
        bars[-2] = make_bar(D("1.0400"), i=18)
        bars[-1] = make_bar(D("1.0999"), i=19)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.confidence <= 0.70

    def test_timestamp_propagated(self) -> None:
        m = BollingerBandsModule(period=10)
        bars = flat_bars(30)
        ts = datetime(2024, 3, 15, 9, 0, 0, tzinfo=UTC)
        sig = m.analyse(bars, ts)
        assert sig.timestamp_utc == ts


# ---------------------------------------------------------------------------
# AdxModule
# ---------------------------------------------------------------------------

class TestAdxModule:
    def test_module_id(self) -> None:
        m = AdxModule(timeframe_id="h4")
        assert m.module_id == "adx_h4"

    def test_min_bars(self) -> None:
        m = AdxModule(period=14)
        assert m.min_bars == 14 * 2 + 5  # 33

    def test_raises_on_too_few_bars(self) -> None:
        m = AdxModule(period=14)
        with pytest.raises(ModuleNotReadyError):
            m.analyse(rising_bars(20), NOW_LONDON_OVERLAP)

    def test_hold_on_ranging_market(self) -> None:
        m = AdxModule(period=14, adx_threshold=25.0)
        bars = alternating_bars(50)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.HOLD

    def test_buy_on_strong_uptrend(self) -> None:
        m = AdxModule(period=14, adx_threshold=20.0)
        bars = rising_bars(80, step=0.0020)  # strong uptrend
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.BUY

    def test_sell_on_strong_downtrend(self) -> None:
        m = AdxModule(period=14, adx_threshold=20.0)
        bars = falling_bars(80, step=0.0020)  # strong downtrend
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.SELL

    def test_confidence_scales_with_adx(self) -> None:
        m = AdxModule(period=14, adx_threshold=20.0)
        # Strong trend → higher ADX → higher confidence
        bars_strong = rising_bars(80, step=0.0030)
        bars_weak = rising_bars(80, step=0.0005)
        sig_strong = m.analyse(bars_strong, NOW_LONDON_OVERLAP)
        sig_weak = m.analyse(bars_weak, NOW_LONDON_OVERLAP)
        if (
            sig_strong.direction != SignalDirection.HOLD
            and sig_weak.direction != SignalDirection.HOLD
        ):
            assert sig_strong.confidence >= sig_weak.confidence

    def test_confidence_cap_respected(self) -> None:
        m = AdxModule(period=14, adx_threshold=10.0, confidence_cap=0.80)
        bars = rising_bars(80, step=0.0020)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.confidence <= 0.80

    def test_metadata_keys_present(self) -> None:
        m = AdxModule(period=14)
        bars = rising_bars(50)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert "adx" in sig.metadata
        assert "plus_di" in sig.metadata
        assert "minus_di" in sig.metadata
        assert "threshold" in sig.metadata

    def test_metadata_adx_non_negative(self) -> None:
        m = AdxModule(period=14)
        bars = rising_bars(50)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.metadata["adx"] >= 0

    def test_raises_on_period_less_than_2(self) -> None:
        with pytest.raises(ValueError):
            AdxModule(period=1)


# ---------------------------------------------------------------------------
# PivotPointsModule
# ---------------------------------------------------------------------------

class TestPivotLevels:
    def test_classic_calculation(self) -> None:
        levels = _compute_pivot_levels(D("1.1200"), D("1.1000"), D("1.1100"))
        expected_p = (D("1.1200") + D("1.1000") + D("1.1100")) / D("3")
        assert abs(levels.pivot - expected_p) < D("0.00001")

    def test_r1_above_pivot(self) -> None:
        levels = _compute_pivot_levels(D("1.1200"), D("1.1000"), D("1.1100"))
        assert levels.r1 > levels.pivot

    def test_r2_above_r1(self) -> None:
        levels = _compute_pivot_levels(D("1.1200"), D("1.1000"), D("1.1100"))
        assert levels.r2 > levels.r1

    def test_s1_below_pivot(self) -> None:
        levels = _compute_pivot_levels(D("1.1200"), D("1.1000"), D("1.1100"))
        assert levels.s1 < levels.pivot

    def test_s2_below_s1(self) -> None:
        levels = _compute_pivot_levels(D("1.1200"), D("1.1000"), D("1.1100"))
        assert levels.s2 < levels.s1


class TestPivotPointsModule:
    def _make_pivot_bars(
        self,
        n_total: int = 60,
        session_bars: int = 24,
        close_near: Decimal | None = None,
    ) -> list[Bar]:
        bars = flat_bars(n_total, price=1.1050)
        if close_near is not None:
            bars[-1] = make_bar(close_near, i=n_total - 1)
        return bars

    def test_module_id(self) -> None:
        m = PivotPointsModule(timeframe_id="h1")
        assert m.module_id == "pivot_points_h1"

    def test_min_bars(self) -> None:
        m = PivotPointsModule(session_bars=24)
        assert m.min_bars == 24 * 2 + 2  # 50

    def test_raises_on_too_few_bars(self) -> None:
        m = PivotPointsModule(session_bars=24)
        with pytest.raises(ModuleNotReadyError):
            m.analyse(flat_bars(30), NOW_LONDON_OVERLAP)

    def test_raises_on_session_bars_less_than_2(self) -> None:
        with pytest.raises(ValueError):
            PivotPointsModule(session_bars=1)

    def test_hold_when_price_not_near_pivot(self) -> None:
        # Use very small atr_touch_mult so price is too far from any level
        m = PivotPointsModule(session_bars=5, atr_touch_mult=0.001)
        # Price at 1.1000, pivot levels around 1.1050 → too far
        bars = flat_bars(25, price=1.1000)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.HOLD

    def test_buy_near_support_s1(self) -> None:
        """If price is exactly at S1, module should emit BUY."""
        # Use atr_period=5 so min_bars = max(5*2+2, 5+5+2) = max(12, 12) = 12
        m = PivotPointsModule(session_bars=5, atr_period=5, atr_touch_mult=5.0)
        # Build bars so the "previous session" has known H/L/C
        # prev session H=1.12, L=1.10, C=1.11 → S1 = 2*P - H
        # P = (1.12+1.10+1.11)/3 = 1.11; S1 = 2*1.11 - 1.12 = 1.10
        base_bars = []
        for i in range(5):
            c = D("1.11")
            h = D("1.12") if i == 2 else c + D("0.001")
            lo = D("1.10") if i == 3 else c - D("0.001")
            ts = datetime(2024, 1, 1, i, 0, 0, tzinfo=UTC)
            base_bars.append(Bar(
                symbol="EURUSD", timeframe=Timeframe.H1,
                timestamp_utc=ts, open=c, high=h, low=lo, close=c, volume=D("100"),
            ))
        # Add a "current session" of 10 more bars with last close near S1 ≈ 1.10
        for i in range(5, 15):
            ts = datetime(2024, 1, 1, i, 0, 0, tzinfo=UTC)
            close_val = D("1.1003") if i == 14 else D("1.1050")
            base_bars.append(Bar(
                symbol="EURUSD", timeframe=Timeframe.H1,
                timestamp_utc=ts, open=close_val,
                high=close_val + D("0.001"), low=close_val - D("0.001"),
                close=close_val, volume=D("100"),
            ))
        sig = m.analyse(base_bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.BUY

    def test_metadata_keys_present(self) -> None:
        m = PivotPointsModule(session_bars=5)
        bars = flat_bars(25)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        for key in ("pivot", "r1", "r2", "s1", "s2", "session_high", "session_low"):
            assert key in sig.metadata

    def test_pivot_levels_ordered_in_metadata(self) -> None:
        m = PivotPointsModule(session_bars=5)
        bars = rising_bars(25)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        md = sig.metadata
        assert md["r2"] >= md["r1"] >= md["pivot"] >= md["s1"] >= md["s2"]

    def test_confidence_in_valid_range(self) -> None:
        m = PivotPointsModule(session_bars=5)
        bars = flat_bars(25)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert 0.0 <= sig.confidence <= 1.0


# ---------------------------------------------------------------------------
# VolatilityRegimeModule
# ---------------------------------------------------------------------------

class TestVolatilityRegimeModule:
    def test_module_id(self) -> None:
        m = VolatilityRegimeModule(timeframe_id="h1")
        assert m.module_id == "volatility_regime_h1"

    def test_min_bars(self) -> None:
        m = VolatilityRegimeModule(atr_period=14, avg_period=50, trend_period=50)
        assert m.min_bars >= 64  # max(50, 50) + 14 + 5

    def test_raises_on_too_few_bars(self) -> None:
        m = VolatilityRegimeModule()
        with pytest.raises(ModuleNotReadyError):
            m.analyse(rising_bars(30), NOW_LONDON_OVERLAP)

    def test_raises_on_invalid_avg_period(self) -> None:
        with pytest.raises(ValueError, match="avg_period"):
            VolatilityRegimeModule(atr_period=14, avg_period=10)

    def test_buy_signal_when_price_above_trend_ema(self) -> None:
        m = VolatilityRegimeModule(
            atr_period=5, avg_period=10, trend_period=10,
            low_atr_mult=0.3, high_atr_mult=3.0,
        )
        # Rising bars → close above EMA → BUY
        bars = rising_bars(80, step=0.0002)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.BUY

    def test_sell_signal_when_price_below_trend_ema(self) -> None:
        m = VolatilityRegimeModule(
            atr_period=5, avg_period=10, trend_period=10,
            low_atr_mult=0.3, high_atr_mult=3.0,
        )
        bars = falling_bars(80, step=0.0002)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.SELL

    def test_hold_on_low_volatility(self) -> None:
        # Use bars with zero spread so ATR=0 → ratio=0 < low_atr_mult → HOLD
        def zero_spread_bar(i: int) -> Bar:
            c = D("1.10000")
            ts = datetime(2024, 1, 1 + i // 24, i % 24, 0, 0, tzinfo=UTC)
            return Bar(
                symbol="EURUSD", timeframe=Timeframe.H1,
                timestamp_utc=ts, open=c, high=c, low=c, close=c, volume=D("100"),
            )
        bars = [zero_spread_bar(i) for i in range(80)]
        m = VolatilityRegimeModule(
            atr_period=5, avg_period=10, trend_period=10,
            low_atr_mult=0.5, high_atr_mult=2.0,
        )
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.HOLD
        assert sig.metadata["regime"] == "low_vol"

    def test_hold_on_high_volatility(self) -> None:
        # avg_period=40 captures the long flat baseline; the explosive final bars
        # push current ATR >> avg_ATR → ratio > high_atr_mult → HOLD
        m = VolatilityRegimeModule(
            atr_period=5, avg_period=40, trend_period=10,
            low_atr_mult=0.5, high_atr_mult=1.5,
        )
        bars = flat_bars(80, price=1.1000)
        # Replace last 5 bars with large explosive swings
        for i in range(75, 80):
            swing = D("0.0300") if i % 2 == 0 else D("-0.0300")
            bars[i] = make_bar(D("1.1000") + swing, i=i)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.HOLD
        assert sig.metadata["regime"] == "high_vol"

    def test_metadata_keys_present(self) -> None:
        m = VolatilityRegimeModule(atr_period=5, avg_period=10, trend_period=10)
        bars = rising_bars(80)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        for key in ("atr", "avg_atr", "atr_ratio", "trend_ema", "regime"):
            assert key in sig.metadata

    def test_confidence_in_valid_range(self) -> None:
        m = VolatilityRegimeModule(atr_period=5, avg_period=10, trend_period=10)
        bars = rising_bars(80)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert 0.0 <= sig.confidence <= 1.0


# ---------------------------------------------------------------------------
# SeasonalityModule
# ---------------------------------------------------------------------------

class TestSessionQualityClassifier:
    """Unit tests for the session quality classification logic."""

    def test_london_ny_overlap_tue_is_excellent(self) -> None:
        assert _classify_session(14, 1) == _SessionQuality.EXCELLENT  # Tue 14:00

    def test_london_ny_overlap_wed_is_excellent(self) -> None:
        assert _classify_session(13, 2) == _SessionQuality.EXCELLENT  # Wed 13:00

    def test_london_ny_overlap_mon_is_good_not_excellent(self) -> None:
        assert _classify_session(14, 0) == _SessionQuality.GOOD       # Mon 14:00

    def test_asian_session_is_poor(self) -> None:
        assert _classify_session(3, 1) == _SessionQuality.POOR        # Tue 03:00

    def test_dead_zone_is_poor(self) -> None:
        assert _classify_session(22, 2) == _SessionQuality.POOR       # Wed 22:00

    def test_friday_afternoon_is_poor(self) -> None:
        assert _classify_session(15, 4) == _SessionQuality.POOR       # Fri 15:00
        assert _classify_session(18, 4) == _SessionQuality.POOR       # Fri 18:00

    def test_friday_morning_is_good(self) -> None:
        assert _classify_session(9, 4) == _SessionQuality.GOOD        # Fri 09:00

    def test_weekend_is_poor(self) -> None:
        assert _classify_session(12, 5) == _SessionQuality.POOR       # Sat
        assert _classify_session(12, 6) == _SessionQuality.POOR       # Sun

    def test_london_session_tue_is_good(self) -> None:
        assert _classify_session(10, 1) == _SessionQuality.GOOD       # Tue 10:00

    def test_midnight_is_poor(self) -> None:
        assert _classify_session(0, 1) == _SessionQuality.POOR        # Tue 00:00


class TestSeasonalityModule:
    def test_module_id(self) -> None:
        m = SeasonalityModule(timeframe_id="h1")
        assert m.module_id == "seasonality_h1"

    def test_min_bars(self) -> None:
        m = SeasonalityModule(momentum_bars=3)
        assert m.min_bars == 4

    def test_raises_on_too_few_bars(self) -> None:
        m = SeasonalityModule(momentum_bars=3)
        with pytest.raises(ModuleNotReadyError):
            m.analyse(rising_bars(2), NOW_LONDON_OVERLAP)

    def test_raises_on_momentum_bars_zero(self) -> None:
        with pytest.raises(ValueError):
            SeasonalityModule(momentum_bars=0)

    def test_hold_on_poor_session(self) -> None:
        m = SeasonalityModule(momentum_bars=3)
        bars = rising_bars(10)
        sig = m.analyse(bars, NOW_ASIAN)
        assert sig.direction == SignalDirection.HOLD
        assert sig.metadata["session_quality"] == "poor"

    def test_hold_on_friday_afternoon(self) -> None:
        m = SeasonalityModule(momentum_bars=3)
        bars = rising_bars(10)
        sig = m.analyse(bars, NOW_FRIDAY_PM)
        assert sig.direction == SignalDirection.HOLD

    def test_excellent_session_buy_on_rising(self) -> None:
        m = SeasonalityModule(momentum_bars=3, momentum_threshold=0.0001)
        bars = rising_bars(10, step=0.0010)  # strong upward momentum
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)  # Tue 14:00 = excellent
        assert sig.direction == SignalDirection.BUY

    def test_excellent_session_sell_on_falling(self) -> None:
        m = SeasonalityModule(momentum_bars=3, momentum_threshold=0.0001)
        bars = falling_bars(10, step=0.0010)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.direction == SignalDirection.SELL

    def test_good_session_hold_on_weak_momentum(self) -> None:
        """Good session but momentum below threshold → HOLD."""
        m = SeasonalityModule(momentum_bars=3, momentum_threshold=0.0100)
        bars = flat_bars(10)  # effectively zero momentum
        # Mon 10:00 = GOOD session
        ts_mon = datetime(2024, 6, 3, 10, 0, 0, tzinfo=UTC)
        sig = m.analyse(bars, ts_mon)
        assert sig.direction == SignalDirection.HOLD

    def test_good_session_directional_on_strong_momentum(self) -> None:
        """Good session with strong momentum → BUY/SELL."""
        m = SeasonalityModule(momentum_bars=3, momentum_threshold=0.0001)
        bars = rising_bars(10, step=0.0020)
        ts_mon = datetime(2024, 6, 3, 10, 0, 0, tzinfo=UTC)
        sig = m.analyse(bars, ts_mon)
        assert sig.direction == SignalDirection.BUY

    def test_confidence_in_valid_range(self) -> None:
        m = SeasonalityModule()
        bars = rising_bars(10)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert 0.0 <= sig.confidence <= 1.0

    def test_confidence_cap_respected(self) -> None:
        m = SeasonalityModule(confidence_cap=0.70, momentum_threshold=0.0001)
        bars = rising_bars(10, step=0.0020)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        assert sig.confidence <= 0.70

    def test_metadata_keys_present(self) -> None:
        m = SeasonalityModule()
        bars = rising_bars(10)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)
        for key in ("session_quality", "weekday", "hour_utc", "momentum", "momentum_threshold"):
            assert key in sig.metadata

    def test_metadata_hour_and_weekday_correct(self) -> None:
        m = SeasonalityModule()
        bars = rising_bars(10)
        sig = m.analyse(bars, NOW_LONDON_OVERLAP)  # Tue 14:00
        assert sig.metadata["hour_utc"] == 14
        assert sig.metadata["weekday"] == 1  # Tuesday = 1

    def test_timestamp_propagated(self) -> None:
        m = SeasonalityModule()
        bars = rising_bars(10)
        ts = datetime(2024, 6, 4, 14, 30, 0, tzinfo=UTC)
        sig = m.analyse(bars, ts)
        assert sig.timestamp_utc == ts
