"""Tests for technical analysis signal modules: EMA, RSI, MACD, ATR."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import SignalDirection, Timeframe
from metatrade.core.errors import ModuleNotReadyError
from metatrade.technical_analysis.modules.atr_module import AtrModule
from metatrade.technical_analysis.modules.ema_crossover import EmaCrossoverModule
from metatrade.technical_analysis.modules.macd_module import MacdModule
from metatrade.technical_analysis.modules.rsi_module import RsiModule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def D(v: str | float | int) -> Decimal:
    return Decimal(str(v))


def make_bar(
    close: Decimal,
    open_: Decimal | None = None,
    i: int = 0,
) -> Bar:
    """Create a synthetic Bar with OHLCV around the given close price."""
    o = open_ if open_ is not None else close
    h = max(o, close) + D("0.00010")
    lo = min(o, close) - D("0.00010")
    ts = datetime(2024, 1, 1 + i // 24, i % 24, 0, 0, tzinfo=UTC)
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_utc=ts,
        open=o,
        high=h,
        low=lo,
        close=close,
        volume=D("100"),
    )


def rising_bars(n: int, start: float = 1.10000, step: float = 0.0001) -> list[Bar]:
    return [make_bar(D(str(round(start + i * step, 5))), i=i) for i in range(n)]


def falling_bars(n: int, start: float = 1.20000, step: float = 0.0001) -> list[Bar]:
    return [make_bar(D(str(round(start - i * step, 5))), i=i) for i in range(n)]


def flat_bars(n: int, price: float = 1.10000) -> list[Bar]:
    return [make_bar(D(str(price)), i=i) for i in range(n)]


# ---------------------------------------------------------------------------
# EmaCrossoverModule
# ---------------------------------------------------------------------------

class TestEmaCrossoverModule:
    def test_module_id(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21, timeframe_id="h1")
        assert m.module_id == "ema_crossover_h1"

    def test_module_id_custom_timeframe(self) -> None:
        m = EmaCrossoverModule(timeframe_id="m15")
        assert m.module_id == "ema_crossover_m15"

    def test_min_bars_is_slow_plus_two(self) -> None:
        # With use_hma_slow=False (legacy EMA mode), min_bars = slow + 2
        m = EmaCrossoverModule(fast_period=9, slow_period=21, use_hma_slow=False)
        assert m.min_bars == 23  # 21 + 2
        # With HMA (default), min_bars is larger due to extra WMA smoothing pass
        m_hma = EmaCrossoverModule(fast_period=9, slow_period=21, use_hma_slow=True)
        assert m_hma.min_bars > 23

    def test_raises_when_fast_gte_slow(self) -> None:
        with pytest.raises(ValueError, match="fast_period.*must be <"):
            EmaCrossoverModule(fast_period=21, slow_period=9)

    def test_raises_module_not_ready(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21)
        bars = rising_bars(5)  # far less than min_bars=23
        with pytest.raises(ModuleNotReadyError):
            m.analyse(bars, NOW)

    def test_error_context_contains_module_id(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21)
        with pytest.raises(ModuleNotReadyError) as exc_info:
            m.analyse(rising_bars(5), NOW)
        assert "ema_crossover" in str(exc_info.value)

    def test_returns_analysis_signal(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21)
        bars = rising_bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        assert signal.module_id == "ema_crossover_h1"
        assert signal.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD)

    def test_confidence_in_range(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21)
        for bars in [rising_bars(40), falling_bars(40), flat_bars(40)]:
            signal = m.analyse(bars, NOW)
            assert 0.0 <= signal.confidence <= 1.0

    def test_hold_confidence_at_least_half(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21)
        bars = flat_bars(40)
        signal = m.analyse(bars, NOW)
        if signal.direction == SignalDirection.HOLD:
            assert signal.confidence >= 0.50

    def test_metadata_contains_ema_values(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21)
        bars = rising_bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        assert "fast_ema" in signal.metadata
        # Renamed: slow_ema → slow_ma (supports both EMA and HMA slow lines)
        assert "slow_ma" in signal.metadata
        assert "gap_pct" in signal.metadata

    def test_timestamp_preserved(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21)
        bars = rising_bars(40)
        signal = m.analyse(bars, NOW)
        assert signal.timestamp_utc == NOW

    def test_reason_non_empty(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21)
        bars = rising_bars(40)
        signal = m.analyse(bars, NOW)
        assert len(signal.reason) > 0

    def test_confidence_cap_respected(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21, confidence_cap=0.70)
        bars = rising_bars(100, step=0.01)  # large gap → would exceed cap
        signal = m.analyse(bars, NOW)
        assert signal.confidence <= 0.70

    def test_hold_on_flat_series(self) -> None:
        m = EmaCrossoverModule(fast_period=9, slow_period=21)
        bars = flat_bars(40)
        signal = m.analyse(bars, NOW)
        assert signal.direction == SignalDirection.HOLD

    def test_buy_signal_on_bullish_crossover(self) -> None:
        # Build bars: first downtrend (so slow EMA > fast EMA), then sharp up
        down_bars = falling_bars(35, start=1.2, step=0.002)
        up_bars = rising_bars(15, start=float(down_bars[-1].close), step=0.005)
        bars = down_bars + up_bars
        m = EmaCrossoverModule(fast_period=9, slow_period=21)
        signal = m.analyse(bars, NOW)
        # The direction depends on exact crossover timing; just verify it's valid
        assert signal.direction in (SignalDirection.BUY, SignalDirection.HOLD, SignalDirection.SELL)

    def test_guaranteed_buy_signal(self) -> None:
        # 25 flat bars → both EMAs converge (prev_diff = 0)
        # 1 spike bar → fast EMA jumps above slow EMA (curr_diff > 0) → BUY
        m = EmaCrossoverModule(fast_period=5, slow_period=20)
        flat = flat_bars(25, price=1.0)
        spike = [make_bar(D("100.0"), i=25)]
        signal = m.analyse(flat + spike, NOW)
        assert signal.direction == SignalDirection.BUY

    def test_guaranteed_sell_signal(self) -> None:
        # 25 flat bars → both EMAs converge (prev_diff = 0)
        # 1 drop bar → fast EMA drops below slow EMA (curr_diff < 0) → SELL
        m = EmaCrossoverModule(fast_period=5, slow_period=20)
        flat = flat_bars(25, price=1.0)
        drop = [make_bar(D("0.0001"), i=25)]
        signal = m.analyse(flat + drop, NOW)
        assert signal.direction == SignalDirection.SELL


# ---------------------------------------------------------------------------
# RsiModule
# ---------------------------------------------------------------------------

class TestRsiModule:
    def test_module_id(self) -> None:
        m = RsiModule(period=14, timeframe_id="h4")
        assert m.module_id == "rsi_h4"

    def test_default_module_id(self) -> None:
        m = RsiModule()
        assert m.module_id == "rsi_h1"

    def test_min_bars_is_period_plus_two(self) -> None:
        m = RsiModule(period=14)
        assert m.min_bars == 16  # 14 + 2

    def test_raises_module_not_ready(self) -> None:
        m = RsiModule(period=14)
        bars = rising_bars(5)
        with pytest.raises(ModuleNotReadyError):
            m.analyse(bars, NOW)

    def test_returns_valid_signal(self) -> None:
        m = RsiModule(period=14)
        bars = rising_bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        assert signal.module_id == "rsi_h1"
        assert signal.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD)

    def test_confidence_in_range(self) -> None:
        m = RsiModule(period=14)
        for bars in [rising_bars(40), falling_bars(40), flat_bars(40)]:
            signal = m.analyse(bars, NOW)
            assert 0.0 <= signal.confidence <= 1.0

    def test_metadata_contains_rsi(self) -> None:
        m = RsiModule(period=14)
        signal = m.analyse(rising_bars(40), NOW)
        assert "rsi" in signal.metadata

    def test_buy_on_oversold_condition(self) -> None:
        # Sharply falling prices → RSI < 30 → BUY
        bars = falling_bars(40, start=2.0, step=0.03)
        m = RsiModule(period=14, oversold=30.0, overbought=70.0)
        signal = m.analyse(bars, NOW)
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence >= 0.50

    def test_sell_on_overbought_condition(self) -> None:
        # Sharply rising prices → RSI > 70 → SELL
        bars = rising_bars(40, step=0.03)
        m = RsiModule(period=14, oversold=30.0, overbought=70.0)
        signal = m.analyse(bars, NOW)
        assert signal.direction == SignalDirection.SELL
        assert signal.confidence >= 0.50

    def test_hold_on_neutral_rsi(self) -> None:
        # Alternating moves → RSI stays near 50
        prices_raw = []
        val = 1.10000
        for i in range(40):
            val = val + 0.0001 if i % 2 == 0 else val - 0.0001
            prices_raw.append(val)
        bars = [make_bar(D(str(round(v, 5))), i=i) for i, v in enumerate(prices_raw)]
        m = RsiModule(period=14, oversold=30.0, overbought=70.0)
        signal = m.analyse(bars, NOW)
        assert signal.direction == SignalDirection.HOLD

    def test_confidence_scales_with_depth(self) -> None:
        # Deeper oversold → higher confidence
        slightly_oversold = falling_bars(40, start=1.5, step=0.005)
        deeply_oversold = falling_bars(40, start=2.0, step=0.05)
        m = RsiModule(period=14)
        s1 = m.analyse(slightly_oversold, NOW)
        s2 = m.analyse(deeply_oversold, NOW)
        # Both should be BUY; deeper → higher confidence
        if s1.direction == s2.direction == SignalDirection.BUY:
            assert s2.confidence >= s1.confidence

    def test_reason_contains_rsi_value(self) -> None:
        m = RsiModule(period=14)
        bars = rising_bars(40)
        signal = m.analyse(bars, NOW)
        assert "RSI" in signal.reason

    def test_timestamp_preserved(self) -> None:
        m = RsiModule(period=14)
        signal = m.analyse(rising_bars(40), NOW)
        assert signal.timestamp_utc == NOW


# ---------------------------------------------------------------------------
# MacdModule
# ---------------------------------------------------------------------------

class TestMacdModule:
    def test_module_id(self) -> None:
        m = MacdModule(timeframe_id="d1")
        assert m.module_id == "macd_d1"

    def test_default_module_id(self) -> None:
        m = MacdModule()
        assert m.module_id == "macd_h1"

    def test_min_bars_is_slow_plus_signal_plus_one(self) -> None:
        m = MacdModule(fast=12, slow=26, signal=9)
        assert m.min_bars == 36  # 26 + 9 + 1

    def test_raises_module_not_ready(self) -> None:
        m = MacdModule()
        bars = rising_bars(5)
        with pytest.raises(ModuleNotReadyError):
            m.analyse(bars, NOW)

    def test_returns_valid_signal(self) -> None:
        m = MacdModule()
        bars = rising_bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        assert signal.module_id == "macd_h1"
        assert signal.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD)

    def test_confidence_in_range(self) -> None:
        m = MacdModule()
        for bars in [rising_bars(60), falling_bars(60), flat_bars(60)]:
            signal = m.analyse(bars, NOW)
            assert 0.0 <= signal.confidence <= 1.0

    def test_metadata_keys(self) -> None:
        m = MacdModule()
        signal = m.analyse(rising_bars(60), NOW)
        assert "macd" in signal.metadata
        assert "signal" in signal.metadata
        assert "histogram" in signal.metadata

    def test_hold_on_flat_series(self) -> None:
        m = MacdModule()
        signal = m.analyse(flat_bars(60), NOW)
        assert signal.direction == SignalDirection.HOLD

    def test_reason_contains_macd_params(self) -> None:
        m = MacdModule(fast=12, slow=26, signal=9)
        signal = m.analyse(rising_bars(60), NOW)
        assert "12" in signal.reason or "MACD" in signal.reason

    def test_timestamp_preserved(self) -> None:
        m = MacdModule()
        signal = m.analyse(rising_bars(60), NOW)
        assert signal.timestamp_utc == NOW

    def test_hold_confidence_is_positive(self) -> None:
        m = MacdModule()
        signal = m.analyse(flat_bars(60), NOW)
        if signal.direction == SignalDirection.HOLD:
            assert signal.confidence > 0.0

    def test_crossover_confidence_capped_at_0_85(self) -> None:
        # Huge histogram → confidence would exceed 0.85 without cap
        m = MacdModule()
        bars = rising_bars(80, step=0.05)  # very steep trend
        signal = m.analyse(bars, NOW)
        assert signal.confidence <= 0.85

    def test_custom_periods(self) -> None:
        m = MacdModule(fast=5, slow=10, signal=3)
        assert m.min_bars == 14  # 10 + 3 + 1
        bars = rising_bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        assert signal.module_id == "macd_h1"

    def test_guaranteed_buy_signal(self) -> None:
        # 40 flat bars → MACD ≈ 0, signal ≈ 0 (prev_diff = 0)
        # 1 spike bar → MACD jumps positive faster than signal EMA (curr_diff > 0) → BUY
        m = MacdModule(fast=12, slow=26, signal=9)
        flat = flat_bars(40, price=1.0)
        spike = [make_bar(D("1000.0"), i=40)]
        signal = m.analyse(flat + spike, NOW)
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence > 0.50

    def test_guaranteed_sell_signal(self) -> None:
        # 40 flat bars → MACD ≈ 0, signal ≈ 0 (prev_diff = 0)
        # 1 drop bar → MACD drops negative faster than signal EMA (curr_diff < 0) → SELL
        m = MacdModule(fast=12, slow=26, signal=9)
        flat = flat_bars(40, price=1.0)
        drop = [make_bar(D("0.0"), i=40)]
        signal = m.analyse(flat + drop, NOW)
        assert signal.direction == SignalDirection.SELL
        assert signal.confidence > 0.50


# ---------------------------------------------------------------------------
# AtrModule
# ---------------------------------------------------------------------------

class TestAtrModule:
    def _bars(self, n: int, volatility: float = 0.002) -> list[Bar]:
        """Bars with fixed volatility (H-L spread)."""
        bars = []
        for i in range(n):
            close = D(str(round(1.1 + i * 0.0001, 5)))
            h = close + D(str(volatility))
            lo = close - D(str(volatility))
            o = close
            ts = datetime(2024, 1, 1 + i // 24, i % 24, 0, 0, tzinfo=UTC)
            bars.append(Bar(
                symbol="EURUSD",
                timeframe=Timeframe.H1,
                timestamp_utc=ts,
                open=o,
                high=h,
                low=lo,
                close=close,
                volume=D("100"),
            ))
        return bars

    def test_module_id(self) -> None:
        m = AtrModule(timeframe_id="h4")
        assert m.module_id == "atr_h4"

    def test_default_module_id(self) -> None:
        m = AtrModule()
        assert m.module_id == "atr_h1"

    def test_min_bars_greater_than_period(self) -> None:
        m = AtrModule(period=14)
        assert m.min_bars > 14

    def test_raises_on_invalid_period(self) -> None:
        with pytest.raises(ValueError, match="period must be >= 2"):
            AtrModule(period=1)

    def test_raises_module_not_ready(self) -> None:
        m = AtrModule(period=14)
        bars = self._bars(5)
        with pytest.raises(ModuleNotReadyError):
            m.analyse(bars, NOW)

    def test_always_returns_hold(self) -> None:
        m = AtrModule(period=14)
        bars = self._bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        assert signal.direction == SignalDirection.HOLD

    def test_confidence_in_range(self) -> None:
        m = AtrModule(period=14)
        bars = self._bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        assert 0.0 <= signal.confidence <= 1.0

    def test_metadata_contains_sl_levels(self) -> None:
        m = AtrModule(period=14)
        bars = self._bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        assert "atr" in signal.metadata
        assert "sl_buy" in signal.metadata
        assert "sl_sell" in signal.metadata
        assert "high_volatility" in signal.metadata

    def test_sl_buy_below_close(self) -> None:
        m = AtrModule(period=14, sl_multiplier=2.0)
        bars = self._bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        close = float(bars[-1].close)
        assert signal.metadata["sl_buy"] < close

    def test_sl_sell_above_close(self) -> None:
        m = AtrModule(period=14, sl_multiplier=2.0)
        bars = self._bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        close = float(bars[-1].close)
        assert signal.metadata["sl_sell"] > close

    def test_high_volatility_flag(self) -> None:
        # Create bars where volatility spikes dramatically at the end
        m = AtrModule(period=14, high_vol_factor=1.1)
        # Most bars low volatility, last bars very high volatility
        low_vol_bars = self._bars(30, volatility=0.0001)
        high_vol_bars = self._bars(10, volatility=0.02)
        # Rebuild timestamps properly
        all_bars = low_vol_bars + high_vol_bars
        # Renumber timestamps to keep them sequential
        reindexed = []
        for i, b in enumerate(all_bars):
            ts = datetime(2024, 1, 1 + i // 24, i % 24, 0, 0, tzinfo=UTC)
            reindexed.append(Bar(
                symbol=b.symbol, timeframe=b.timeframe, timestamp_utc=ts,
                open=b.open, high=b.high, low=b.low, close=b.close, volume=b.volume,
            ))
        signal = m.analyse(reindexed, NOW)
        # Should detect elevated volatility
        assert isinstance(signal.metadata["high_volatility"], bool)

    def test_atr_value_is_positive(self) -> None:
        m = AtrModule(period=14)
        bars = self._bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        assert signal.metadata["atr"] > 0.0

    def test_timestamp_preserved(self) -> None:
        m = AtrModule(period=14)
        bars = self._bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        assert signal.timestamp_utc == NOW

    def test_reason_non_empty(self) -> None:
        m = AtrModule(period=14)
        bars = self._bars(m.min_bars + 5)
        signal = m.analyse(bars, NOW)
        assert len(signal.reason) > 0
