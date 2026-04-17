"""Volatility Regime analysis module.

Identifies whether the market is in a "tradeable" volatility regime and emits
directional signals only when conditions are favourable.

Three regimes:
    LOW VOLATILITY  (ATR < low_atr_mult × avg_ATR):
        Market is too flat — signals from other modules are likely false.
        Emits HOLD to withhold its vote from consensus.

    HIGH VOLATILITY (ATR > high_atr_mult × avg_ATR):
        Market is too explosive — spread widens, stops get hit erratically.
        Emits HOLD to withhold its vote.

    NORMAL VOLATILITY:
        Market is in a tradeable range.  Emits BUY if price is above the
        trend EMA, SELL if below.  This module acts both as a volatility
        filter (via HOLD in extremes) and as a mild trend-direction signal
        in normal conditions.

Parameters:
    atr_period:     ATR look-back (default 14).
    avg_period:     Rolling window to compute "average" ATR (default 50).
    trend_period:   EMA period for trend direction (default 50).
    low_atr_mult:   ATR/avg_ATR below this → low-vol HOLD (default 0.5).
    high_atr_mult:  ATR/avg_ATR above this → high-vol HOLD (default 2.0).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.indicators.atr import atr as compute_atr
from metatrade.technical_analysis.indicators.ema import ema as compute_ema
from metatrade.technical_analysis.interface import ITechnicalModule

_VERSION = ModuleVersion(1, 0, 0)


class VolatilityRegimeModule(ITechnicalModule):
    """ATR-based volatility regime filter + trend direction signal.

    Args:
        atr_period:     ATR look-back period (default 14).
        avg_period:     Bars used to compute the ATR rolling average (default 50).
        trend_period:   EMA period for directional bias (default 50).
        low_atr_mult:   ATR/avg_ATR ratio below which volatility is "too low" (default 0.5).
        high_atr_mult:  ATR/avg_ATR ratio above which volatility is "too high" (default 2.0).
        timeframe_id:   Suffix for ``module_id`` (e.g. "h1").
        confidence_cap: Maximum confidence for directional signals (default 0.80).
    """

    def __init__(
        self,
        atr_period: int = 14,
        avg_period: int = 50,
        trend_period: int = 50,
        low_atr_mult: float = 0.5,
        high_atr_mult: float = 2.0,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.80,
    ) -> None:
        if atr_period < 2:
            raise ValueError(f"atr_period must be >= 2, got {atr_period}")
        if avg_period <= atr_period:
            raise ValueError("avg_period must be > atr_period")
        self._atr_period = atr_period
        self._avg_period = avg_period
        self._trend_period = trend_period
        self._low_mult = low_atr_mult
        self._high_mult = high_atr_mult
        self._timeframe_id = timeframe_id
        self._confidence_cap = confidence_cap

    @property
    def module_id(self) -> str:
        return f"volatility_regime_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # Need enough bars for ATR + avg ATR window + trend EMA
        return max(self._avg_period, self._trend_period) + self._atr_period + 5

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]

        # ── ATR and rolling average ───────────────────────────────────────────
        atr_series = compute_atr(highs, lows, closes, self._atr_period)
        curr_atr = atr_series[-1]

        # Rolling average of ATR over the last avg_period values
        atr_window = atr_series[-self._avg_period :]
        avg_atr = sum(atr_window) / Decimal(len(atr_window))

        if avg_atr == 0:
            # Both current and average ATR are zero → perfectly flat market
            ratio = 0.0
        else:
            ratio = float(curr_atr / avg_atr)

        # ── Trend EMA ─────────────────────────────────────────────────────────
        trend_ema = compute_ema(closes, self._trend_period)
        curr_close = closes[-1]
        curr_ema = trend_ema[-1]

        # ── Regime classification ─────────────────────────────────────────────
        if ratio < self._low_mult:
            # Market too flat
            direction = SignalDirection.HOLD
            confidence = 0.50
            regime = "low_vol"
            reason = (
                f"Low volatility regime: ATR={float(curr_atr):.5f} is "
                f"{ratio:.2f}× avg (threshold {self._low_mult}×) — flat market"
            )

        elif ratio > self._high_mult:
            # Market too explosive
            direction = SignalDirection.HOLD
            confidence = 0.50
            regime = "high_vol"
            reason = (
                f"High volatility regime: ATR={float(curr_atr):.5f} is "
                f"{ratio:.2f}× avg (threshold {self._high_mult}×) — explosive market"
            )

        else:
            # Normal regime — use price vs trend EMA for direction
            regime = "normal"
            if curr_close > curr_ema:
                direction = SignalDirection.BUY
                reason = (
                    f"Normal vol regime (ATR ratio={ratio:.2f}): "
                    f"close={float(curr_close):.5f} > EMA({self._trend_period})={float(curr_ema):.5f}"
                )
            else:
                direction = SignalDirection.SELL
                reason = (
                    f"Normal vol regime (ATR ratio={ratio:.2f}): "
                    f"close={float(curr_close):.5f} < EMA({self._trend_period})={float(curr_ema):.5f}"
                )
            # Confidence scales with how centred the ratio is in the normal range
            # Mid of normal range = (low_mult + high_mult) / 2; further from edges = higher conf
            range_mid = (self._low_mult + self._high_mult) / 2
            centrality = 1.0 - abs(ratio - range_mid) / (range_mid - self._low_mult)
            centrality = max(0.0, min(1.0, centrality))
            confidence = min(self._confidence_cap, 0.55 + centrality * 0.25)

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "atr": float(curr_atr),
                "avg_atr": float(avg_atr),
                "atr_ratio": round(ratio, 4),
                "trend_ema": float(curr_ema),
                "regime": regime,
            },
        )
