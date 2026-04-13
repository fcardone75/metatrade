"""Adaptive RSI analysis module.

Solves the core problem with fixed-threshold RSI: in a strong trend, RSI
stays in "overbought" territory for extended periods, generating false SELL
signals against the trend.

Solution: the oversold/overbought thresholds dynamically widen as ADX rises.

    ADX < 20 (ranging)  → oversold=30, overbought=70  (standard)
    ADX 20–35           → oversold=25, overbought=75  (tighter)
    ADX > 35 (trending) → oversold=20, overbought=80  (very tight — almost never fires)

In a strong trend (ADX > 35), it takes an extreme RSI move to generate a
counter-trend signal.  This drastically reduces false mean-reversion signals
when the market is trending strongly.

The module computes both RSI and ADX internally — no external dependency on
the ADX module's signals.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule
from metatrade.technical_analysis.indicators.rsi import rsi as compute_rsi
from metatrade.technical_analysis.indicators.adx import adx as compute_adx

_VERSION = ModuleVersion(1, 0, 0)


def _adaptive_thresholds(
    adx_value: float,
    ranging_threshold: float = 20.0,
    trending_threshold: float = 35.0,
    oversold_ranging: float = 30.0,
    overbought_ranging: float = 70.0,
    oversold_trending: float = 20.0,
    overbought_trending: float = 80.0,
) -> tuple[float, float]:
    """Linearly interpolate thresholds between ranging and trending regimes.

    Returns (oversold_threshold, overbought_threshold).
    """
    if adx_value <= ranging_threshold:
        return oversold_ranging, overbought_ranging
    if adx_value >= trending_threshold:
        return oversold_trending, overbought_trending

    # Linear interpolation between ranging and trending
    t = (adx_value - ranging_threshold) / (trending_threshold - ranging_threshold)
    oversold = oversold_ranging + t * (oversold_trending - oversold_ranging)
    overbought = overbought_ranging + t * (overbought_trending - overbought_ranging)
    return oversold, overbought


class AdaptiveRsiModule(ITechnicalModule):
    """RSI with ADX-adaptive oversold/overbought thresholds.

    Args:
        rsi_period:           RSI calculation period (default 14).
        adx_period:           ADX calculation period (default 14).
        ranging_adx:          ADX level below which thresholds are at their widest
                              (most signals allowed).  Default 20.
        trending_adx:         ADX level above which thresholds are at their tightest
                              (fewest signals allowed — avoids counter-trend).  Default 35.
        oversold_ranging:     Oversold threshold when ADX < ranging_adx.  Default 30.
        overbought_ranging:   Overbought threshold when ADX < ranging_adx.  Default 70.
        oversold_trending:    Oversold threshold when ADX > trending_adx.  Default 20.
        overbought_trending:  Overbought threshold when ADX > trending_adx.  Default 80.
        timeframe_id:         Suffix for ``module_id``.
        confidence_cap:       Maximum signal confidence.  Default 0.88.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        adx_period: int = 14,
        ranging_adx: float = 20.0,
        trending_adx: float = 35.0,
        oversold_ranging: float = 30.0,
        overbought_ranging: float = 70.0,
        oversold_trending: float = 20.0,
        overbought_trending: float = 80.0,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.88,
    ) -> None:
        if rsi_period < 2:
            raise ValueError(f"rsi_period must be >= 2, got {rsi_period}")
        if adx_period < 2:
            raise ValueError(f"adx_period must be >= 2, got {adx_period}")
        if ranging_adx >= trending_adx:
            raise ValueError("ranging_adx must be < trending_adx")
        self._rsi_period = rsi_period
        self._adx_period = adx_period
        self._ranging_adx = ranging_adx
        self._trending_adx = trending_adx
        self._os_ranging = oversold_ranging
        self._ob_ranging = overbought_ranging
        self._os_trending = oversold_trending
        self._ob_trending = overbought_trending
        self._timeframe_id = timeframe_id
        self._confidence_cap = confidence_cap

    @property
    def module_id(self) -> str:
        return f"adaptive_rsi_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # Needs enough bars for both RSI and ADX (ADX needs 2*period+5)
        return max(self._rsi_period + 2, self._adx_period * 2 + 5)

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        # ── Compute RSI ───────────────────────────────────────────────────────
        rsi_vals = compute_rsi(closes, self._rsi_period)
        curr_rsi = float(rsi_vals[-1])

        # ── Compute ADX ───────────────────────────────────────────────────────
        adx_vals, plus_di, minus_di = compute_adx(highs, lows, closes, self._adx_period)
        curr_adx = float(adx_vals[-1])

        # ── Adaptive thresholds ───────────────────────────────────────────────
        oversold, overbought = _adaptive_thresholds(
            curr_adx,
            ranging_threshold=self._ranging_adx,
            trending_threshold=self._trending_adx,
            oversold_ranging=self._os_ranging,
            overbought_ranging=self._ob_ranging,
            oversold_trending=self._os_trending,
            overbought_trending=self._ob_trending,
        )

        regime = (
            "ranging" if curr_adx < self._ranging_adx
            else "trending" if curr_adx > self._trending_adx
            else "transitioning"
        )

        # ── Signal ───────────────────────────────────────────────────────────
        if curr_rsi < oversold:
            direction = SignalDirection.BUY
            depth = (oversold - curr_rsi) / oversold
            confidence = min(self._confidence_cap, 0.50 + depth * 0.50)
            reason = (
                f"AdaptiveRSI({self._rsi_period})={curr_rsi:.1f} < {oversold:.0f} "
                f"(ADX={curr_adx:.1f} → {regime} regime)"
            )

        elif curr_rsi > overbought:
            direction = SignalDirection.SELL
            depth = (curr_rsi - overbought) / (100.0 - overbought)
            confidence = min(self._confidence_cap, 0.50 + depth * 0.50)
            reason = (
                f"AdaptiveRSI({self._rsi_period})={curr_rsi:.1f} > {overbought:.0f} "
                f"(ADX={curr_adx:.1f} → {regime} regime)"
            )

        else:
            direction = SignalDirection.HOLD
            confidence = 0.55
            reason = (
                f"AdaptiveRSI({self._rsi_period})={curr_rsi:.1f} neutral "
                f"[{oversold:.0f}, {overbought:.0f}] "
                f"(ADX={curr_adx:.1f} → {regime})"
            )

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "rsi": round(curr_rsi, 2),
                "adx": round(curr_adx, 2),
                "oversold_threshold": round(oversold, 1),
                "overbought_threshold": round(overbought, 1),
                "regime": regime,
            },
        )
