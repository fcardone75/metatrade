"""Market Regime classifier — single source of truth for regime detection.

Synthesises ADX (trend strength/direction), ATR ratio (volatility level),
and the Hurst Exponent (statistical persistence) into a single regime label
and a directional signal.

Regime taxonomy:
    TRENDING_UP    ADX ≥ adx_threshold  AND  +DI > −DI  AND  ATR ratio ≤ high_vol
    TRENDING_DOWN  ADX ≥ adx_threshold  AND  −DI > +DI  AND  ATR ratio ≤ high_vol
    RANGING        ADX < adx_threshold  AND  ATR ratio ≤ high_vol
    VOLATILE       ATR ratio > high_vol  (dominates all other conditions)

Signal logic:
    TRENDING_UP    → BUY  (trend-following bias, ride the move)
    TRENDING_DOWN  → SELL
    RANGING        → HOLD (mean-reversion conditions; other modules decide)
    VOLATILE       → HOLD (spread/gap risk too high)

Hurst Exponent adjustment (applied after regime classification):
    H > hurst_trend_threshold (0.55)  — price series is persistent / trending.
        Trending regimes: confidence += hurst_boost (0.04)  — Hurst confirms
        Ranging regimes:  confidence -= hurst_boost          — contradicts
    H < hurst_mean_rev_threshold (0.45) — price series is mean-reverting.
        Ranging regimes:  confidence += hurst_boost           — Hurst confirms
        Trending regimes: confidence -= hurst_boost           — contradicts
    0.45 ≤ H ≤ 0.55 — near-random walk: no adjustment.

Confidence (before Hurst adjustment):
    Trending: base 0.55 + ADX surplus above threshold / 100
              (stronger trend = higher confidence, capped at confidence_cap)
    HOLD:     0.50 (neutral — does not suppress other modules)
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule
from metatrade.technical_analysis.indicators.adx import adx as compute_adx
from metatrade.technical_analysis.indicators.atr import atr as compute_atr
from metatrade.technical_analysis.indicators.hurst import hurst_exponent

_VERSION = ModuleVersion(2, 0, 0)  # bumped: Hurst Exponent added


class _Regime(str, Enum):
    TRENDING_UP   = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING       = "ranging"
    VOLATILE      = "volatile"


class MarketRegimeModule(ITechnicalModule):
    """Centralized market regime classifier with Hurst Exponent confirmation.

    Args:
        adx_period:              ADX calculation period (default 14).
        adx_threshold:           Minimum ADX to declare a trend (default 25).
        atr_period:              ATR period for volatility measurement (default 14).
        atr_avg_period:          Period for computing average ATR (default 50).
        high_vol_mult:           ATR > high_vol_mult × avg_ATR → VOLATILE (default 2.0).
        hurst_window:            Bars used for Hurst Exponent calculation (default 100).
                                 Set to 0 to disable Hurst adjustment.
        hurst_trend_threshold:   H above this → trending confirmed (default 0.55).
        hurst_mean_rev_threshold: H below this → mean-reverting confirmed (default 0.45).
        hurst_boost:             Confidence adjustment when Hurst confirms regime
                                 (default 0.04, subtracted when it contradicts).
        timeframe_id:            Suffix for module_id.
        confidence_cap:          Maximum signal confidence (default 0.88).
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        atr_period: int = 14,
        atr_avg_period: int = 50,
        high_vol_mult: float = 2.0,
        hurst_window: int = 100,
        hurst_trend_threshold: float = 0.55,
        hurst_mean_rev_threshold: float = 0.45,
        hurst_boost: float = 0.04,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.88,
    ) -> None:
        if adx_period < 2:
            raise ValueError(f"adx_period must be >= 2, got {adx_period}")
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if atr_avg_period < atr_period:
            raise ValueError("atr_avg_period must be >= atr_period")
        self._adx_period              = adx_period
        self._adx_threshold           = adx_threshold
        self._atr_period              = atr_period
        self._atr_avg_period          = atr_avg_period
        self._high_vol_mult           = high_vol_mult
        self._hurst_window            = hurst_window
        self._hurst_trend_thr         = hurst_trend_threshold
        self._hurst_mean_rev_thr      = hurst_mean_rev_threshold
        self._hurst_boost             = hurst_boost
        self._timeframe_id            = timeframe_id
        self._confidence_cap          = confidence_cap

    @property
    def module_id(self) -> str:
        return f"market_regime_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # ADX needs 2*period+5; ATR average needs atr_avg_period bars
        base = max(self._adx_period * 2 + 5, self._atr_avg_period + self._atr_period + 5)
        # If Hurst is enabled, also need hurst_window bars (but that's usually smaller)
        if self._hurst_window > 0:
            return max(base, self._hurst_window)
        return base

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        highs  = [b.high  for b in bars]
        lows   = [b.low   for b in bars]
        closes = [b.close for b in bars]

        # ── ATR volatility ratio ──────────────────────────────────────────────
        atr_series = compute_atr(highs, lows, closes, self._atr_period)
        curr_atr   = float(atr_series[-1])
        avg_window = atr_series[-(self._atr_avg_period + 1):-1]
        avg_atr    = sum(float(v) for v in avg_window) / len(avg_window) if avg_window else 0.0
        vol_ratio  = (curr_atr / avg_atr) if avg_atr > 0 else 0.0

        # ── ADX ───────────────────────────────────────────────────────────────
        adx_vals, plus_di, minus_di = compute_adx(highs, lows, closes, self._adx_period)
        curr_adx  = float(adx_vals[-1])
        curr_pdi  = float(plus_di[-1])
        curr_mdi  = float(minus_di[-1])

        # ── Hurst Exponent ────────────────────────────────────────────────────
        hurst_val = 0.5  # default = random walk (no adjustment)
        if self._hurst_window > 0 and len(closes) >= self._hurst_window:
            hurst_val = hurst_exponent(closes[-self._hurst_window:])

        # ── Regime classification ─────────────────────────────────────────────
        if vol_ratio > self._high_vol_mult:
            regime = _Regime.VOLATILE
        elif curr_adx >= self._adx_threshold:
            regime = _Regime.TRENDING_UP if curr_pdi >= curr_mdi else _Regime.TRENDING_DOWN
        else:
            regime = _Regime.RANGING

        # ── Base confidence & direction ───────────────────────────────────────
        if regime == _Regime.TRENDING_UP:
            direction  = SignalDirection.BUY
            surplus    = max(0.0, curr_adx - self._adx_threshold)
            confidence = 0.55 + surplus / 100.0
            reason = (
                f"MarketRegime: TRENDING_UP "
                f"(ADX={curr_adx:.1f}, +DI={curr_pdi:.1f} > −DI={curr_mdi:.1f}, "
                f"vol_ratio={vol_ratio:.2f})"
            )

        elif regime == _Regime.TRENDING_DOWN:
            direction  = SignalDirection.SELL
            surplus    = max(0.0, curr_adx - self._adx_threshold)
            confidence = 0.55 + surplus / 100.0
            reason = (
                f"MarketRegime: TRENDING_DOWN "
                f"(ADX={curr_adx:.1f}, −DI={curr_mdi:.1f} > +DI={curr_pdi:.1f}, "
                f"vol_ratio={vol_ratio:.2f})"
            )

        elif regime == _Regime.VOLATILE:
            direction  = SignalDirection.HOLD
            confidence = 0.50
            reason = (
                f"MarketRegime: VOLATILE — ATR ratio={vol_ratio:.2f} > "
                f"{self._high_vol_mult:.1f}× avg ({curr_atr:.5f} / {avg_atr:.5f})"
            )

        else:  # RANGING
            direction  = SignalDirection.HOLD
            confidence = 0.50
            reason = (
                f"MarketRegime: RANGING "
                f"(ADX={curr_adx:.1f} < {self._adx_threshold:.0f}, "
                f"vol_ratio={vol_ratio:.2f})"
            )

        # ── Hurst adjustment ──────────────────────────────────────────────────
        hurst_label = "random"
        if hurst_val > self._hurst_trend_thr:
            hurst_label = "trending"
            if regime in (_Regime.TRENDING_UP, _Regime.TRENDING_DOWN):
                confidence += self._hurst_boost    # confirmed
            elif regime in (_Regime.RANGING, _Regime.VOLATILE):
                confidence -= self._hurst_boost    # contradicted
        elif hurst_val < self._hurst_mean_rev_thr:
            hurst_label = "mean_reverting"
            if regime in (_Regime.RANGING, _Regime.VOLATILE):
                confidence += self._hurst_boost    # confirmed
            elif regime in (_Regime.TRENDING_UP, _Regime.TRENDING_DOWN):
                confidence -= self._hurst_boost    # contradicted

        # Append Hurst info to reason
        reason += f", H={hurst_val:.3f}({hurst_label})"

        # Apply final cap and floor
        confidence = max(0.40, min(self._confidence_cap, confidence))

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "regime":    regime.value,
                "adx":       round(curr_adx, 2),
                "plus_di":   round(curr_pdi, 2),
                "minus_di":  round(curr_mdi, 2),
                "atr":       round(curr_atr, 6),
                "avg_atr":   round(avg_atr, 6),
                "vol_ratio": round(vol_ratio, 3),
                "hurst":     round(hurst_val, 4),
                "hurst_label": hurst_label,
            },
        )
