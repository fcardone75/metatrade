"""Pivot Points analysis module.

Classic pivot points are daily support/resistance levels computed from the
previous session's High, Low, and Close.  They are widely watched by
institutional traders and often act as price magnets.

Calculation (standard / floor method):
    P  = (H + L + C) / 3          # pivot
    R1 = 2P − L                   # resistance 1
    R2 = P + (H − L)              # resistance 2
    S1 = 2P − H                   # support 1
    S2 = P − (H − L)              # support 2

Signal logic:
    BUY:  Current close is within ``atr_touch_mult × ATR`` of S1 or S2
    SELL: Current close is within ``atr_touch_mult × ATR`` of R1 or R2
    HOLD: Price is not near any significant pivot level

Touch distance is ATR-adaptive: ``touch_dist = atr_touch_mult × ATR(atr_period)``.
This self-calibrates to current volatility — tight in quiet markets, wider
when volatility expands — eliminating the need to hand-tune a fixed pip count.

Confidence:
    - R2/S2 touches → higher confidence (stronger levels)
    - R1/S1 touches → base confidence
    - Scales down with distance from the level

The "previous session" is approximated by taking the H1 bars from the
preceding 24-hour window.  Works well for daily pivots on H1/H4 data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule
from metatrade.technical_analysis.indicators.atr import atr as compute_atr

_VERSION = ModuleVersion(1, 0, 0)


@dataclass(frozen=True)
class PivotLevels:
    """Classic pivot point levels for a session."""

    pivot: Decimal
    r1: Decimal
    r2: Decimal
    s1: Decimal
    s2: Decimal


def _compute_pivot_levels(high: Decimal, low: Decimal, close: Decimal) -> PivotLevels:
    """Compute classic pivot levels from H/L/C of a session."""
    p = (high + low + close) / Decimal("3")
    r1 = Decimal("2") * p - low
    r2 = p + (high - low)
    s1 = Decimal("2") * p - high
    s2 = p - (high - low)
    return PivotLevels(pivot=p, r1=r1, r2=r2, s1=s1, s2=s2)


class PivotPointsModule(ITechnicalModule):
    """Pivot-point support/resistance signal generator.

    Args:
        session_bars:    Number of bars that make up one "session" (e.g. 24
                         for H1 data → daily pivots).  Default 24.
        atr_period:      ATR period for adaptive touch distance.  Default 14.
        atr_touch_mult:  Touch distance = atr_touch_mult × ATR.  Default 0.5.
                         Replaces the old fixed ``touch_pct`` — self-calibrates
                         to current volatility.
        timeframe_id:    Suffix for ``module_id`` (e.g. "h1").
        confidence_cap:  Maximum confidence any signal can carry (default 0.82).
    """

    def __init__(
        self,
        session_bars: int = 24,
        atr_period: int = 14,
        atr_touch_mult: float = 0.5,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.82,
    ) -> None:
        if session_bars < 2:
            raise ValueError(f"session_bars must be >= 2, got {session_bars}")
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        self._session_bars = session_bars
        self._atr_period = atr_period
        self._atr_touch_mult = atr_touch_mult
        self._timeframe_id = timeframe_id
        self._confidence_cap = confidence_cap

    @property
    def module_id(self) -> str:
        return f"pivot_points_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # Need 2 complete sessions + ATR warmup
        return max(self._session_bars * 2 + 2, self._atr_period + self._session_bars + 2)

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        curr_close = closes[-1]

        # ── ATR for adaptive touch distance ───────────────────────────────────
        atr_series = compute_atr(highs, lows, closes, self._atr_period)
        curr_atr = atr_series[-1]
        touch_dist = curr_atr * Decimal(str(self._atr_touch_mult))

        # ── Find previous session H/L/C ───────────────────────────────────────
        # "previous session" = bars ending session_bars ago
        prev_end = len(bars) - self._session_bars - 1
        prev_start = max(0, prev_end - self._session_bars + 1)
        prev_bars = bars[prev_start : prev_end + 1]

        prev_high = max(b.high for b in prev_bars)
        prev_low = min(b.low for b in prev_bars)
        prev_close = prev_bars[-1].close

        levels = _compute_pivot_levels(prev_high, prev_low, prev_close)

        # ── Check proximity to each level ─────────────────────────────────────
        dist_s2 = abs(curr_close - levels.s2)
        dist_s1 = abs(curr_close - levels.s1)
        dist_r1 = abs(curr_close - levels.r1)
        dist_r2 = abs(curr_close - levels.r2)

        near_s2 = dist_s2 <= touch_dist
        near_s1 = dist_s1 <= touch_dist
        near_r1 = dist_r1 <= touch_dist
        near_r2 = dist_r2 <= touch_dist

        if near_s2 or near_s1:
            direction = SignalDirection.BUY
            level_name = "S2" if near_s2 else "S1"
            level_price = levels.s2 if near_s2 else levels.s1
            dist = dist_s2 if near_s2 else dist_s1
            # S2 gives higher confidence than S1; proximity boosts it further
            base_conf = 0.72 if near_s2 else 0.62
            proximity_bonus = float(1.0 - float(dist / touch_dist)) * 0.10
            confidence = min(self._confidence_cap, base_conf + proximity_bonus)
            reason = (
                f"Price near {level_name}={float(level_price):.5f} "
                f"(dist={float(dist):.5f}): potential support bounce"
            )

        elif near_r2 or near_r1:
            direction = SignalDirection.SELL
            level_name = "R2" if near_r2 else "R1"
            level_price = levels.r2 if near_r2 else levels.r1
            dist = dist_r2 if near_r2 else dist_r1
            base_conf = 0.72 if near_r2 else 0.62
            proximity_bonus = float(1.0 - float(dist / touch_dist)) * 0.10
            confidence = min(self._confidence_cap, base_conf + proximity_bonus)
            reason = (
                f"Price near {level_name}={float(level_price):.5f} "
                f"(dist={float(dist):.5f}): potential resistance rejection"
            )

        else:
            direction = SignalDirection.HOLD
            confidence = 0.50
            reason = (
                f"Price {float(curr_close):.5f} not near pivot levels "
                f"(P={float(levels.pivot):.5f}, "
                f"S1={float(levels.s1):.5f}, R1={float(levels.r1):.5f})"
            )

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "pivot": float(levels.pivot),
                "r1": float(levels.r1),
                "r2": float(levels.r2),
                "s1": float(levels.s1),
                "s2": float(levels.s2),
                "session_high": float(prev_high),
                "session_low": float(prev_low),
                "atr": float(curr_atr),
                "touch_dist": float(touch_dist),
            },
        )
