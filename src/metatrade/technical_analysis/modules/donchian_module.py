"""Donchian Channel breakout module.

Signals a breakout when the current close exceeds the N-period price range.

Logic (checked against the *previous* bar's channel to avoid intra-bar
look-ahead — the channel is computed without including the current bar's
high/low):

    Close > Donchian_upper[-2]  →  BUY   (new N-period high breakout)
    Close < Donchian_lower[-2]  →  SELL  (new N-period low breakout)
    Otherwise                   →  HOLD

At H1 timeframe:
    period=20 captures approximately the previous 20-hour price range
             (slightly more than one trading day) — the classic Turtle entry.
    period=55 captures the long-term Turtle channel.

Confidence
    Scales with how far beyond the channel boundary the close is:
    depth = (close − upper) / channel_width  →  0 at boundary, 1 at full width
    confidence = 0.60 + depth × 0.20, capped at confidence_cap.
"""

from __future__ import annotations

from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.indicators.donchian import donchian_channel
from metatrade.technical_analysis.interface import ITechnicalModule

_VERSION = ModuleVersion(1, 0, 0)


class DonchianBreakoutModule(ITechnicalModule):
    """N-period Donchian Channel breakout detector.

    Args:
        period:        Look-back window in bars (default 20).
        timeframe_id:  Suffix for ``module_id``.
        confidence_cap: Maximum confidence for a breakout signal.
    """

    def __init__(
        self,
        period: int = 20,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.82,
    ) -> None:
        if period < 2:
            raise ValueError(f"Donchian period must be >= 2, got {period}")
        self._period        = period
        self._timeframe_id  = timeframe_id
        self._confidence_cap = confidence_cap

    @property
    def module_id(self) -> str:
        return f"donchian_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # Need period + 1 so that bars[-2] is a fully-formed channel value
        return self._period + 2

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        highs  = [b.high for b in bars]
        lows   = [b.low  for b in bars]
        closes = [b.close for b in bars]

        upper, middle, lower = donchian_channel(highs, lows, self._period)

        curr_close = closes[-1]

        # Use previous bar's channel values to avoid look-ahead
        prev_upper = upper[-2]
        prev_lower = lower[-2]
        prev_width = float(prev_upper - prev_lower)

        # ── Signal logic ─────────────────────────────────────────────────────
        if curr_close > prev_upper:
            overshoot  = float(curr_close - prev_upper)
            depth      = min(1.0, overshoot / prev_width) if prev_width > 0 else 0.0
            confidence = min(self._confidence_cap, 0.60 + depth * 0.20)
            direction  = SignalDirection.BUY
            reason = (
                f"Donchian({self._period}): bullish breakout "
                f"(close={float(curr_close):.5f} > upper={float(prev_upper):.5f})"
            )

        elif curr_close < prev_lower:
            overshoot  = float(prev_lower - curr_close)
            depth      = min(1.0, overshoot / prev_width) if prev_width > 0 else 0.0
            confidence = min(self._confidence_cap, 0.60 + depth * 0.20)
            direction  = SignalDirection.SELL
            reason = (
                f"Donchian({self._period}): bearish breakout "
                f"(close={float(curr_close):.5f} < lower={float(prev_lower):.5f})"
            )

        else:
            direction  = SignalDirection.HOLD
            confidence = 0.50
            reason = (
                f"Donchian({self._period}): inside channel "
                f"[{float(prev_lower):.5f}–{float(prev_upper):.5f}]"
            )

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "upper":  float(prev_upper),
                "lower":  float(prev_lower),
                "middle": float(middle[-2]),
                "close":  float(curr_close),
                "width":  round(prev_width, 6),
            },
        )
