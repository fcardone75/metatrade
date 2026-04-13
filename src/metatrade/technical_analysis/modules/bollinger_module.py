"""Bollinger Bands analysis module.

Generates mean-reversion BUY/SELL signals when price touches or crosses the
Bollinger Band extremes, then begins to recover toward the middle band.

Signal logic:
    BUY:  Previous close was at or below the lower band AND current close
          has risen back above it — oversold bounce.
    SELL: Previous close was at or above the upper band AND current close
          has fallen back below it — overbought rejection.
    HOLD: Price is within the bands (no extreme touch detected).

Confidence scales with how far outside the bands the previous bar was
(deeper touch → higher confidence) and is capped at ``confidence_cap``.
"""

from __future__ import annotations

from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule
from metatrade.technical_analysis.indicators.bollinger import (
    bollinger_bands,
    percent_b,
    band_width,
)

_VERSION = ModuleVersion(1, 0, 0)


class BollingerBandsModule(ITechnicalModule):
    """Bollinger Bands mean-reversion signal generator.

    Args:
        period:         Moving average and standard-deviation look-back (default 20).
        num_std:        Band width in standard deviations (default 2.0).
        timeframe_id:   Suffix appended to ``module_id`` (e.g. "h1").
        confidence_cap: Maximum confidence any signal can carry (default 0.85).
    """

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.85,
    ) -> None:
        if period < 2:
            raise ValueError(f"period must be >= 2, got {period}")
        if num_std <= 0:
            raise ValueError(f"num_std must be > 0, got {num_std}")
        self._period = period
        self._num_std = num_std
        self._timeframe_id = timeframe_id
        self._confidence_cap = confidence_cap

    @property
    def module_id(self) -> str:
        return f"bollinger_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # Need period bars for first BB value, plus 1 extra for crossover detection
        return self._period + 1

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        closes = [b.close for b in bars]
        upper, mid, lower = bollinger_bands(closes, self._period, self._num_std)

        curr_close = closes[-1]
        prev_close = closes[-2]
        curr_upper = upper[-1]
        curr_lower = lower[-1]
        curr_mid = mid[-1]
        prev_upper = upper[-2]
        prev_lower = lower[-2]

        pct_b = percent_b(curr_close, curr_upper, curr_lower)
        bw = band_width(curr_upper, curr_lower, curr_mid)

        # ── Signal detection ──────────────────────────────────────────────────
        if prev_close <= prev_lower and curr_close > curr_lower:
            # Oversold bounce: price was at/below lower band, now recovering
            direction = SignalDirection.BUY
            # Depth of the touch drives confidence: how far below the band was prev_close?
            depth = max(0.0, float(prev_lower - prev_close))
            atr_proxy = float(curr_lower) * 0.001  # ~0.1% of price as scale
            depth_ratio = depth / atr_proxy if atr_proxy > 0 else 0.0
            confidence = min(self._confidence_cap, 0.55 + depth_ratio * 0.30)
            reason = (
                f"BB({self._period},{self._num_std}) oversold bounce: "
                f"prev_close={float(prev_close):.5f} ≤ lower={float(prev_lower):.5f}, "
                f"now {float(curr_close):.5f} > {float(curr_lower):.5f}"
            )

        elif prev_close >= prev_upper and curr_close < curr_upper:
            # Overbought rejection: price was at/above upper band, now dropping
            direction = SignalDirection.SELL
            depth = max(0.0, float(prev_close - prev_upper))
            atr_proxy = float(curr_upper) * 0.001
            depth_ratio = depth / atr_proxy if atr_proxy > 0 else 0.0
            confidence = min(self._confidence_cap, 0.55 + depth_ratio * 0.30)
            reason = (
                f"BB({self._period},{self._num_std}) overbought rejection: "
                f"prev_close={float(prev_close):.5f} ≥ upper={float(prev_upper):.5f}, "
                f"now {float(curr_close):.5f} < {float(curr_upper):.5f}"
            )

        else:
            direction = SignalDirection.HOLD
            confidence = 0.50
            reason = (
                f"BB({self._period},{self._num_std}) price within bands: "
                f"%B={pct_b:.2f}  bw={bw:.4f}"
            )

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "upper": float(curr_upper),
                "middle": float(curr_mid),
                "lower": float(curr_lower),
                "percent_b": round(pct_b, 4),
                "band_width": round(bw, 6),
            },
        )
