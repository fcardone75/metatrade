"""Multi-Timeframe (MTF) alignment module.

The single most common cause of premature entries is trading *against* the
higher-timeframe trend.  This module filters signals by requiring alignment
with the trend on a higher timeframe.

Implementation:
    The module receives lower-timeframe (e.g. H1) bars and internally
    resamples them into higher-timeframe (e.g. H4) bars using standard
    OHLC resampling.  It then computes a fast and a slow EMA on the
    higher-timeframe closes to determine trend direction.

    BUY:  Higher-TF fast EMA > slow EMA  (higher-TF uptrend)
    SELL: Higher-TF fast EMA < slow EMA  (higher-TF downtrend)
    HOLD: EMAs are flat / undecided or insufficient bars

Confidence:
    Scales with the normalised spread between fast and slow EMA:
        spread_pct = (fast - slow) / slow
    Larger spread → stronger trend → higher confidence.

Resampling:
    Given ``resample_factor`` lower-TF bars per higher-TF bar:
        H1 → H4 : resample_factor = 4
        H1 → D1 : resample_factor = 24
    Each group of ``resample_factor`` consecutive bars is collapsed into
    one higher-TF bar (open = first open, high = max high, low = min low,
    close = last close).
"""

from __future__ import annotations

from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.indicators.ema import ema as compute_ema
from metatrade.technical_analysis.interface import ITechnicalModule

_VERSION = ModuleVersion(1, 0, 0)


def _resample_bars(bars: list[Bar], factor: int) -> list[Bar]:
    """Collapse ``bars`` into higher-TF bars by grouping ``factor`` bars each.

    Only complete groups are kept; any trailing partial group is discarded.
    """
    n = len(bars)
    n_complete = n - (n % factor)
    result: list[Bar] = []
    for start in range(0, n_complete, factor):
        group = bars[start : start + factor]
        htf_bar = Bar(
            symbol=group[0].symbol,
            timeframe=group[0].timeframe,
            timestamp_utc=group[0].timestamp_utc,
            open=group[0].open,
            high=max(b.high for b in group),
            low=min(b.low for b in group),
            close=group[-1].close,
            volume=sum(b.volume for b in group),
        )
        result.append(htf_bar)
    return result


class MultiTimeframeModule(ITechnicalModule):
    """Higher-timeframe trend-alignment filter.

    Args:
        resample_factor:  How many lower-TF bars make one higher-TF bar.
                          E.g. 4 for H1→H4, 24 for H1→D1.  Default 4.
        fast_period:      Fast EMA period on the higher-TF bars.  Default 9.
        slow_period:      Slow EMA period on the higher-TF bars.  Default 21.
        timeframe_id:     Suffix for ``module_id`` (e.g. "h1_to_h4").
        confidence_cap:   Maximum confidence any signal can carry.  Default 0.85.
    """

    def __init__(
        self,
        resample_factor: int = 4,
        fast_period: int = 9,
        slow_period: int = 21,
        timeframe_id: str = "h1_to_h4",
        confidence_cap: float = 0.85,
    ) -> None:
        if resample_factor < 2:
            raise ValueError(f"resample_factor must be >= 2, got {resample_factor}")
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
            )
        self._resample_factor = resample_factor
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._timeframe_id = timeframe_id
        self._confidence_cap = confidence_cap

    @property
    def module_id(self) -> str:
        return f"multi_tf_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # Need enough lower-TF bars so that, after resampling, we have at least
        # slow_period + 2 higher-TF bars (extra 2 for crossover detection).
        return (self._slow_period + 2) * self._resample_factor

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        # ── Resample to higher timeframe ──────────────────────────────────────
        htf_bars = _resample_bars(bars, self._resample_factor)

        if len(htf_bars) < self._slow_period + 1:
            # Shouldn't happen given min_bars, but guard anyway
            return AnalysisSignal(
                direction=SignalDirection.HOLD,
                confidence=0.50,
                reason="MultiTF: insufficient higher-TF bars after resampling",
                module_id=self.module_id,
                module_version=_VERSION,
                timestamp_utc=timestamp_utc,
                metadata={"htf_bars": len(htf_bars)},
            )

        htf_closes = [b.close for b in htf_bars]

        # ── EMA on higher-TF closes ───────────────────────────────────────────
        fast_ema = compute_ema(htf_closes, self._fast_period)
        slow_ema = compute_ema(htf_closes, self._slow_period)

        curr_fast = fast_ema[-1]
        curr_slow = slow_ema[-1]

        # Normalised spread gives trend strength
        spread = float(curr_fast - curr_slow)
        spread_pct = spread / float(curr_slow) if float(curr_slow) != 0 else 0.0

        # ── Trend direction ───────────────────────────────────────────────────
        if curr_fast > curr_slow:
            direction = SignalDirection.BUY
            depth = min(1.0, abs(spread_pct) / 0.005)  # 0.5% spread = full depth
            confidence = min(self._confidence_cap, 0.60 + depth * 0.25)
            reason = (
                f"MultiTF({self._timeframe_id}): higher-TF uptrend "
                f"(fast_EMA={float(curr_fast):.5f} > slow_EMA={float(curr_slow):.5f}, "
                f"spread={spread_pct:+.4%})"
            )

        elif curr_fast < curr_slow:
            direction = SignalDirection.SELL
            depth = min(1.0, abs(spread_pct) / 0.005)
            confidence = min(self._confidence_cap, 0.60 + depth * 0.25)
            reason = (
                f"MultiTF({self._timeframe_id}): higher-TF downtrend "
                f"(fast_EMA={float(curr_fast):.5f} < slow_EMA={float(curr_slow):.5f}, "
                f"spread={spread_pct:+.4%})"
            )

        else:
            direction = SignalDirection.HOLD
            confidence = 0.50
            reason = (
                f"MultiTF({self._timeframe_id}): EMAs flat, no clear higher-TF trend"
            )

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "htf_bars": len(htf_bars),
                "fast_ema": float(curr_fast),
                "slow_ema": float(curr_slow),
                "spread_pct": round(spread_pct, 6),
                "resample_factor": self._resample_factor,
            },
        )
