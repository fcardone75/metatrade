"""EMA/HMA Crossover analysis module.

Generates BUY/SELL/HOLD signals based on fast/slow moving-average crossovers.

By default the *fast* line is an EMA and the *slow* line is an HMA:

    Fast = EMA(9)    — reactive entry trigger
    Slow = HMA(21)   — near-zero-lag trend direction filter

Using the Hull Moving Average for the slow line eliminates ~50 % of the lag
present in a standard EMA(21) without introducing additional noise, reducing
false crossovers in ranging markets compared to a plain EMA/EMA crossover.

The original plain-EMA behaviour is preserved via ``use_hma_slow=False`` for
backwards-compatibility and testing.

Confidence scales with the normalised gap between the two lines:
    gap_pct = |fast − slow| / price
    confidence = min(cap, gap_pct / 0.001)
"""

from __future__ import annotations

import math
from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule
from metatrade.technical_analysis.indicators.ema import ema as compute_ema
from metatrade.technical_analysis.indicators.hma import hma as compute_hma

_VERSION = ModuleVersion(2, 0, 0)  # bumped: slow line upgraded EMA → HMA


class EmaCrossoverModule(ITechnicalModule):
    """Fast EMA / slow HMA crossover signal generator.

    Args:
        fast_period:   Fast EMA period (default 9).
        slow_period:   Slow HMA period (default 21).
        use_hma_slow:  If True (default) uses HMA for the slow line.
                       Set False to fall back to plain EMA for both lines.
        timeframe_id:  String suffix for ``module_id``.
        confidence_cap: Maximum confidence a crossover signal can emit.
    """

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        use_hma_slow: bool = True,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.85,
    ) -> None:
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
            )
        self._fast            = fast_period
        self._slow            = slow_period
        self._use_hma         = use_hma_slow
        self._timeframe_id    = timeframe_id
        self._confidence_cap  = confidence_cap

    @property
    def module_id(self) -> str:
        return f"ema_crossover_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        if self._use_hma:
            # HMA(n) needs n + int(sqrt(n)) bars
            return self._slow + max(2, int(math.sqrt(self._slow))) + 2
        return self._slow + 2

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)
        closes = [b.close for b in bars]

        fast_line = compute_ema(closes, self._fast)
        slow_line = (
            compute_hma(closes, self._slow)
            if self._use_hma
            else compute_ema(closes, self._slow)
        )

        slow_label = f"HMA({self._slow})" if self._use_hma else f"EMA({self._slow})"

        prev_diff = fast_line[-2] - slow_line[-2]
        curr_diff = fast_line[-1] - slow_line[-1]

        # ── Crossover detection ───────────────────────────────────────────────
        if prev_diff <= 0 and curr_diff > 0:
            direction = SignalDirection.BUY
            reason = (
                f"EMA({self._fast}) crossed above {slow_label}: "
                f"diff={float(curr_diff):.5f}"
            )
        elif prev_diff >= 0 and curr_diff < 0:
            direction = SignalDirection.SELL
            reason = (
                f"EMA({self._fast}) crossed below {slow_label}: "
                f"diff={float(curr_diff):.5f}"
            )
        else:
            direction = SignalDirection.HOLD
            reason = (
                f"No crossover: EMA({self._fast}) vs {slow_label} "
                f"diff={float(curr_diff):.5f}"
            )

        # ── Confidence from normalised gap ────────────────────────────────────
        price_scale = float(closes[-1]) if closes[-1] > 0 else 1.0
        gap_pct     = abs(float(curr_diff)) / price_scale
        confidence  = min(self._confidence_cap, gap_pct / 0.001)
        if direction == SignalDirection.HOLD:
            confidence = max(0.50, confidence)

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "fast_ema":  float(fast_line[-1]),
                "slow_ma":   float(slow_line[-1]),
                "slow_type": "hma" if self._use_hma else "ema",
                "gap_pct":   gap_pct,
            },
        )
