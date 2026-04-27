"""Triple barrier labeling — path-aware label generation.

Instead of looking only at the closing price after N bars (as ``label_bars()``
does), the triple barrier method checks whether price touches an upper or lower
ATR-based barrier *at any point* during the forward window (using bar highs and
lows).  If neither horizontal barrier is touched within ``forward_bars``, the
time barrier fires and the label is HOLD.

Label scheme:
    1  = BUY  — upper barrier hit first (or tied)
   -1  = SELL — lower barrier hit first
    0  = HOLD — time barrier fires (neither horizontal barrier hit)
    None = insufficient future data OR ATR warmup period

Spread / cost proxy:
    When ``spread_pct > 0``, the effective barriers are tightened by the
    estimated round-trip cost, making directional labels more conservative:
        upper_barrier = entry + ATR * atr_mult - entry * spread_pct
        lower_barrier = entry - ATR * atr_mult + entry * spread_pct
    Trades that barely clear the barrier are demoted to HOLD.
"""

from __future__ import annotations

from metatrade.core.contracts.market import Bar
from metatrade.technical_analysis.indicators.atr import atr as compute_atr


class TripleBarrierLabeler:
    """Path-aware labeler using the triple barrier method.

    Args:
        forward_bars: Maximum bars to look ahead (time barrier).
        atr_period:   ATR period (default 14).
        atr_mult:     ATR multiplier for horizontal barrier distance.
        spread_pct:   Estimated round-trip cost as fraction of price (e.g.
                      0.0001 = 1 pip on EURUSD).  Tightens barriers by this
                      amount.  Default 0.0 (no cost proxy).
    """

    def __init__(
        self,
        forward_bars: int = 5,
        atr_period: int = 14,
        atr_mult: float = 1.5,
        spread_pct: float = 0.0,
    ) -> None:
        if forward_bars < 1:
            raise ValueError(f"forward_bars must be >= 1, got {forward_bars}")
        if atr_period < 2:
            raise ValueError(f"atr_period must be >= 2, got {atr_period}")
        if atr_mult <= 0:
            raise ValueError(f"atr_mult must be > 0, got {atr_mult}")
        if spread_pct < 0:
            raise ValueError(f"spread_pct must be >= 0, got {spread_pct}")
        self._forward_bars = forward_bars
        self._atr_period = atr_period
        self._atr_mult = atr_mult
        self._spread_pct = spread_pct

    @property
    def forward_bars(self) -> int:
        return self._forward_bars

    @property
    def atr_period(self) -> int:
        return self._atr_period

    def label(self, bars: list[Bar]) -> list[int | None]:
        """Compute triple barrier labels for every bar in the series.

        Args:
            bars: Full bar series, oldest first.

        Returns:
            List of labels aligned to ``bars``:
                1  = BUY
               -1  = SELL
                0  = HOLD
                None = ATR warmup period or insufficient future data.
        """
        n = len(bars)
        min_atr_bars = self._atr_period + 1

        if n < min_atr_bars + self._forward_bars:
            return [None] * n

        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        try:
            atr_series = compute_atr(highs, lows, closes, self._atr_period)
        except ValueError:
            return [None] * n

        labels: list[int | None] = []

        for i in range(n):
            # ATR warmup: not enough history
            if i < min_atr_bars:
                labels.append(None)
                continue

            # Time barrier: not enough future data
            if i + self._forward_bars >= n:
                labels.append(None)
                continue

            entry = float(closes[i])
            atr_val = float(atr_series[i])
            cost = entry * self._spread_pct

            upper = entry + atr_val * self._atr_mult - cost
            lower = entry - atr_val * self._atr_mult + cost

            label = 0  # default: time barrier (HOLD)

            for j in range(i + 1, i + self._forward_bars + 1):
                bar_high = float(bars[j].high)
                bar_low = float(bars[j].low)

                upper_hit = bar_high >= upper
                lower_hit = bar_low <= lower

                if upper_hit and lower_hit:
                    # Both barriers hit on the same bar.
                    # Assign to the side with the larger excursion from entry.
                    if (bar_high - entry) >= (entry - bar_low):
                        label = 1
                    else:
                        label = -1
                    break
                elif upper_hit:
                    label = 1
                    break
                elif lower_hit:
                    label = -1
                    break

            labels.append(label)

        return labels
