"""Continuous target builder for the EV (Expected Value) regressor.

Computes ATR-scaled R-multiple targets from a bar series.  Each target
represents how many R-units the market moved over the next ``forward_bars``
bars, where 1 R = ATR × atr_mult (the default stop-loss distance).

    r_multiple = (close[t + forward_bars] - close[t]) / (ATR(t) × atr_mult)

Positive values indicate bullish movement; negative values indicate bearish.
The regressor is direction-agnostic: the caller decides whether the predicted
R-multiple aligns with the intended trade direction.

Bars at the tail of the series (last ``forward_bars`` positions) cannot have
a forward target and are excluded from ``build()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.technical_analysis.indicators.atr import atr as compute_atr


@dataclass(frozen=True)
class MlContinuousTarget:
    """ATR-scaled R-multiple outcome for a single bar.

    Args:
        bar_index:    Index of the entry bar in the original bar series.
        r_multiple:   Signed R-multiple over the forward window.
                      Positive = price rose, negative = price fell.
        forward_bars: Number of bars used for the forward window.
        atr_value:    ATR value at the entry bar (reference, not used in inference).
    """

    bar_index: int
    r_multiple: float
    forward_bars: int
    atr_value: float


class ContinuousTargetBuilder:
    """Build ``MlContinuousTarget`` labels for every eligible bar in a series.

    Args:
        forward_bars: Bars to look ahead for the outcome (default 5, same as
                      the classifier's default label horizon).
        atr_period:   Period for the ATR calculation (default 14).
        atr_mult:     ATR multiplier defining 1 R unit (default 2.0, matching
                      the Chandelier Exit stop-loss used by the risk manager).
    """

    def __init__(
        self,
        forward_bars: int = 5,
        atr_period: int = 14,
        atr_mult: float = 2.0,
    ) -> None:
        if forward_bars < 1:
            raise ValueError(f"forward_bars must be >= 1, got {forward_bars}")
        if atr_period < 2:
            raise ValueError(f"atr_period must be >= 2, got {atr_period}")
        if atr_mult <= 0:
            raise ValueError(f"atr_mult must be > 0, got {atr_mult}")
        self._forward_bars = forward_bars
        self._atr_period = atr_period
        self._atr_mult = Decimal(str(atr_mult))

    @property
    def forward_bars(self) -> int:
        return self._forward_bars

    @property
    def atr_period(self) -> int:
        return self._atr_period

    def build(self, bars: list[Bar]) -> list[MlContinuousTarget]:
        """Build R-multiple targets for all eligible bars.

        Eligible bars: index ``i`` where ``i + forward_bars < len(bars)`` AND
        the ATR can be computed (requires at least ``atr_period + 1`` bars up
        to and including bar ``i``).

        Args:
            bars: Bar series, oldest first.

        Returns:
            List of ``MlContinuousTarget``, one per eligible bar.  May be
            empty if the series is too short.
        """
        n = len(bars)
        min_bars_for_atr = self._atr_period + 1
        targets: list[MlContinuousTarget] = []

        if n < min_bars_for_atr + self._forward_bars:
            return targets

        # Pre-compute the full ATR series once for efficiency
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        try:
            atr_series = compute_atr(highs, lows, closes, self._atr_period)
        except ValueError:
            return targets

        for i in range(min_bars_for_atr - 1, n - self._forward_bars):
            atr_val = atr_series[i]
            stop_distance = atr_val * self._atr_mult
            if stop_distance == 0:
                continue

            entry = closes[i]
            future = closes[i + self._forward_bars]
            r = float((future - entry) / stop_distance)

            targets.append(
                MlContinuousTarget(
                    bar_index=i,
                    r_multiple=r,
                    forward_bars=self._forward_bars,
                    atr_value=float(atr_val),
                )
            )

        return targets
