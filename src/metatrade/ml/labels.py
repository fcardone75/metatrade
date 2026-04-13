"""Label generation for supervised ML training.

Generates forward-looking labels from OHLCV data for classification tasks.

Label scheme (3-class):
    1  = BUY  — future return exceeds +threshold
   -1  = SELL — future return drops below -threshold
    0  = HOLD — future return within ±threshold (no-trade zone)

The threshold is expressed in multiples of ATR to make it adaptive
to market volatility (e.g. threshold = 1.5 × ATR14).
"""

from __future__ import annotations

from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.technical_analysis.indicators.atr import atr


def label_bars(
    bars: list[Bar],
    forward_bars: int = 5,
    atr_threshold_mult: float = 1.5,
    atr_period: int = 14,
) -> list[int | None]:
    """Compute forward-return labels for each bar in the series.

    Args:
        bars:               Complete bar series (oldest first).
        forward_bars:       Number of bars to look ahead for label computation.
        atr_threshold_mult: Label threshold as ATR multiple (default 1.5×ATR).
        atr_period:         ATR period (default 14).

    Returns:
        List of labels aligned to bars:
            1  = BUY
           -1  = SELL
            0  = HOLD
            None = insufficient future data (last `forward_bars` entries)

    Raises:
        ValueError: If forward_bars < 1 or atr_threshold_mult <= 0.
    """
    if forward_bars < 1:
        raise ValueError(f"forward_bars must be >= 1, got {forward_bars}")
    if atr_threshold_mult <= 0:
        raise ValueError(f"atr_threshold_mult must be > 0, got {atr_threshold_mult}")

    n = len(bars)
    min_atr_bars = atr_period + 1

    labels: list[int | None] = []

    for i in range(n):
        # Future close is not available for the last `forward_bars` bars
        future_idx = i + forward_bars
        if future_idx >= n:
            labels.append(None)
            continue

        current_close = float(bars[i].close)
        future_close = float(bars[future_idx].close)
        future_return = (future_close / current_close) - 1.0

        # Compute ATR threshold at this bar (adaptive to volatility)
        if i >= min_atr_bars:
            slice_highs = [b.high for b in bars[: i + 1]]
            slice_lows = [b.low for b in bars[: i + 1]]
            slice_closes = [b.close for b in bars[: i + 1]]
            atr_vals = atr(slice_highs, slice_lows, slice_closes, atr_period)
            current_atr = float(atr_vals[-1])
            threshold = (current_atr / current_close) * atr_threshold_mult
        else:
            # Before ATR warmup: use a simple fixed 0.5% threshold
            threshold = 0.005

        if future_return > threshold:
            labels.append(1)
        elif future_return < -threshold:
            labels.append(-1)
        else:
            labels.append(0)

    return labels


def label_single(
    current_close: float,
    future_close: float,
    threshold_pct: float,
) -> int:
    """Compute a single label given current price, future price, and threshold.

    Args:
        current_close:  Current bar close price.
        future_close:   Close price N bars in the future.
        threshold_pct:  Minimum return (as fraction, e.g. 0.005 = 0.5%) to assign
                        a directional label.

    Returns:
        1 = BUY, -1 = SELL, 0 = HOLD
    """
    if threshold_pct < 0:
        raise ValueError(f"threshold_pct must be >= 0, got {threshold_pct}")
    ret = (future_close / current_close) - 1.0
    if ret > threshold_pct:
        return 1
    if ret < -threshold_pct:
        return -1
    return 0
