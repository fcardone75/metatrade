"""Donchian Channel indicator.

    Upper  = max(high[i − period + 1 … i])
    Lower  = min(low[i − period + 1 … i])
    Middle = (Upper + Lower) / 2

Named after Richard Donchian, the channel is the basis of the original
Turtle Trading system (20-period breakout entry, 10-period breakout exit).

Signal semantics (checked against the *previous* bar's channel to avoid
intra-bar look-ahead):
    Close breaks above prior Upper → BUY  (new N-period high)
    Close breaks below prior Lower → SELL (new N-period low)
    Otherwise                      → HOLD

At H1 timeframe:
    period=20 ≈ previous trading day's high/low range
    period=55 ≈ the original Turtle long-term trend channel
"""

from __future__ import annotations

from decimal import Decimal


def donchian_channel(
    highs: list[Decimal],
    lows: list[Decimal],
    period: int = 20,
) -> tuple[list[Decimal], list[Decimal], list[Decimal]]:
    """Compute Donchian Channel bands.

    Args:
        highs:  Bar high prices (oldest first).
        lows:   Bar low prices.
        period: Rolling look-back window (default 20).

    Returns:
        ``(upper, middle, lower)`` — three lists of Decimal, same length
        as the input.  Values before the first complete window are
        back-filled with the first real value.

    Raises:
        ValueError: If ``period < 2``, inputs are mismatched, or
                    insufficient data.
    """
    if period < 2:
        raise ValueError(f"Donchian period must be >= 2, got {period}")
    n = len(highs)
    if len(lows) != n:
        raise ValueError("highs and lows must have equal length")
    if n < period:
        raise ValueError(
            f"Need at least {period} bars for Donchian(period={period}), got {n}"
        )

    upper: list[Decimal] = [Decimal("0")] * n
    lower: list[Decimal] = [Decimal("0")] * n

    for i in range(period - 1, n):
        upper[i] = max(highs[i - period + 1 : i + 1])
        lower[i] = min(lows[i - period + 1 : i + 1])

    # Back-fill warm-up
    first = period - 1
    for i in range(first):
        upper[i] = upper[first]
        lower[i] = lower[first]

    middle = [(u + lo) / Decimal("2") for u, lo in zip(upper, lower)]
    return upper, middle, lower
