"""Hull Moving Average (HMA) indicator.

HMA ≈ eliminates lag from standard EMA/SMA while maintaining smoothness.

Formula (Alan Hull, 2005):
    1. WMA₁ = WMA(closes, n // 2)          -- fast weighted MA
    2. WMA₂ = WMA(closes, n)               -- slow weighted MA
    3. raw[i] = 2 × WMA₁[i] − WMA₂[i]     -- de-lagged series
    4. HMA   = WMA(raw, int(√n))           -- final smoothing pass

Advantages over standard EMA at the same period:
    - ~50 % less lag in trend direction changes.
    - Fewer whipsaws than raw price while still being highly responsive.
    - Suited for trend-entry signals where entry timing matters most.

Minimum bars required: period + int(√period) − 1
"""

from __future__ import annotations

import math
from decimal import Decimal


def wma(prices: list[Decimal], period: int) -> list[Decimal]:
    """Linearly Weighted Moving Average.

    Each price is weighted by its position in the window:
    the most recent bar has the highest weight (``period``), the oldest has 1.

    Returns a list of the same length as ``prices``.  Values at indices before
    the first complete window are back-filled with the first real WMA value.

    Raises:
        ValueError: If ``period < 1`` or not enough prices.
    """
    if period < 1:
        raise ValueError(f"WMA period must be >= 1, got {period}")
    n = len(prices)
    if n < period:
        raise ValueError(
            f"Need at least {period} prices for WMA(period={period}), got {n}"
        )

    denom = Decimal(period * (period + 1) // 2)  # triangular number
    result: list[Decimal] = [Decimal("0")] * n

    for i in range(period - 1, n):
        window = prices[i - period + 1 : i + 1]
        weighted = sum(window[j] * Decimal(j + 1) for j in range(period))
        result[i] = weighted / denom

    # Back-fill warm-up placeholders
    first = period - 1
    for i in range(first):
        result[i] = result[first]

    return result


def hma(prices: list[Decimal], period: int) -> list[Decimal]:
    """Compute the Hull Moving Average.

    Args:
        prices: Close prices, oldest first.
        period: HMA period (commonly 21).  Must be ≥ 4.

    Returns:
        HMA values aligned with input (same length as ``prices``).
        Early warm-up values are back-filled from the first real HMA value.

    Raises:
        ValueError: If ``period < 4`` or insufficient data.
    """
    if period < 4:
        raise ValueError(f"HMA period must be >= 4, got {period}")

    half   = max(2, period // 2)
    sqrt_p = max(2, int(math.sqrt(period)))
    min_needed = period + sqrt_p

    n = len(prices)
    if n < min_needed:
        raise ValueError(
            f"Need at least {min_needed} prices for HMA(period={period}), got {n}"
        )

    # Step 1 & 2 — WMA at half and full period
    wma_half = wma(prices, half)
    wma_full = wma(prices, period)

    # Step 3 — de-lagged raw series
    raw = [
        Decimal("2") * wma_half[i] - wma_full[i]
        for i in range(n)
    ]

    # Step 4 — final smoothing
    result = wma(raw, sqrt_p)
    return result
