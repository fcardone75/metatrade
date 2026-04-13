"""Hurst Exponent — statistical regime detector via R/S analysis.

Measures the *memory* (persistence) of a price time series:

    H > 0.55  → trending / persistent  — trend-following strategies work
    H < 0.45  → mean-reverting          — oscillator / mean-reversion strategies work
    H ≈ 0.50  → random walk             — no directional edge

Method: Rescaled Range (R/S) analysis across multiple sub-window sizes.
    For each scale τ:
        1. Split the returns series into non-overlapping windows of length τ.
        2. For each window, compute:
               R = max(cumulative_deviation) − min(cumulative_deviation)
               S = std(returns in window)
               RS = R / S  (Rescaled Range)
        3. Average RS across windows at scale τ.
    Hurst = OLS slope of log(avg_RS) vs log(τ).

Computational complexity: O(n) per call with 4–5 scale levels.
Minimum meaningful input: ~30–40 prices.  Returns 0.5 (neutral) when data
is insufficient.

References:
    - Hurst, H.E. (1951) — Long-term storage capacity of reservoirs
    - Mandelbrot & Wallis (1969) — Computer experiments with fractional Gaussian noises
    - Macrosynergy (2023) — "Detecting Trends and Mean Reversion with the Hurst Exponent"
"""

from __future__ import annotations

import math
from decimal import Decimal


def _rs(returns: list[float]) -> float:
    """Compute the R/S (Rescaled Range) for a single return window."""
    n = len(returns)
    if n < 2:
        return 0.0

    mean = sum(returns) / n
    # Mean-centered cumulative deviations
    cumsum: list[float] = []
    total = 0.0
    for r in returns:
        total += r - mean
        cumsum.append(total)

    R = max(cumsum) - min(cumsum)

    # Population standard deviation of returns
    variance = sum((r - mean) ** 2 for r in returns) / n
    S = math.sqrt(variance)

    return R / S if S > 0 else 0.0


def hurst_exponent(
    closes: list[Decimal],
    min_period: int = 8,
) -> float:
    """Estimate the Hurst Exponent using R/S analysis.

    Args:
        closes:     Close prices, oldest first.  At least ``2 * min_period``
                    values are required for a meaningful estimate.
        min_period: Smallest sub-window size to use (default 8).  Scales up
                    by powers of 2: 8, 16, 32, 64, … up to len(closes).

    Returns:
        Float H ∈ [0.10, 0.90]:
            H > 0.55 — trending (trend-following has positive expected value)
            H < 0.45 — mean-reverting (oscillators have positive expected value)
            H ≈ 0.50 — random walk (neutral, no directional edge)

        Returns **0.5** when there is insufficient data or the calculation
        degenerates (e.g. flat price series).
    """
    n = len(closes)
    if n < 2 * min_period:
        return 0.5  # insufficient data → neutral

    # Log returns
    prices = [float(c) for c in closes]
    returns: list[float] = []
    for i in range(1, n):
        p0, p1 = prices[i - 1], prices[i]
        if p0 > 0 and p1 > 0:
            returns.append(math.log(p1 / p0))
        else:
            returns.append(0.0)

    m = len(returns)  # = n - 1

    # Build the set of scales (powers of 2 from min_period up to m)
    scales: list[int] = []
    s = min_period
    while s <= m:
        scales.append(s)
        s *= 2
    # Always include the full series as the largest scale
    if not scales or scales[-1] < m:
        scales.append(m)
    scales = sorted(set(scales))

    log_scales: list[float] = []
    log_rs:     list[float] = []

    for size in scales:
        if size < 4 or size > m:
            continue
        # Average R/S over non-overlapping windows of this size
        rs_list: list[float] = []
        for start in range(0, m - size + 1, size):
            rs = _rs(returns[start : start + size])
            if rs > 0:
                rs_list.append(rs)
        if rs_list:
            avg_rs = sum(rs_list) / len(rs_list)
            log_scales.append(math.log(size))
            log_rs.append(math.log(avg_rs))

    if len(log_scales) < 2:
        return 0.5  # not enough scale points for a regression

    # OLS slope of log(RS) vs log(τ)  →  H
    k     = len(log_scales)
    x_bar = sum(log_scales) / k
    y_bar = sum(log_rs)     / k
    num   = sum((log_scales[i] - x_bar) * (log_rs[i] - y_bar) for i in range(k))
    den   = sum((log_scales[i] - x_bar) ** 2 for i in range(k))

    if den == 0:
        return 0.5

    h = num / den
    # Clamp to a sensible range — theoretical range is (0, 1) but empirically
    # financial series rarely stray far from [0.2, 0.8]
    return max(0.10, min(0.90, h))
