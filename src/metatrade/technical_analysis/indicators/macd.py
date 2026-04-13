"""MACD — Moving Average Convergence Divergence.

Standard parameters: fast=12, slow=26, signal=9.

Components:
    macd_line    = EMA(fast) - EMA(slow)
    signal_line  = EMA(macd_line, signal_period)
    histogram    = macd_line - signal_line

Signal logic:
    MACD line crosses above signal line → bullish (BUY)
    MACD line crosses below signal line → bearish (SELL)
"""

from __future__ import annotations

from decimal import Decimal

from metatrade.technical_analysis.indicators.ema import ema


def macd(
    prices: list[Decimal],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[Decimal], list[Decimal], list[Decimal]]:
    """Compute MACD, signal, and histogram.

    Args:
        prices: Close prices (oldest first).
        fast:   Fast EMA period (default 12).
        slow:   Slow EMA period (default 26).
        signal: Signal line EMA period (default 9).

    Returns:
        (macd_line, signal_line, histogram) — all aligned to len(prices).
        Values before the slow EMA warms up use Decimal("0").

    Raises:
        ValueError: If fast >= slow or insufficient prices.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    min_required = slow + signal - 1
    if len(prices) < min_required:
        raise ValueError(
            f"Need at least {min_required} prices for MACD({fast},{slow},{signal}), "
            f"got {len(prices)}"
        )

    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)

    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    sig_line = ema(macd_line, signal)
    histogram = [m - s for m, s in zip(macd_line, sig_line)]

    return macd_line, sig_line, histogram


def macd_signal(
    prices: list[Decimal],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> int:
    """Return crossover signal from the latest two MACD bars.

    Returns:
         1 = bullish (MACD crossed above signal)
        -1 = bearish (MACD crossed below signal)
         0 = no crossover
    """
    macd_line, sig_line, _ = macd(prices, fast, slow, signal_period)
    prev_diff = macd_line[-2] - sig_line[-2]
    curr_diff = macd_line[-1] - sig_line[-1]
    if prev_diff <= 0 and curr_diff > 0:
        return 1
    if prev_diff >= 0 and curr_diff < 0:
        return -1
    return 0
