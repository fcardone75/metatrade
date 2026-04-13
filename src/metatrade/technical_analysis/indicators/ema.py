"""Exponential Moving Average (EMA) indicator."""

from __future__ import annotations

from decimal import Decimal


def ema(prices: list[Decimal], period: int) -> list[Decimal]:
    """Compute EMA over a list of prices.

    Args:
        prices: Ordered price values (oldest first).
        period: EMA lookback period.

    Returns:
        List of EMA values, same length as prices.
        Values at indices < period-1 are bootstrapped from SMA.

    Raises:
        ValueError: If period < 1 or prices is empty.
    """
    if period < 1:
        raise ValueError(f"EMA period must be >= 1, got {period}")
    if not prices:
        raise ValueError("prices list is empty")
    if len(prices) < period:
        raise ValueError(
            f"Need at least {period} prices for period={period}, got {len(prices)}"
        )

    k = Decimal(2) / Decimal(period + 1)
    # Bootstrap: first EMA value is the SMA of the first `period` prices
    result: list[Decimal] = []
    first_sma = sum(prices[:period]) / Decimal(period)
    result.extend([first_sma] * period)

    for i in range(period, len(prices)):
        prev = result[-1]
        result.append(prices[i] * k + prev * (1 - k))

    return result


def ema_crossover_signal(
    prices: list[Decimal],
    fast_period: int,
    slow_period: int,
) -> int:
    """Determine EMA crossover signal using the last two bars.

    Returns:
         1 = bullish crossover (fast crossed above slow)
        -1 = bearish crossover (fast crossed below slow)
         0 = no crossover
    """
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
        )
    fast = ema(prices, fast_period)
    slow = ema(prices, slow_period)

    # Need at least 2 points for crossover detection
    if len(fast) < 2:
        return 0

    prev_diff = fast[-2] - slow[-2]
    curr_diff = fast[-1] - slow[-1]

    if prev_diff <= 0 and curr_diff > 0:
        return 1   # bullish: fast crossed above slow
    if prev_diff >= 0 and curr_diff < 0:
        return -1  # bearish: fast crossed below slow
    return 0
