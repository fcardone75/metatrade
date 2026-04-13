"""Relative Strength Index (RSI) indicator.

Standard 14-period RSI using Wilder's smoothing (equivalent to EMA with
alpha = 1/period). Values range from 0 to 100.

Typical thresholds:
    > 70 → overbought (bearish signal)
    < 30 → oversold (bullish signal)
"""

from __future__ import annotations

from decimal import Decimal


def rsi(prices: list[Decimal], period: int = 14) -> list[Decimal]:
    """Compute RSI values for a price series.

    Args:
        prices: Ordered close prices (oldest first).
        period: RSI period (default 14).

    Returns:
        List of RSI values aligned with prices.
        The first `period` entries are None-padded with Decimal("50") as
        a neutral placeholder to maintain list length.

    Raises:
        ValueError: If period < 2 or insufficient prices.
    """
    if period < 2:
        raise ValueError(f"RSI period must be >= 2, got {period}")
    if len(prices) < period + 1:
        raise ValueError(
            f"Need at least {period + 1} prices for RSI(period={period}), "
            f"got {len(prices)}"
        )

    # Compute price changes
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]

    gains = [max(Decimal("0"), c) for c in changes]
    losses = [max(Decimal("0"), -c) for c in changes]

    # Initial averages (SMA over first period)
    avg_gain = sum(gains[:period]) / Decimal(period)
    avg_loss = sum(losses[:period]) / Decimal(period)

    result: list[Decimal] = [Decimal("50")] * (period + 1)  # padding for alignment

    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / Decimal(period)
        avg_loss = (avg_loss * (period - 1) + losses[i]) / Decimal(period)
        if avg_loss == 0:
            result.append(Decimal("100"))
        else:
            rs = avg_gain / avg_loss
            result.append(Decimal("100") - Decimal("100") / (1 + rs))

    return result


def rsi_signal(prices: list[Decimal], period: int = 14,
               oversold: float = 30.0, overbought: float = 70.0) -> int:
    """Return signal based on latest RSI value.

    Returns:
         1  = oversold (potential BUY)
        -1  = overbought (potential SELL)
         0  = neutral
    """
    values = rsi(prices, period)
    latest = values[-1]
    if latest < Decimal(str(oversold)):
        return 1
    if latest > Decimal(str(overbought)):
        return -1
    return 0
