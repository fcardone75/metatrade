"""Average True Range (ATR) — volatility indicator.

ATR measures market volatility as the average of True Range over N periods.
It is used for:
    - Dynamic stop-loss placement (e.g. SL = entry ± 2*ATR)
    - Position sizing adjustments
    - Regime detection (high vs low volatility)

True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
ATR = Wilder's smoothed average of TR (equivalent to EMA with alpha=1/period)
"""

from __future__ import annotations

from decimal import Decimal


def true_range(
    high: Decimal,
    low: Decimal,
    prev_close: Decimal,
) -> Decimal:
    """Compute the True Range for a single bar.

    Args:
        high:       Current bar high.
        low:        Current bar low.
        prev_close: Previous bar close.

    Returns:
        True Range value (always >= 0).
    """
    return max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close),
    )


def atr(
    highs: list[Decimal],
    lows: list[Decimal],
    closes: list[Decimal],
    period: int = 14,
) -> list[Decimal]:
    """Compute ATR series using Wilder's smoothing.

    Args:
        highs:  List of bar high prices (oldest first).
        lows:   List of bar low prices.
        closes: List of bar close prices.
        period: ATR period (default 14).

    Returns:
        List of ATR values aligned to the input series.
        First `period` values are bootstrapped from simple average of TR.

    Raises:
        ValueError: If inputs have different lengths or insufficient data.
    """
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows, closes must have equal length")
    n = len(highs)
    if n < period + 1:
        raise ValueError(
            f"Need at least {period + 1} bars for ATR(period={period}), got {n}"
        )

    # True ranges (starting from bar 1, needing prev_close)
    trs = [
        true_range(highs[i], lows[i], closes[i - 1])
        for i in range(1, n)
    ]

    # Bootstrap: first ATR = SMA of first `period` TRs
    first_atr = sum(trs[:period]) / Decimal(period)
    result: list[Decimal] = [first_atr] * (period + 1)  # pad to align with input

    for i in range(period, len(trs)):
        prev = result[-1]
        result.append((prev * (period - 1) + trs[i]) / Decimal(period))

    return result


def atr_stop_loss(
    entry_price: Decimal,
    atr_value: Decimal,
    multiplier: float = 2.0,
    side_is_buy: bool = True,
) -> Decimal:
    """Compute an ATR-based stop-loss price.

    Args:
        entry_price: Trade entry price.
        atr_value:   Current ATR value.
        multiplier:  ATR multiple for SL distance (default 2.0).
        side_is_buy: True for long (SL below entry), False for short (SL above).

    Returns:
        Stop-loss price.
    """
    distance = atr_value * Decimal(str(multiplier))
    if side_is_buy:
        return entry_price - distance
    return entry_price + distance
