"""Bollinger Bands indicator.

Bollinger Bands place two standard-deviation envelopes around a simple moving
average.  They adapt to volatility: bands widen during high volatility and
contract during low volatility.

    middle = SMA(close, period)
    upper  = middle + num_std × σ(close, period)
    lower  = middle - num_std × σ(close, period)

Typical interpretation
    - Price touches / crosses below lower band → potential oversold bounce (BUY)
    - Price touches / crosses above upper band → potential overbought rejection (SELL)
    - Band width (upper − lower) / middle → normalized volatility proxy
    - %B = (price − lower) / (upper − lower) → position within bands [0, 1]
"""

from __future__ import annotations

import math
from decimal import Decimal


def bollinger_bands(
    closes: list[Decimal],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[list[Decimal], list[Decimal], list[Decimal]]:
    """Compute Bollinger Bands over a price series.

    Args:
        closes:  Close prices, oldest first.
        period:  Look-back period for the moving average (default 20).
        num_std: Number of standard deviations for the bands (default 2.0).

    Returns:
        ``(upper, middle, lower)`` — three lists of the same length as
        ``closes``.  Values at indices < ``period - 1`` are filled with the
        first fully-formed value (edge-pad) so the returned lists are always
        aligned with the input.

    Raises:
        ValueError: If ``period < 2`` or ``closes`` has fewer than ``period``
                    elements.
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    n = len(closes)
    if n < period:
        raise ValueError(
            f"Need at least {period} closes for period={period}, got {n}"
        )

    k = Decimal(str(num_std))
    upper: list[Decimal] = []
    middle: list[Decimal] = []
    lower: list[Decimal] = []

    for i in range(n):
        if i < period - 1:
            # Not enough bars yet — use the first window once it's available
            # (will be back-filled below)
            upper.append(Decimal("0"))
            middle.append(Decimal("0"))
            lower.append(Decimal("0"))
            continue

        window = closes[i - period + 1 : i + 1]
        sma = sum(window) / Decimal(period)
        variance = sum((p - sma) ** 2 for p in window) / Decimal(period)
        std = Decimal(str(math.sqrt(float(variance))))

        upper.append(sma + k * std)
        middle.append(sma)
        lower.append(sma - k * std)

    # Back-fill the warm-up period with the first real value
    first_real = period - 1
    for i in range(first_real):
        upper[i] = upper[first_real]
        middle[i] = middle[first_real]
        lower[i] = lower[first_real]

    return upper, middle, lower


def percent_b(
    price: Decimal,
    upper: Decimal,
    lower: Decimal,
) -> float:
    """Compute %B — the position of ``price`` within the Bollinger Bands.

    Returns:
        Float in [0, 1] where 0 = at lower band, 1 = at upper band.
        Values outside [0, 1] indicate the price is outside the bands.
        Returns 0.5 if bands have zero width.
    """
    band_width = upper - lower
    if band_width == 0:
        return 0.5
    return float((price - lower) / band_width)


def band_width(
    upper: Decimal,
    lower: Decimal,
    middle: Decimal,
) -> float:
    """Normalised band width: (upper − lower) / middle.

    A proxy for volatility relative to price level.
    Returns 0.0 if ``middle`` is zero.
    """
    if middle == 0:
        return 0.0
    return float((upper - lower) / middle)
