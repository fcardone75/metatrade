"""Keltner Channel indicator.

    Upper  = EMA(close, ema_period) + multiplier × ATR(atr_period)
    Middle = EMA(close, ema_period)
    Lower  = EMA(close, ema_period) − multiplier × ATR(atr_period)

Key difference vs Bollinger Bands
    Keltner uses ATR for band width (smooth, volatility-representative).
    Bollinger uses std-dev (reacts sharply to single-candle spikes).

The Bollinger/Keltner Squeeze
    When Bollinger Bands contract *inside* the Keltner Channel
    (BB_upper < KC_upper AND BB_lower > KC_lower) the market is coiling —
    a significant directional move is likely imminent.  This is the highest-
    value signal from combining these two volatility envelopes.

Typical parameters
    ema_period=20, atr_period=10, multiplier=2.0  (Raschke original)
    ema_period=20, atr_period=20, multiplier=1.5  (Lazybear variant)
"""

from __future__ import annotations

from decimal import Decimal

from metatrade.technical_analysis.indicators.atr import atr as compute_atr
from metatrade.technical_analysis.indicators.ema import ema as compute_ema


def keltner_channel(
    highs: list[Decimal],
    lows: list[Decimal],
    closes: list[Decimal],
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> tuple[list[Decimal], list[Decimal], list[Decimal]]:
    """Compute Keltner Channel bands.

    Args:
        highs:      Bar high prices (oldest first).
        lows:       Bar low prices.
        closes:     Bar close prices.
        ema_period: Central EMA period (default 20).
        atr_period: ATR period for band width (default 10).
        multiplier: ATR multiplier (default 2.0).

    Returns:
        ``(upper, middle, lower)`` — three lists of Decimal, same length
        as the input series.

    Raises:
        ValueError: If parameters are invalid or data is insufficient.
    """
    if ema_period < 2:
        raise ValueError(f"ema_period must be >= 2, got {ema_period}")
    if atr_period < 1:
        raise ValueError(f"atr_period must be >= 1, got {atr_period}")

    n = len(closes)
    if len(highs) != n or len(lows) != n:
        raise ValueError("highs, lows, closes must have equal length")

    min_needed = max(ema_period, atr_period + 1)
    if n < min_needed:
        raise ValueError(
            f"Need at least {min_needed} bars for Keltner, got {n}"
        )

    middle  = compute_ema(closes, ema_period)
    atr_val = compute_atr(highs, lows, closes, atr_period)
    mult    = Decimal(str(multiplier))

    upper = [m + mult * a for m, a in zip(middle, atr_val)]
    lower = [m - mult * a for m, a in zip(middle, atr_val)]

    return upper, middle, lower


def is_squeeze(
    bb_upper: Decimal,
    bb_lower: Decimal,
    kc_upper: Decimal,
    kc_lower: Decimal,
) -> bool:
    """Return True when Bollinger Bands are inside the Keltner Channel.

    This indicates volatility compression — the market is coiling before a
    directional move.

    Args:
        bb_upper: Current Bollinger Band upper value.
        bb_lower: Current Bollinger Band lower value.
        kc_upper: Current Keltner Channel upper value.
        kc_lower: Current Keltner Channel lower value.
    """
    return bb_upper < kc_upper and bb_lower > kc_lower
