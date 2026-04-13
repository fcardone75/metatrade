"""Stochastic RSI indicator.

Stochastic RSI applies the Stochastic formula to RSI values rather than to
raw price.  This makes it more sensitive and faster to react than plain RSI,
while remaining bounded in [0, 1].

Algorithm:
    1. Compute RSI(``rsi_period``) on close prices.
    2. Apply the Stochastic formula over ``stoch_period`` RSI values:
           %K_raw = (RSI[t] - min(RSI, stoch_period)) /
                    (max(RSI, stoch_period) - min(RSI, stoch_period) + ε)
    3. Smooth %K_raw with SMA(``smooth_k``) → %K line.
    4. Smooth %K with SMA(``smooth_d``) → %D line.

Interpretation:
    %K < 0.20  → oversold zone
    %K > 0.80  → overbought zone
    %K crosses above %D from below 0.20 → BUY signal
    %K crosses below %D from above 0.80 → SELL signal
    Values in [0.20, 0.80] → neutral / trending

Compared with plain RSI:
    - StochRSI oscillates more frequently (faster entry/exit signals)
    - Better at capturing short-term momentum turns
    - More susceptible to noise in very volatile markets
"""

from __future__ import annotations

from decimal import Decimal

from metatrade.technical_analysis.indicators.rsi import rsi as compute_rsi


def _sma(values: list[Decimal], period: int) -> list[Decimal]:
    """Simple Moving Average, front-padded with the first real value."""
    n = len(values)
    result: list[Decimal] = []
    for i in range(n):
        if i < period - 1:
            result.append(Decimal("0"))
            continue
        window = values[i - period + 1 : i + 1]
        result.append(sum(window) / Decimal(period))
    # Back-fill padding
    first_real = period - 1
    if first_real < n:
        for i in range(first_real):
            result[i] = result[first_real]
    return result


def stochastic_rsi(
    closes: list[Decimal],
    rsi_period: int = 14,
    stoch_period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> tuple[list[Decimal], list[Decimal]]:
    """Compute Stochastic RSI (%K and %D) for a price series.

    Args:
        closes:       Close prices, oldest first.
        rsi_period:   Period for the underlying RSI (default 14).
        stoch_period: Look-back for the Stochastic min/max window (default 14).
        smooth_k:     SMA smoothing applied to raw %K (default 3).
        smooth_d:     SMA smoothing applied to %K to produce %D (default 3).

    Returns:
        ``(k_line, d_line)`` — two lists of Decimal values in [0, 1], aligned
        with ``closes``.  Warm-up values are front-padded with 0.5 (neutral).

    Raises:
        ValueError: If inputs are too short or periods are invalid.
    """
    if rsi_period < 2:
        raise ValueError(f"rsi_period must be >= 2, got {rsi_period}")
    if stoch_period < 2:
        raise ValueError(f"stoch_period must be >= 2, got {stoch_period}")
    min_len = rsi_period + stoch_period + smooth_k + smooth_d + 2
    if len(closes) < min_len:
        raise ValueError(
            f"Need at least {min_len} closes for StochRSI configuration, "
            f"got {len(closes)}"
        )

    # ── Step 1: RSI series ────────────────────────────────────────────────────
    rsi_vals = compute_rsi(closes, rsi_period)  # len == len(closes)

    # ── Step 2: Stochastic of RSI ─────────────────────────────────────────────
    eps = Decimal("1e-10")
    k_raw: list[Decimal] = []
    warmup_end = rsi_period + stoch_period - 1  # first index with a full window

    for i in range(len(rsi_vals)):
        if i < warmup_end:
            k_raw.append(Decimal("0.5"))
            continue
        window = rsi_vals[i - stoch_period + 1 : i + 1]
        lo = min(window)
        hi = max(window)
        rng = hi - lo
        if rng < eps:
            k_raw.append(Decimal("0.5"))
        else:
            raw = (rsi_vals[i] - lo) / rng
            k_raw.append(max(Decimal("0"), min(Decimal("1"), raw)))

    # ── Step 3: Smooth %K ─────────────────────────────────────────────────────
    k_smooth = _sma(k_raw, smooth_k)

    # ── Step 4: %D = SMA of smoothed %K ──────────────────────────────────────
    d_line = _sma(k_smooth, smooth_d)

    return k_smooth, d_line
