"""Average Directional Index (ADX) indicator.

ADX measures **trend strength**, not direction.  It is used together with the
Directional Indicators (+DI / −DI) which show trend direction.

Algorithm (Wilder, 1978):
    True Range (TR):
        TR = max(high − low, |high − prev_close|, |low − prev_close|)

    Directional Movement:
        +DM = high − prev_high  if  > 0  and  > −(low − prev_low),  else 0
        −DM = −(low − prev_low) if  > 0  and  > (high − prev_high), else 0

    Smoothed with Wilder EMA (alpha = 1/period):
        Smoothed_TR[i]  = Smoothed_TR[i-1]  − Smoothed_TR[i-1]/period  + TR[i]
        Smoothed_+DM[i] = Smoothed_+DM[i-1] − Smoothed_+DM[i-1]/period + +DM[i]
        Smoothed_−DM[i] = Smoothed_−DM[i-1] − Smoothed_−DM[i-1]/period + −DM[i]

    +DI = 100 × Smoothed_+DM / Smoothed_TR
    −DI = 100 × Smoothed_−DM / Smoothed_TR
    DX  = 100 × |+DI − −DI| / (+DI + −DI)
    ADX = Wilder EMA of DX over `period` bars

Interpretation:
    ADX < 20  → ranging / no trend
    ADX 20–25 → trend developing
    ADX > 25  → strong trend (+DI vs −DI shows direction)
    ADX > 40  → very strong trend (often near exhaustion)
"""

from __future__ import annotations

from decimal import Decimal


def adx(
    highs: list[Decimal],
    lows: list[Decimal],
    closes: list[Decimal],
    period: int = 14,
) -> tuple[list[Decimal], list[Decimal], list[Decimal]]:
    """Compute ADX, +DI, and −DI series.

    Args:
        highs:   Bar high prices, oldest first.
        lows:    Bar low prices.
        closes:  Bar close prices.
        period:  Look-back period (default 14, Wilder standard).

    Returns:
        ``(adx_values, plus_di, minus_di)`` — three lists aligned with the
        input.  The first ``period * 2`` values are warm-up padding.

    Raises:
        ValueError: If input lists have different lengths or are too short.
    """
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows, closes must have equal length")
    n = len(highs)
    if n < period * 2 + 1:
        raise ValueError(
            f"Need at least {period * 2 + 1} bars for ADX(period={period}), got {n}"
        )

    p = Decimal(period)

    # ── Step 1: compute raw TR, +DM, −DM from bar 1 onward ───────────────────
    trs: list[Decimal] = []
    plus_dms: list[Decimal] = []
    minus_dms: list[Decimal] = []

    for i in range(1, n):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)

        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        if up_move > down_move and up_move > 0:
            plus_dms.append(up_move)
        else:
            plus_dms.append(Decimal("0"))

        if down_move > up_move and down_move > 0:
            minus_dms.append(down_move)
        else:
            minus_dms.append(Decimal("0"))

    # ── Step 2: Wilder smoothing of TR, +DM, −DM ─────────────────────────────
    # Seed = sum of first `period` values
    smooth_tr = sum(trs[:period])
    smooth_pdm = sum(plus_dms[:period])
    smooth_mdm = sum(minus_dms[:period])

    plus_di_vals: list[Decimal] = []
    minus_di_vals: list[Decimal] = []
    dx_vals: list[Decimal] = []

    def _di(smoothed_dm: Decimal, smoothed_tr: Decimal) -> Decimal:
        if smoothed_tr == 0:
            return Decimal("0")
        return Decimal("100") * smoothed_dm / smoothed_tr

    # First valid DI at index `period` of the raw series
    plus_di_vals.append(_di(smooth_pdm, smooth_tr))
    minus_di_vals.append(_di(smooth_mdm, smooth_tr))

    for i in range(period, len(trs)):
        smooth_tr = smooth_tr - smooth_tr / p + trs[i]
        smooth_pdm = smooth_pdm - smooth_pdm / p + plus_dms[i]
        smooth_mdm = smooth_mdm - smooth_mdm / p + minus_dms[i]
        plus_di_vals.append(_di(smooth_pdm, smooth_tr))
        minus_di_vals.append(_di(smooth_mdm, smooth_tr))

    # ── Step 3: DX from DI values ─────────────────────────────────────────────
    for pdi, mdi in zip(plus_di_vals, minus_di_vals):
        total = pdi + mdi
        if total == 0:
            dx_vals.append(Decimal("0"))
        else:
            dx_vals.append(Decimal("100") * abs(pdi - mdi) / total)

    # ── Step 4: ADX = Wilder EMA of DX over `period` values ──────────────────
    adx_vals: list[Decimal] = []

    seed_adx = sum(dx_vals[:period]) / p
    adx_vals.append(seed_adx)

    for i in range(period, len(dx_vals)):
        prev = adx_vals[-1]
        adx_vals.append((prev * (p - 1) + dx_vals[i]) / p)

    # ── Align all outputs to the original input length ────────────────────────
    # adx_vals[0] corresponds to input index (period*2)
    # Pad the front with zeros
    pad_adx = period * 2
    pad_di = period

    full_adx = [Decimal("0")] * pad_adx + adx_vals
    full_pdi = [Decimal("0")] * (pad_di + 1) + plus_di_vals[1:]
    full_mdi = [Decimal("0")] * (pad_di + 1) + minus_di_vals[1:]

    # Trim or extend to exactly n
    def _align(lst: list[Decimal]) -> list[Decimal]:
        if len(lst) >= n:
            return lst[:n]
        return [Decimal("0")] * (n - len(lst)) + lst

    return _align(full_adx), _align(full_pdi), _align(full_mdi)
