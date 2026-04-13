"""ADX (Average Directional Index) analysis module.

Emits directional signals only when the market is trending strongly enough
to make trend-following worthwhile.

Signal logic:
    BUY:  ADX ≥ adx_threshold  AND  +DI > −DI  (strong uptrend)
    SELL: ADX ≥ adx_threshold  AND  −DI > +DI  (strong downtrend)
    HOLD: ADX < adx_threshold  (ranging / no clear trend)

Confidence scales with ADX value:
    confidence = min(cap, 0.50 + (ADX − threshold) / 100)
    i.e. an ADX of 50 with threshold=25 gives ≈ 0.75 confidence.

This module complements trend-following modules (EMA crossover, MACD) and
counters mean-reversion modules (RSI, Bollinger Bands) in a ranging market.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule
from metatrade.technical_analysis.indicators.adx import adx as compute_adx

_VERSION = ModuleVersion(1, 0, 0)


class AdxModule(ITechnicalModule):
    """ADX trend-strength signal generator.

    Args:
        period:         ADX/DI look-back period (default 14).
        adx_threshold:  Minimum ADX value to emit a directional signal (default 25).
        timeframe_id:   Suffix for ``module_id`` (e.g. "h1").
        confidence_cap: Maximum confidence any signal can carry (default 0.90).
    """

    def __init__(
        self,
        period: int = 14,
        adx_threshold: float = 25.0,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.90,
    ) -> None:
        if period < 2:
            raise ValueError(f"period must be >= 2, got {period}")
        self._period = period
        self._adx_threshold = Decimal(str(adx_threshold))
        self._timeframe_id = timeframe_id
        self._confidence_cap = confidence_cap

    @property
    def module_id(self) -> str:
        return f"adx_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # ADX needs period*2+1 bars for a valid reading
        return self._period * 2 + 5

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]

        adx_vals, plus_di, minus_di = compute_adx(highs, lows, closes, self._period)

        curr_adx = adx_vals[-1]
        curr_pdi = plus_di[-1]
        curr_mdi = minus_di[-1]

        adx_float = float(curr_adx)
        pdi_float = float(curr_pdi)
        mdi_float = float(curr_mdi)
        threshold_float = float(self._adx_threshold)

        # ── Signal detection ──────────────────────────────────────────────────
        if curr_adx >= self._adx_threshold:
            if curr_pdi > curr_mdi:
                direction = SignalDirection.BUY
                reason = (
                    f"ADX({self._period})={adx_float:.1f} ≥ {threshold_float:.0f}: "
                    f"+DI={pdi_float:.1f} > −DI={mdi_float:.1f} → uptrend"
                )
            else:
                direction = SignalDirection.SELL
                reason = (
                    f"ADX({self._period})={adx_float:.1f} ≥ {threshold_float:.0f}: "
                    f"−DI={mdi_float:.1f} > +DI={pdi_float:.1f} → downtrend"
                )
            # Confidence scales with ADX strength above threshold
            surplus = max(0.0, adx_float - threshold_float)
            confidence = min(self._confidence_cap, 0.55 + surplus / 100.0)
        else:
            direction = SignalDirection.HOLD
            confidence = 0.50
            reason = (
                f"ADX({self._period})={adx_float:.1f} < {threshold_float:.0f}: "
                f"no clear trend (ranging)"
            )

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "adx": round(adx_float, 2),
                "plus_di": round(pdi_float, 2),
                "minus_di": round(mdi_float, 2),
                "threshold": threshold_float,
            },
        )
