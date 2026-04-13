"""ATR (Average True Range) analysis module.

ATR does not generate directional signals — it is a volatility indicator.
This module emits HOLD with high confidence when volatility is elevated
(ATR > threshold × average ATR) and lower confidence otherwise.

Primary use-case: feed ATR-based stop-loss distances to the risk manager.
The signal metadata always includes `atr_value` and `sl_buy`/`sl_sell`.
"""

from __future__ import annotations

from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule
from metatrade.technical_analysis.indicators.atr import atr, atr_stop_loss

_VERSION = ModuleVersion(1, 0, 0)


class AtrModule(ITechnicalModule):
    """ATR volatility module.

    Always emits HOLD — it is a supporting/meta module that contributes
    volatility context (stop-loss levels) rather than directional signals.

    Args:
        period:          ATR period (default 14).
        sl_multiplier:   SL distance = sl_multiplier × ATR (default 2.0).
        timeframe_id:    Suffix for module_id (e.g. "h1" → "atr_h1").
        high_vol_factor: If current ATR > high_vol_factor × average ATR,
                         the signal metadata flags high-volatility regime.
    """

    def __init__(
        self,
        period: int = 14,
        sl_multiplier: float = 2.0,
        timeframe_id: str = "h1",
        high_vol_factor: float = 1.5,
    ) -> None:
        if period < 2:
            raise ValueError(f"ATR period must be >= 2, got {period}")
        self._period = period
        self._sl_multiplier = sl_multiplier
        self._timeframe_id = timeframe_id
        self._high_vol_factor = high_vol_factor

    @property
    def module_id(self) -> str:
        return f"atr_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # period+1 for ATR warmup + a few extra so avg_atr is meaningful
        return self._period + 5

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]

        atr_values = atr(highs, lows, closes, self._period)
        current_atr = atr_values[-1]

        # Use the last few ATR values to estimate average (avoids bootstrap noise)
        lookback = min(self._period, len(atr_values))
        avg_atr = sum(atr_values[-lookback:]) / lookback

        high_volatility = float(current_atr) > float(avg_atr) * self._high_vol_factor

        entry = closes[-1]
        sl_buy = atr_stop_loss(entry, current_atr, self._sl_multiplier, side_is_buy=True)
        sl_sell = atr_stop_loss(entry, current_atr, self._sl_multiplier, side_is_buy=False)

        # ATR always emits HOLD — it only provides context
        confidence = 0.75 if high_volatility else 0.60
        reason = (
            f"ATR({self._period})={float(current_atr):.5f} "
            f"({'high' if high_volatility else 'normal'} volatility)"
        )

        return AnalysisSignal(
            direction=SignalDirection.HOLD,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "atr": float(current_atr),
                "avg_atr": float(avg_atr),
                "high_volatility": high_volatility,
                "sl_buy": float(sl_buy),
                "sl_sell": float(sl_sell),
            },
        )
