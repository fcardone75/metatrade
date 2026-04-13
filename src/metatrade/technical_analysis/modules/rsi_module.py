"""RSI analysis module.

Generates BUY when RSI < oversold threshold, SELL when RSI > overbought threshold.
Confidence scales with how deep into the extreme zone the RSI is.
"""

from __future__ import annotations

from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule
from metatrade.technical_analysis.indicators.rsi import rsi

_VERSION = ModuleVersion(1, 0, 0)


class RsiModule(ITechnicalModule):
    """RSI mean-reversion signal generator.

    Args:
        period:      RSI calculation period (default 14).
        oversold:    RSI level below which BUY signal is generated (default 30).
        overbought:  RSI level above which SELL signal is generated (default 70).
        timeframe_id: Suffix for module_id.
    """

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        timeframe_id: str = "h1",
    ) -> None:
        self._period = period
        self._oversold = oversold
        self._overbought = overbought
        self._timeframe_id = timeframe_id

    @property
    def module_id(self) -> str:
        return f"rsi_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        return self._period + 2  # period + 1 for RSI warmup + 1 for latest

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)
        closes = [b.close for b in bars]
        rsi_values = rsi(closes, self._period)
        latest_rsi = float(rsi_values[-1])

        if latest_rsi < self._oversold:
            direction = SignalDirection.BUY
            # Deeper into oversold → higher confidence
            depth = (self._oversold - latest_rsi) / self._oversold
            confidence = min(0.90, 0.50 + depth * 0.50)
            reason = f"RSI({self._period})={latest_rsi:.1f} oversold (< {self._oversold})"
        elif latest_rsi > self._overbought:
            direction = SignalDirection.SELL
            depth = (latest_rsi - self._overbought) / (100 - self._overbought)
            confidence = min(0.90, 0.50 + depth * 0.50)
            reason = f"RSI({self._period})={latest_rsi:.1f} overbought (> {self._overbought})"
        else:
            direction = SignalDirection.HOLD
            confidence = 0.60
            reason = f"RSI({self._period})={latest_rsi:.1f} neutral"

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={"rsi": latest_rsi},
        )
