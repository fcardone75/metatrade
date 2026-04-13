"""MACD analysis module.

Generates BUY/SELL based on MACD line crossing the signal line.
Confidence scales with the histogram absolute value (momentum strength).
"""

from __future__ import annotations

from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule
from metatrade.technical_analysis.indicators.macd import macd

_VERSION = ModuleVersion(1, 0, 0)


class MacdModule(ITechnicalModule):
    """MACD crossover signal generator.

    Args:
        fast:         Fast EMA period (default 12).
        slow:         Slow EMA period (default 26).
        signal:       Signal line EMA period (default 9).
        timeframe_id: Suffix for module_id.
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        timeframe_id: str = "h1",
    ) -> None:
        self._fast = fast
        self._slow = slow
        self._signal = signal
        self._timeframe_id = timeframe_id

    @property
    def module_id(self) -> str:
        return f"macd_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        return self._slow + self._signal + 1  # warmup + crossover detection

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)
        closes = [b.close for b in bars]
        macd_line, signal_line, histogram = macd(closes, self._fast, self._slow, self._signal)

        prev_diff = macd_line[-2] - signal_line[-2]
        curr_diff = macd_line[-1] - signal_line[-1]
        hist_val = abs(float(histogram[-1]))

        if prev_diff <= 0 and curr_diff > 0:
            direction = SignalDirection.BUY
            reason = (
                f"MACD({self._fast},{self._slow},{self._signal}) bullish crossover, "
                f"histogram={float(histogram[-1]):.6f}"
            )
        elif prev_diff >= 0 and curr_diff < 0:
            direction = SignalDirection.SELL
            reason = (
                f"MACD({self._fast},{self._slow},{self._signal}) bearish crossover, "
                f"histogram={float(histogram[-1]):.6f}"
            )
        else:
            direction = SignalDirection.HOLD
            reason = (
                f"MACD({self._fast},{self._slow},{self._signal}) no crossover, "
                f"histogram={float(histogram[-1]):.6f}"
            )

        # Confidence from histogram magnitude (0.0001 → ~50%, 0.001 → cap)
        if direction != SignalDirection.HOLD:
            confidence = min(0.85, 0.50 + hist_val * 500)
        else:
            confidence = 0.55

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "macd": float(macd_line[-1]),
                "signal": float(signal_line[-1]),
                "histogram": float(histogram[-1]),
            },
        )
