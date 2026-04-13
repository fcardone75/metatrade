"""Stochastic RSI analysis module.

Replaces the MACD module.  While MACD is essentially a derivative of EMA
(which the EmaCrossoverModule already captures), Stochastic RSI operates on
RSI values rather than on raw price — making it genuinely complementary to
both EMA-based trend modules and the plain RSI mean-reversion module.

Signal logic (K/D crossover within extreme zones):
    BUY:  %K crosses above %D AND both lines were below ``oversold`` (0.20).
          → RSI momentum turning upward from an oversold RSI level.
    SELL: %K crosses below %D AND both lines were above ``overbought`` (0.80).
          → RSI momentum turning downward from an overbought RSI level.
    HOLD: No crossover, or crossover outside the extreme zones.

Confidence:
    - Scales with depth into the oversold/overbought zone at signal time.
    - A crossover from deeper in the zone → higher confidence.
"""

from __future__ import annotations

from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule
from metatrade.technical_analysis.indicators.stochastic import stochastic_rsi

_VERSION = ModuleVersion(1, 0, 0)


class StochasticRsiModule(ITechnicalModule):
    """Stochastic RSI crossover signal generator.

    Args:
        rsi_period:     RSI calculation period (default 14).
        stoch_period:   Stochastic look-back on RSI values (default 14).
        smooth_k:       %K smoothing period (default 3).
        smooth_d:       %D smoothing period (default 3).
        oversold:       %K/%D level below which a K-cross-D is bullish (default 0.20).
        overbought:     %K/%D level above which a K-cross-D is bearish (default 0.80).
        timeframe_id:   Suffix for ``module_id`` (e.g. "h1").
        confidence_cap: Maximum confidence any signal can carry (default 0.85).
    """

    def __init__(
        self,
        rsi_period: int = 14,
        stoch_period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3,
        oversold: float = 0.20,
        overbought: float = 0.80,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.85,
    ) -> None:
        if rsi_period < 2:
            raise ValueError(f"rsi_period must be >= 2, got {rsi_period}")
        if stoch_period < 2:
            raise ValueError(f"stoch_period must be >= 2, got {stoch_period}")
        if not (0 < oversold < overbought < 1):
            raise ValueError("oversold must be in (0, overbought) and overbought < 1")
        self._rsi_period = rsi_period
        self._stoch_period = stoch_period
        self._smooth_k = smooth_k
        self._smooth_d = smooth_d
        self._oversold = oversold
        self._overbought = overbought
        self._timeframe_id = timeframe_id
        self._confidence_cap = confidence_cap

    @property
    def module_id(self) -> str:
        return f"stoch_rsi_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # rsi_period + stoch_period + smooth_k + smooth_d + 2 (for crossover detection)
        return self._rsi_period + self._stoch_period + self._smooth_k + self._smooth_d + 4

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        closes = [b.close for b in bars]
        k_line, d_line = stochastic_rsi(
            closes,
            rsi_period=self._rsi_period,
            stoch_period=self._stoch_period,
            smooth_k=self._smooth_k,
            smooth_d=self._smooth_d,
        )

        k_curr = float(k_line[-1])
        k_prev = float(k_line[-2])
        d_curr = float(d_line[-1])
        d_prev = float(d_line[-2])

        # ── Crossover detection ───────────────────────────────────────────────
        # BUY: K crosses above D (prev: K <= D; curr: K > D) in oversold zone
        bullish_cross = k_prev <= d_prev and k_curr > d_curr
        # SELL: K crosses below D (prev: K >= D; curr: K < D) in overbought zone
        bearish_cross = k_prev >= d_prev and k_curr < d_curr

        # Zone check: at least one of the two bars must be in the extreme zone
        was_oversold = min(k_prev, d_prev) < self._oversold
        was_overbought = max(k_prev, d_prev) > self._overbought

        if bullish_cross and was_oversold:
            direction = SignalDirection.BUY
            # Depth = how far into oversold zone was the K line before crossover
            depth = max(0.0, self._oversold - min(k_prev, k_curr))
            depth_ratio = depth / self._oversold
            confidence = min(self._confidence_cap, 0.55 + depth_ratio * 0.30)
            reason = (
                f"StochRSI({self._rsi_period},{self._stoch_period}) K({k_curr:.3f}) "
                f"crossed above D({d_curr:.3f}) from oversold zone (<{self._oversold})"
            )

        elif bearish_cross and was_overbought:
            direction = SignalDirection.SELL
            depth = max(0.0, max(k_prev, k_curr) - self._overbought)
            depth_ratio = depth / (1.0 - self._overbought)
            confidence = min(self._confidence_cap, 0.55 + depth_ratio * 0.30)
            reason = (
                f"StochRSI({self._rsi_period},{self._stoch_period}) K({k_curr:.3f}) "
                f"crossed below D({d_curr:.3f}) from overbought zone (>{self._overbought})"
            )

        else:
            direction = SignalDirection.HOLD
            confidence = 0.50
            reason = (
                f"StochRSI({self._rsi_period},{self._stoch_period}) "
                f"K={k_curr:.3f} D={d_curr:.3f}: "
                f"no extreme-zone crossover"
            )

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "k": round(k_curr, 4),
                "d": round(d_curr, 4),
                "oversold": self._oversold,
                "overbought": self._overbought,
            },
        )
