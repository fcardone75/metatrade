"""Seasonality analysis module.

Combines **session quality** (time-of-day) with **calendar patterns**
(day-of-week) to decide whether current market conditions favour directional
trading.  When conditions are unfavourable, the module withholds its vote
(HOLD) so that other directional modules need stronger consensus to act.

Session schedule (UTC):
    Asian session    00:00 – 08:00   Low liquidity for EUR/USD majors
    London open      07:00 – 09:00   Strong directional moves, high liquidity
    London session   09:00 – 13:00   Sustained trending conditions
    NY open          13:00 – 15:00   High liquidity, volatile
    London–NY overlap 13:00 – 17:00  Highest combined liquidity of the day
    NY session       15:00 – 21:00   USD-driven moves
    Dead zone        21:00 – 00:00   Low liquidity, unpredictable gaps

Day-of-week patterns:
    Monday   — often continuation/reversal of weekend gap; slightly lower
               reliability, treat as neutral
    Tue–Thu  — most reliable trending days; full weight
    Friday   — avoid after 15:00 UTC (end-of-week position unwinding, low
               liquidity, mean-reversion rather than trend)
    Weekend  — market closed (should not occur in live data)

Signal logic:
    The module combines session quality with short-term price momentum
    (close vs. close 3 bars ago):

    EXCELLENT session (London–NY overlap, Tue–Thu):
        BUY  if price momentum is positive
        SELL if price momentum is negative

    GOOD session (London open/session, NY open, Mon–Thu):
        BUY/SELL if momentum is decisive (≥ 0.03% move)
        HOLD     if momentum is ambiguous

    POOR session (Asian, Friday 15:00+, dead zone):
        HOLD regardless of momentum

This mirrors the empirical observation that EUR/USD and other majors trend
most reliably during the London–NY overlap and least reliably during the
Asian session.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule

_VERSION = ModuleVersion(1, 0, 0)


class _SessionQuality(Enum):
    POOR = "poor"
    GOOD = "good"
    EXCELLENT = "excellent"


def _classify_session(hour_utc: int, weekday: int) -> _SessionQuality:
    """Return session quality for a given UTC hour and ISO weekday (0=Mon, 6=Sun).

    The classification is intentionally conservative — when in doubt, GOOD
    rather than EXCELLENT, and POOR only for clearly low-liquidity windows.
    """
    # Weekend — should not occur but handle gracefully
    if weekday >= 5:
        return _SessionQuality.POOR

    # Friday afternoon: position unwinding, avoid trend-following
    if weekday == 4 and hour_utc >= 15:
        return _SessionQuality.POOR

    # Asian dead zone (no major session)
    if 0 <= hour_utc < 7:
        return _SessionQuality.POOR

    # End-of-day dead zone
    if hour_utc >= 21:
        return _SessionQuality.POOR

    # London–NY overlap: Tue–Thu only (Mon less reliable)
    if 13 <= hour_utc < 17 and weekday in (1, 2, 3):
        return _SessionQuality.EXCELLENT

    # London session + NY open: Tue–Thu
    if 7 <= hour_utc < 21 and weekday in (1, 2, 3):
        return _SessionQuality.GOOD

    # Monday or Friday morning: treat as GOOD (not EXCELLENT)
    if 7 <= hour_utc < 21:
        return _SessionQuality.GOOD

    return _SessionQuality.POOR


class SeasonalityModule(ITechnicalModule):
    """Session quality + calendar pattern signal generator.

    Args:
        momentum_bars:      Number of bars used to compute short-term price
                            momentum (default 3).
        momentum_threshold: Minimum |return| to consider momentum decisive
                            in GOOD sessions (default 0.0003 = 3 pips on
                            EURUSD).
        timeframe_id:       Suffix for ``module_id`` (e.g. "h1").
        confidence_cap:     Maximum confidence any signal can carry (default 0.78).
    """

    def __init__(
        self,
        momentum_bars: int = 3,
        momentum_threshold: float = 0.0003,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.78,
    ) -> None:
        if momentum_bars < 1:
            raise ValueError(f"momentum_bars must be >= 1, got {momentum_bars}")
        self._momentum_bars = momentum_bars
        self._momentum_threshold = momentum_threshold
        self._timeframe_id = timeframe_id
        self._confidence_cap = confidence_cap

    @property
    def module_id(self) -> str:
        return f"seasonality_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        return self._momentum_bars + 1

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        hour_utc = timestamp_utc.hour
        # ISO weekday: Monday=0 … Sunday=6
        weekday = timestamp_utc.weekday()

        quality = _classify_session(hour_utc, weekday)
        session_name = f"{timestamp_utc.strftime('%a')} {hour_utc:02d}:00 UTC"

        # ── Short-term momentum ───────────────────────────────────────────────
        curr_close = bars[-1].close
        past_close = bars[-self._momentum_bars - 1].close

        if past_close == 0:
            momentum = 0.0
        else:
            momentum = float((curr_close - past_close) / past_close)

        # ── Combine quality + momentum ────────────────────────────────────────
        if quality == _SessionQuality.POOR:
            direction = SignalDirection.HOLD
            confidence = 0.50
            reason = (
                f"Poor session ({session_name}): withholding vote "
                f"(low liquidity / end-of-week)"
            )

        elif quality == _SessionQuality.EXCELLENT:
            # Always emit a directional signal
            if momentum >= 0:
                direction = SignalDirection.BUY
            else:
                direction = SignalDirection.SELL
            # Confidence: base + momentum strength bonus
            mom_bonus = min(0.15, abs(momentum) / self._momentum_threshold * 0.05)
            confidence = min(self._confidence_cap, 0.65 + mom_bonus)
            reason = (
                f"Excellent session ({session_name}): "
                f"momentum={momentum:+.4%} → {direction.value}"
            )

        else:  # GOOD
            # Only emit if momentum is decisive enough
            if abs(momentum) >= self._momentum_threshold:
                direction = (
                    SignalDirection.BUY if momentum > 0 else SignalDirection.SELL
                )
                mom_ratio = abs(momentum) / self._momentum_threshold
                confidence = min(self._confidence_cap, 0.55 + min(mom_ratio - 1, 1.0) * 0.10)
                reason = (
                    f"Good session ({session_name}): "
                    f"momentum={momentum:+.4%} → {direction.value}"
                )
            else:
                direction = SignalDirection.HOLD
                confidence = 0.50
                reason = (
                    f"Good session ({session_name}): "
                    f"momentum={momentum:+.4%} too weak (threshold {self._momentum_threshold:.4%})"
                )

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "session_quality": quality.value,
                "weekday": weekday,
                "hour_utc": hour_utc,
                "momentum": round(momentum, 6),
                "momentum_threshold": self._momentum_threshold,
            },
        )
