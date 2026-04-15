"""Time-based exit rule.

Two triggers (independently configurable):
  1. max_holding_bars  — close if the position has been open ≥ N bars.
  2. session_end       — close when the UTC hour reaches session_end_hour_utc.
"""

from __future__ import annotations

from metatrade.exit_engine.config import TimeExitConfig
from metatrade.exit_engine.contracts import ExitSignal, PositionContext
from metatrade.exit_engine.rules.base import IExitRule


class TimeExitRule(IExitRule):
    """Close position after max bars held or at session end."""

    def __init__(self, cfg: TimeExitConfig | None = None) -> None:
        self._cfg = cfg or TimeExitConfig()

    @property
    def rule_id(self) -> str:
        return "time_exit"

    def evaluate(self, ctx: PositionContext) -> ExitSignal:
        if not self._cfg.enabled:
            return self._hold("time_exit disabled")

        # Check max holding bars first (hard limit)
        if ctx.bars_held >= self._cfg.max_holding_bars:
            return self._exit_full(
                reason=f"Max holding time reached ({ctx.bars_held} bars ≥ {self._cfg.max_holding_bars})",
                confidence=0.85,
                bars_held=ctx.bars_held,
                max_bars=self._cfg.max_holding_bars,
            )

        # Check session end
        if self._cfg.close_on_session_end:
            bar_hour = ctx.current_bar.timestamp_utc.hour
            if bar_hour >= self._cfg.session_end_hour_utc:
                return self._exit_full(
                    reason=f"Session end reached (bar hour {bar_hour} UTC ≥ {self._cfg.session_end_hour_utc})",
                    confidence=0.80,
                    bar_hour=bar_hour,
                    session_end_hour=self._cfg.session_end_hour_utc,
                )

        return self._hold(
            reason=f"Time OK: {ctx.bars_held}/{self._cfg.max_holding_bars} bars held",
            bars_held=ctx.bars_held,
        )
