"""Partial exit rule.

Closes a fraction of the position at each configured profit level.

Example config:
    levels:
      - pips: 20   close_pct: 0.33   ← close 33% at +20 pip
      - pips: 40   close_pct: 0.33   ← close another 33% at +40 pip

Levels are evaluated in ascending pip order.  A level is skipped if it has
already been executed (tracked via ctx.partial_closes).

The rule signals CLOSE_PARTIAL only for the first un-executed level that has
been reached.  It does not signal multiple levels in one bar to avoid
over-trading.
"""

from __future__ import annotations

from metatrade.exit_engine.config import PartialExitConfig
from metatrade.exit_engine.contracts import ExitSignal, PositionContext
from metatrade.exit_engine.rules.base import IExitRule


class PartialExitRule(IExitRule):
    """Close a fraction of the position at each pip-profit level."""

    def __init__(self, cfg: PartialExitConfig | None = None) -> None:
        self._cfg = cfg or PartialExitConfig()

    @property
    def rule_id(self) -> str:
        return "partial_exit"

    def evaluate(self, ctx: PositionContext) -> ExitSignal:
        if not self._cfg.enabled or not self._cfg.levels:
            return self._hold("partial_exit disabled or no levels configured")

        current_pips = float(ctx.unrealized_pips)
        already_executed = {pips for pips, _ in ctx.partial_closes}

        # Evaluate levels in order — signal only the first un-executed, reached level
        for level in sorted(self._cfg.levels, key=lambda lv: lv.pips):
            if level.pips in already_executed:
                continue
            if current_pips >= level.pips:
                return self._exit_partial(
                    close_pct=level.close_pct,
                    reason=f"Partial exit: profit {current_pips:.1f} pip reached level {level.pips} pip",
                    confidence=0.80,
                    level_pips=level.pips,
                    close_pct_meta=level.close_pct,
                    current_pips=round(current_pips, 2),
                )

        return self._hold(
            reason=f"No partial exit level reached yet ({current_pips:.1f} pip profit)",
            current_pips=round(current_pips, 2),
        )
