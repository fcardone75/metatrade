"""Give-back from peak exit rule.

Closes the position when the floating profit has retreated more than
``give_back_pct`` (e.g. 40%) from its peak value.

Example:
    Entry at 1.1000, peak at 1.1040 (40 pip peak profit).
    give_back_pct = 0.40 → close if profit drops to 40 × (1 - 0.40) = 24 pip.
    I.e. close if current price drops below 1.1000 + 0.0024 = 1.1024.

Rationale: locks in a meaningful portion of the winning trade without a
fixed TP, letting the trade run while the trend holds.
"""

from __future__ import annotations

from decimal import Decimal

from metatrade.exit_engine.config import GiveBackConfig
from metatrade.exit_engine.contracts import ExitSignal, PositionContext
from metatrade.exit_engine.rules.base import IExitRule

_PIP = Decimal("0.0001")


class GiveBackRule(IExitRule):
    """Exit when floating profit gives back more than give_back_pct from peak."""

    def __init__(self, cfg: GiveBackConfig | None = None) -> None:
        self._cfg = cfg or GiveBackConfig()

    @property
    def rule_id(self) -> str:
        return "give_back"

    def evaluate(self, ctx: PositionContext) -> ExitSignal:
        if not self._cfg.enabled:
            return self._hold("give_back disabled")

        peak_pips = ctx.peak_profit_pips
        if peak_pips <= Decimal("0"):
            return self._hold("No peak profit yet — give-back rule inactive")

        current_pips = ctx.unrealized_pips
        give_back_pct = Decimal(str(self._cfg.give_back_pct))
        min_profit_to_keep = peak_pips * (1 - give_back_pct)

        if current_pips < min_profit_to_keep:
            actual_give_back_pct = float((peak_pips - current_pips) / peak_pips)
            return self._exit_full(
                reason=(
                    f"Give-back {actual_give_back_pct:.0%} from peak "
                    f"({float(peak_pips):.1f} pip peak -> {float(current_pips):.1f} pip now)"
                ),
                confidence=min(0.90, 0.65 + actual_give_back_pct * 0.5),
                peak_pips=float(peak_pips),
                current_pips=float(current_pips),
                give_back_pct=round(actual_give_back_pct, 3),
            )
        return self._hold(
            reason=(
                f"Give-back OK: {float(current_pips):.1f} pip profit "
                f"(peak {float(peak_pips):.1f}, min_keep {float(min_profit_to_keep):.1f})"
            ),
            peak_pips=float(peak_pips),
            current_pips=float(current_pips),
        )
