"""Break-even exit rule.

After the position has moved ``trigger_pips`` in the favourable direction,
suggest moving the stop-loss to ``entry_price + buffer_pips`` (LONG) or
``entry_price - buffer_pips`` (SHORT).

This rule never directly closes the position — it only updates the SL.
The trailing stop or SL hit is then detected by the TrailingStopRule on the
next evaluation.
"""

from __future__ import annotations

from decimal import Decimal

from metatrade.exit_engine.config import BreakEvenConfig
from metatrade.exit_engine.contracts import ExitSignal, PositionContext, PositionSide
from metatrade.exit_engine.rules.base import IExitRule

_PIP = Decimal("0.0001")


class BreakEvenRule(IExitRule):
    """Move SL to break-even once profit exceeds trigger_pips."""

    def __init__(self, cfg: BreakEvenConfig | None = None) -> None:
        self._cfg = cfg or BreakEvenConfig()

    @property
    def rule_id(self) -> str:
        return "break_even"

    def evaluate(self, ctx: PositionContext) -> ExitSignal:
        if not self._cfg.enabled:
            return self._hold("break_even disabled")

        trigger = Decimal(str(self._cfg.trigger_pips)) * _PIP
        buffer  = Decimal(str(self._cfg.buffer_pips))  * _PIP
        be_sl   = self._be_sl(ctx, buffer)

        # Check if SL is already at or beyond break-even
        if ctx.stop_loss is not None:
            if ctx.side == PositionSide.LONG  and ctx.stop_loss >= be_sl:
                return self._hold("SL already at or above break-even")
            if ctx.side == PositionSide.SHORT and ctx.stop_loss <= be_sl:
                return self._hold("SL already at or below break-even")

        if ctx.unrealized_pips >= trigger / _PIP:
            return self._hold(
                reason=f"Break-even trigger reached: suggest SL -> {be_sl:.5f}",
                be_sl=float(be_sl),
                trigger_pips=float(self._cfg.trigger_pips),
                activated=True,
            )
        return self._hold(
            reason=f"Break-even not yet triggered ({float(ctx.unrealized_pips):.1f} / {self._cfg.trigger_pips} pips)",
            activated=False,
        )

    def suggested_sl(self, ctx: PositionContext) -> float | None:
        if not self._cfg.enabled:
            return None
        trigger = Decimal(str(self._cfg.trigger_pips))
        if ctx.unrealized_pips < trigger:
            return None
        buffer = Decimal(str(self._cfg.buffer_pips)) * _PIP
        be_sl  = self._be_sl(ctx, buffer)
        # Only suggest if it improves on current SL
        if ctx.stop_loss is not None:
            if ctx.side == PositionSide.LONG  and be_sl <= ctx.stop_loss:
                return None
            if ctx.side == PositionSide.SHORT and be_sl >= ctx.stop_loss:
                return None
        return float(be_sl)

    @staticmethod
    def _be_sl(ctx: PositionContext, buffer: Decimal) -> Decimal:
        if ctx.side == PositionSide.LONG:
            return ctx.entry_price + buffer
        return ctx.entry_price - buffer
