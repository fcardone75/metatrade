"""Trailing stop exit rule.

Three modes:
  atr        — SL = peak_favorable ± ATR(period) × mult
  fixed_pct  — SL = peak_favorable ± peak_favorable × pct
  fixed_pips — SL = peak_favorable ± pips × pip_size
"""

from __future__ import annotations

from decimal import Decimal

from metatrade.exit_engine.config import TrailingStopConfig
from metatrade.exit_engine.contracts import ExitSignal, PositionContext, PositionSide
from metatrade.exit_engine.rules.base import IExitRule

_PIP = Decimal("0.0001")


class TrailingStopRule(IExitRule):
    """ATR / fixed-% / fixed-pip trailing stop."""

    def __init__(self, cfg: TrailingStopConfig | None = None) -> None:
        self._cfg = cfg or TrailingStopConfig()

    @property
    def rule_id(self) -> str:
        return "trailing_stop"

    def evaluate(self, ctx: PositionContext) -> ExitSignal:
        if not self._cfg.enabled:
            return self._hold("trailing_stop disabled")

        sl = self._compute_sl(ctx)
        if sl is None:
            return self._hold("insufficient bars for ATR")

        hit = (
            ctx.side == PositionSide.LONG  and ctx.current_price <= sl
        ) or (
            ctx.side == PositionSide.SHORT and ctx.current_price >= sl
        )

        if hit:
            return self._exit_full(
                reason=f"Trailing SL hit at {sl:.5f} (current {ctx.current_price:.5f})",
                confidence=0.95,
                sl_level=float(sl),
                mode=self._cfg.mode,
            )
        return self._hold(
            f"Trailing SL at {sl:.5f} not hit",
            sl_level=float(sl),
        )

    def suggested_sl(self, ctx: PositionContext) -> float | None:
        sl = self._compute_sl(ctx)
        return float(sl) if sl is not None else None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_sl(self, ctx: PositionContext) -> Decimal | None:
        mode = self._cfg.mode
        peak = ctx.peak_favorable_price

        if mode == "atr":
            atr = self._atr(ctx)
            if atr is None:
                return None
            distance = atr * Decimal(str(self._cfg.atr_mult))
        elif mode == "fixed_pct":
            pct = self._cfg.fixed_pct or 0.005
            distance = peak * Decimal(str(pct))
        elif mode == "fixed_pips":
            pips = self._cfg.fixed_pips or 15.0
            distance = Decimal(str(pips)) * _PIP
        else:
            return None

        if ctx.side == PositionSide.LONG:
            return peak - distance
        return peak + distance

    def _atr(self, ctx: PositionContext) -> Decimal | None:
        bars = ctx.bars_since_entry
        period = self._cfg.atr_period
        if len(bars) < period + 1:
            # Fall back to entry ATR if available
            return ctx.entry_atr
        try:
            from metatrade.technical_analysis.indicators.atr import atr as compute_atr
            highs  = [b.high  for b in bars]
            lows   = [b.low   for b in bars]
            closes = [b.close for b in bars]
            vals = compute_atr(highs, lows, closes, period)
            return vals[-1] if vals else ctx.entry_atr
        except Exception:
            return ctx.entry_atr
