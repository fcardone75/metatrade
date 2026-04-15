"""Volatility-expansion exit rule.

Exits the position when the current ATR exceeds a multiple of the ATR
at the time of entry.  A major volatility expansion (e.g. news event) means
the risk profile of the trade has changed — the original SL distance is now
too small relative to noise, and the position is better closed.
"""

from __future__ import annotations

from decimal import Decimal

from metatrade.exit_engine.config import VolatilityExitConfig
from metatrade.exit_engine.contracts import ExitSignal, PositionContext
from metatrade.exit_engine.rules.base import IExitRule


class VolatilityExitRule(IExitRule):
    """Exit when current ATR significantly exceeds entry ATR."""

    def __init__(self, cfg: VolatilityExitConfig | None = None) -> None:
        self._cfg = cfg or VolatilityExitConfig()

    @property
    def rule_id(self) -> str:
        return "volatility_exit"

    def evaluate(self, ctx: PositionContext) -> ExitSignal:
        if not self._cfg.enabled:
            return self._hold("volatility_exit disabled")
        if ctx.entry_atr is None or ctx.entry_atr == 0:
            return self._hold("No entry ATR available — cannot compare volatility")

        current_atr = self._current_atr(ctx)
        if current_atr is None:
            return self._hold("Insufficient bars to compute current ATR")

        ratio = float(current_atr) / float(ctx.entry_atr)
        threshold = self._cfg.atr_expansion_mult

        if ratio >= threshold:
            return self._exit_full(
                reason=(
                    f"Volatility expanded {ratio:.2f}x vs entry ATR "
                    f"(threshold {threshold:.1f}x)"
                ),
                confidence=min(0.95, 0.60 + (ratio - threshold) * 0.10),
                current_atr=float(current_atr),
                entry_atr=float(ctx.entry_atr),
                ratio=round(ratio, 3),
            )
        return self._hold(
            reason=f"ATR ratio {ratio:.2f}x below threshold {threshold:.1f}x",
            ratio=round(ratio, 3),
        )

    def _current_atr(self, ctx: PositionContext) -> Decimal | None:
        bars = ctx.bars_since_entry
        period = self._cfg.atr_period
        if len(bars) < period + 1:
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
