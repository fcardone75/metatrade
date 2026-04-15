"""Setup-invalidation exit rule.

Closes the position when the original trade setup is no longer valid:

  1. MA cross invalidation — the fast MA crosses the slow MA in the
     opposite direction from the trade (e.g. fast falls below slow for LONG).
  2. Price-level invalidation — the price closes beyond a hard level
     that was the structural basis of the trade.
"""

from __future__ import annotations

from decimal import Decimal

from metatrade.exit_engine.config import SetupInvalidationConfig
from metatrade.exit_engine.contracts import ExitSignal, PositionContext, PositionSide
from metatrade.exit_engine.rules.base import IExitRule


class SetupInvalidationRule(IExitRule):
    """Exit when the entry setup (MA structure / price level) breaks."""

    def __init__(self, cfg: SetupInvalidationConfig | None = None) -> None:
        self._cfg = cfg or SetupInvalidationConfig()

    @property
    def rule_id(self) -> str:
        return "setup_invalidation"

    def evaluate(self, ctx: PositionContext) -> ExitSignal:
        if not self._cfg.enabled:
            return self._hold("setup_invalidation disabled")

        # 1. Hard price-level invalidation
        if self._cfg.price_level is not None:
            level = Decimal(str(self._cfg.price_level))
            if ctx.side == PositionSide.LONG and ctx.current_price < level:
                return self._exit_full(
                    reason=f"Price {ctx.current_price:.5f} broke below invalidation level {level:.5f}",
                    confidence=0.90,
                    level=float(level),
                )
            if ctx.side == PositionSide.SHORT and ctx.current_price > level:
                return self._exit_full(
                    reason=f"Price {ctx.current_price:.5f} broke above invalidation level {level:.5f}",
                    confidence=0.90,
                    level=float(level),
                )

        # 2. MA cross invalidation
        if self._cfg.invalidate_on_ma_cross:
            result = self._check_ma_cross(ctx)
            if result is not None:
                return result

        return self._hold("Setup intact — no invalidation triggered")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_ma_cross(self, ctx: PositionContext) -> ExitSignal | None:
        bars = ctx.bars_since_entry
        fast_p = self._cfg.ma_cross_fast
        slow_p = self._cfg.ma_cross_slow
        # Need at least slow_p + 1 bars for a valid cross detection
        if len(bars) < slow_p + 1:
            return None

        closes = [float(b.close) for b in bars]
        fast_now  = self._ema(closes, fast_p)
        slow_now  = self._ema(closes, slow_p)
        fast_prev = self._ema(closes[:-1], fast_p)
        slow_prev = self._ema(closes[:-1], slow_p)

        if fast_now is None or slow_now is None or fast_prev is None or slow_prev is None:
            return None

        # LONG: invalidated when fast drops below slow (bearish cross)
        if ctx.side == PositionSide.LONG:
            if fast_prev >= slow_prev and fast_now < slow_now:
                return self._exit_full(
                    reason=f"Bearish MA cross: fast EMA({fast_p}) crossed below slow EMA({slow_p})",
                    confidence=0.82,
                    fast_ma=round(fast_now, 6),
                    slow_ma=round(slow_now, 6),
                )
        # SHORT: invalidated when fast rises above slow (bullish cross)
        else:
            if fast_prev <= slow_prev and fast_now > slow_now:
                return self._exit_full(
                    reason=f"Bullish MA cross: fast EMA({fast_p}) crossed above slow EMA({slow_p})",
                    confidence=0.82,
                    fast_ma=round(fast_now, 6),
                    slow_ma=round(slow_now, 6),
                )
        return None

    @staticmethod
    def _ema(values: list[float], period: int) -> float | None:
        if len(values) < period:
            return None
        k = 2.0 / (period + 1)
        result = sum(values[:period]) / period
        for v in values[period:]:
            result = v * k + result * (1.0 - k)
        return result
