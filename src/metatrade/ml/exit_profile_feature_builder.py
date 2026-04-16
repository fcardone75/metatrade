"""Feature extraction for exit profile selection.

Produces a flat dict of finite floats from an ExitProfileContext for use by the
ML profile selector.  Features capture the signals most predictive of which
exit strategy will perform best in the current market condition:

- Volatility level and regime (ATR magnitude, relative range)
- Short-term ATR trend (expanding vs compressing volatility)
- Spread cost relative to ATR (transaction-cost friction)
- Time-of-day and day-of-week (session context)
- Direction of the trade (BUY vs SELL)

All features are normalised to finite floats.  NaN or Inf values are replaced
with 0.0 so the downstream model always receives valid input.
"""

from __future__ import annotations

import math

from metatrade.ml.exit_profile_contracts import ExitProfileContext


class ExitProfileFeatureBuilder:
    """Extracts a flat feature dict from an ExitProfileContext.

    Instances are stateless — the same object can be reused across calls.
    """

    # Ordered list of feature names; guarantees a consistent column order
    # when building numpy arrays for sklearn.
    FEATURE_NAMES: tuple[str, ...] = (
        "atr_pips",
        "spread_pips",
        "spread_atr_ratio",
        "recent_range_pips",
        "vol_regime",
        "atr_trend_ratio",
        "hour",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_buy",
    )

    def build(self, ctx: ExitProfileContext) -> dict[str, float]:
        """Return a feature dict for *ctx*.

        All values are guaranteed to be finite floats.

        Args:
            ctx: Entry context from the signal pipeline.

        Returns:
            Dict mapping feature name → float value.
        """
        pip = float(ctx.pip_size)
        atr_pips = float(ctx.atr) / pip if pip > 0 else 0.0
        spread_pips = float(ctx.spread_pips)

        recent_range_pips = self._recent_range_pips(ctx, pip)
        atr_trend = self._atr_trend_ratio(ctx)

        vol_regime = (atr_pips / recent_range_pips) if recent_range_pips > 0 else 1.0
        spread_atr = spread_pips / atr_pips if atr_pips > 0 else 0.0

        hour = ctx.timestamp_utc.hour
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow = float(ctx.timestamp_utc.weekday())
        is_buy = 1.0 if ctx.side.value == "BUY" else 0.0

        raw = {
            "atr_pips": atr_pips,
            "spread_pips": spread_pips,
            "spread_atr_ratio": spread_atr,
            "recent_range_pips": recent_range_pips,
            "vol_regime": vol_regime,
            "atr_trend_ratio": atr_trend,
            "hour": float(hour),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week": dow,
            "is_buy": is_buy,
        }
        # Sanitise: replace any non-finite value with 0.0
        return {k: (v if math.isfinite(v) else 0.0) for k, v in raw.items()}

    def build_vector(self, ctx: ExitProfileContext) -> list[float]:
        """Return features as an ordered list (for numpy array construction)."""
        feats = self.build(ctx)
        return [feats[k] for k in self.FEATURE_NAMES]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _recent_range_pips(self, ctx: ExitProfileContext, pip: float) -> float:
        """Price range of the most recent 20 bars in pips."""
        n = min(20, len(ctx.recent_highs), len(ctx.recent_lows))
        if n == 0:
            return float(ctx.atr) / pip if pip > 0 else 0.0
        highs = [float(h) for h in ctx.recent_highs[-n:]]
        lows  = [float(l) for l in ctx.recent_lows[-n:]]
        if pip <= 0:
            return 0.0
        return (max(highs) - min(lows)) / pip

    def _atr_trend_ratio(self, ctx: ExitProfileContext) -> float:
        """Ratio of short-term range (last 14 bars) to longer-term range.

        > 1.0 → expanding volatility
        < 1.0 → compressing volatility
        """
        n_short = 14
        n = min(len(ctx.recent_highs), len(ctx.recent_lows))
        if n < n_short:
            return 1.0
        short_hi = [float(h) for h in ctx.recent_highs[-n_short:]]
        short_lo = [float(l) for l in ctx.recent_lows[-n_short:]]
        long_hi  = [float(h) for h in ctx.recent_highs]
        long_lo  = [float(l) for l in ctx.recent_lows]
        short_range = max(short_hi) - min(short_lo)
        long_range  = max(long_hi)  - min(long_lo)
        if long_range <= 0:
            return 1.0
        return short_range / long_range
