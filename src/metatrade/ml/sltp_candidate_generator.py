"""SL/TP candidate generator.

Produces a finite set of plausible SL/TP candidate policies for each signal
context.  Two families of candidates are supported:

ATR-based (always generated when ATR is available):
    ATR_1_1_5  — SL = 1×ATR, TP = 1.5×ATR (R:R 1.5)
    ATR_1_2    — SL = 1×ATR, TP = 2×ATR   (R:R 2.0)  ← current default
    ATR_1_5_2  — SL = 1.5×ATR, TP = 3×ATR  (R:R 2.0, wider stop)
    ATR_1_5_3  — SL = 1.5×ATR, TP = 4.5×ATR (R:R 3.0)

Swing-based (generated when enough bar history is available):
    SWING_2R   — SL at nearest swing high/low, TP = 2× SL distance
    SWING_TRAIL— SL at nearest swing high/low, TP = 3× SL distance

Validation rules applied to every candidate (invalid ones are silently dropped):
- SL on the correct side of entry
- TP on the correct side of entry
- SL distance ≥ min_stop_pips + spread (prevents immediate stop-out)
- TP distance ≥ min_tp_pips
- R:R ≥ min_rr
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import NamedTuple

from metatrade.core.enums import OrderSide
from metatrade.ml.sltp_contracts import SlTpCandidate, SlTpContext


class _AtrSpec(NamedTuple):
    """Spec for one ATR-based candidate."""
    policy_id: str
    sl_atr_mult: float   # SL distance = sl_atr_mult × ATR
    tp_atr_mult: float   # TP distance = tp_atr_mult × ATR


_ATR_SPECS: tuple[_AtrSpec, ...] = (
    _AtrSpec("ATR_1_1_5", 1.0, 1.5),
    _AtrSpec("ATR_1_2",   1.0, 2.0),
    _AtrSpec("ATR_1_5_2", 1.5, 3.0),
    _AtrSpec("ATR_1_5_3", 1.5, 4.5),
)

# Swing-based R:R ratios mapped to policy IDs
_SWING_SPECS: tuple[tuple[str, float], ...] = (
    ("SWING_2R",    2.0),
    ("SWING_TRAIL", 3.0),
)


class SlTpCandidateGenerator:
    """Generates valid SL/TP candidates from ATR and swing policies.

    Args:
        min_stop_pips:  Minimum SL distance from entry in pips.  The actual
                        minimum is ``max(min_stop_pips, spread_pips + 1)`` to
                        prevent immediate stop-out due to spread at fill time.
        min_tp_pips:    Minimum TP distance from entry in pips.
        min_rr:         Minimum risk-reward ratio (tp_pips / sl_pips).
        swing_lookback: Number of recent bars to scan for swing levels.
        swing_buffer_pips: Buffer (in pips) added beyond the swing level.
    """

    def __init__(
        self,
        min_stop_pips: float = 5.0,
        min_tp_pips: float = 5.0,
        min_rr: float = 1.0,
        swing_lookback: int = 30,
        swing_buffer_pips: float = 1.0,
    ) -> None:
        self._min_stop_pips = Decimal(str(min_stop_pips))
        self._min_tp_pips = Decimal(str(min_tp_pips))
        self._min_rr = Decimal(str(min_rr))
        self._swing_lookback = swing_lookback
        self._swing_buffer_pips = Decimal(str(swing_buffer_pips))

    # ── Public ────────────────────────────────────────────────────────────────

    def generate(self, context: SlTpContext) -> list[SlTpCandidate]:
        """Generate all valid SL/TP candidates for the given entry context.

        Args:
            context: Current signal context (ATR, spread, recent bars, …).

        Returns:
            List of valid candidates, sorted by R:R descending.
            Empty list if no valid candidate can be constructed.
        """
        candidates: list[SlTpCandidate] = []

        # ── ATR-based candidates ──────────────────────────────────────────────
        for spec in _ATR_SPECS:
            cand = self._make_atr_candidate(spec, context)
            if cand is not None:
                candidates.append(cand)

        # ── Swing-based candidates ────────────────────────────────────────────
        swing_sl = self._find_swing_sl(context)
        if swing_sl is not None:
            for policy_id, rr in _SWING_SPECS:
                cand = self._make_swing_candidate(
                    policy_id, context, swing_sl, Decimal(str(rr))
                )
                if cand is not None:
                    candidates.append(cand)

        # Sort by R:R descending; within equal R:R prefer ATR over swing
        candidates.sort(
            key=lambda c: (c.risk_reward, 1 if c.method == "atr" else 0),
            reverse=True,
        )
        return candidates

    # ── ATR candidates ────────────────────────────────────────────────────────

    def _make_atr_candidate(
        self, spec: _AtrSpec, ctx: SlTpContext
    ) -> SlTpCandidate | None:
        sl_dist = ctx.atr * Decimal(str(spec.sl_atr_mult))
        tp_dist = ctx.atr * Decimal(str(spec.tp_atr_mult))
        if ctx.side == OrderSide.BUY:
            sl_price = ctx.entry_price - sl_dist
            tp_price = ctx.entry_price + tp_dist
        else:
            sl_price = ctx.entry_price + sl_dist
            tp_price = ctx.entry_price - tp_dist
        return self._build_candidate(spec.policy_id, "atr", ctx, sl_price, tp_price)

    # ── Swing candidates ──────────────────────────────────────────────────────

    def _find_swing_sl(self, ctx: SlTpContext) -> Decimal | None:
        """Identify the nearest significant swing level for the stop-loss.

        For BUY: use the lowest low in the recent lookback window (swing low),
        plus a 1-pip buffer below it.
        For SELL: use the highest high (swing high), plus a 1-pip buffer above.

        Returns None when there is insufficient bar history.
        """
        lookback = self._swing_lookback
        buffer = self._swing_buffer_pips * ctx.pip_size

        if ctx.side == OrderSide.BUY:
            if len(ctx.recent_lows) < 3:
                return None
            lows = ctx.recent_lows[-lookback:] if len(ctx.recent_lows) >= lookback else ctx.recent_lows
            return min(lows) - buffer
        else:
            if len(ctx.recent_highs) < 3:
                return None
            highs = ctx.recent_highs[-lookback:] if len(ctx.recent_highs) >= lookback else ctx.recent_highs
            return max(highs) + buffer

    def _make_swing_candidate(
        self,
        policy_id: str,
        ctx: SlTpContext,
        swing_sl: Decimal,
        rr: Decimal,
    ) -> SlTpCandidate | None:
        sl_dist = abs(ctx.entry_price - swing_sl)
        tp_dist = sl_dist * rr
        if ctx.side == OrderSide.BUY:
            tp_price = ctx.entry_price + tp_dist
        else:
            tp_price = ctx.entry_price - tp_dist
        return self._build_candidate(policy_id, "swing", ctx, swing_sl, tp_price)

    # ── Validation & construction ─────────────────────────────────────────────

    def _build_candidate(
        self,
        policy_id: str,
        method: str,
        ctx: SlTpContext,
        sl_price: Decimal,
        tp_price: Decimal,
    ) -> SlTpCandidate | None:
        """Validate prices and construct a SlTpCandidate; return None if invalid."""
        # SL must be on the correct side of entry
        if ctx.side == OrderSide.BUY and sl_price >= ctx.entry_price:
            return None
        if ctx.side == OrderSide.SELL and sl_price <= ctx.entry_price:
            return None
        # TP must be on the correct side of entry
        if ctx.side == OrderSide.BUY and tp_price <= ctx.entry_price:
            return None
        if ctx.side == OrderSide.SELL and tp_price >= ctx.entry_price:
            return None

        sl_pips = ctx.price_to_pips(abs(ctx.entry_price - sl_price))
        tp_pips = ctx.price_to_pips(abs(ctx.entry_price - tp_price))

        # Minimum SL distance: must clear the spread plus the absolute minimum
        effective_min_sl = self._min_stop_pips + Decimal(str(max(ctx.spread_pips, 0.0)))
        if sl_pips < effective_min_sl:
            return None
        if tp_pips < self._min_tp_pips:
            return None
        if sl_pips <= 0:
            return None

        rr = (tp_pips / sl_pips).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if rr < self._min_rr:
            return None

        return SlTpCandidate(
            policy_id=policy_id,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            risk_reward=rr,
            method=method,
        )
