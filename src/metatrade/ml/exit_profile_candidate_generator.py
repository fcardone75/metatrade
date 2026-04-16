"""Exit profile candidate generator.

Produces a finite set of plausible exit profiles for each signal context.
Three families of profiles are generated:

Fixed-TP profiles (exit_mode="fixed_tp")
    ATR_1_2      — SL=1×ATR,   TP=2×ATR   (R:R 2.0) ← default
    ATR_1_5_3    — SL=1.5×ATR, TP=4.5×ATR (R:R 3.0, wider stop / bigger target)
    SWING_2R     — SL at nearest swing level, TP=2×SL distance

Trailing-stop profiles (exit_mode="trailing_stop", tp_price=None)
    ATR_TRAIL_TIGHT    — SL=1×ATR,   no fixed TP, trailing_atr_mult=1.5
    ATR_TRAIL_STANDARD — SL=1.5×ATR, no fixed TP, trailing_atr_mult=2.0
    SWING_TRAIL        — SL at nearest swing level, no fixed TP,
                         trailing_atr_mult=2.0

Hybrid profiles (exit_mode="hybrid")
    ATR_HYBRID_1_5     — SL=1×ATR, TP=2×ATR as cap, trailing_atr_mult=1.0
                         ExitEngine trails aggressively and exits at TP at most.

Validation rules applied to every candidate (invalid ones are silently dropped):
- SL on the correct side of entry
- TP (when set) on the correct side of entry
- SL distance ≥ min_stop_pips + spread (prevents immediate stop-out)
- TP distance ≥ min_tp_pips (when TP is set)
- R:R ≥ min_rr

Sorting
-------
Results are sorted by R:R descending.  Within equal R:R, ATR profiles come
before swing profiles (more predictable distances).  trailing_stop and hybrid
profiles are included in the sort using their estimated R:R.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import NamedTuple

from metatrade.core.enums import OrderSide
from metatrade.ml.exit_profile_contracts import ExitProfileCandidate, ExitProfileContext


class _FixedTpSpec(NamedTuple):
    profile_id: str
    sl_atr_mult: float
    tp_atr_mult: float


class _TrailingSpec(NamedTuple):
    profile_id: str
    sl_atr_mult: float
    trailing_atr_mult: float
    estimated_rr: float   # empirical estimate for sorting


class _HybridSpec(NamedTuple):
    profile_id: str
    sl_atr_mult: float
    tp_atr_mult: float
    trailing_atr_mult: float


_FIXED_TP_SPECS: tuple[_FixedTpSpec, ...] = (
    _FixedTpSpec("ATR_1_2",    1.0, 2.0),
    _FixedTpSpec("ATR_1_5_3",  1.5, 4.5),
)

_TRAILING_SPECS: tuple[_TrailingSpec, ...] = (
    _TrailingSpec("ATR_TRAIL_TIGHT",    1.0, 1.5, 2.5),
    _TrailingSpec("ATR_TRAIL_STANDARD", 1.5, 2.0, 2.5),
)

_HYBRID_SPECS: tuple[_HybridSpec, ...] = (
    _HybridSpec("ATR_HYBRID_1_5", 1.0, 2.0, 1.0),
)

# Swing-based fixed-TP R:R ratios
_SWING_FIXED_TP_RR: float = 2.0
_SWING_FIXED_TP_ID = "SWING_2R"

# Swing trailing: no fixed TP, trailing at 2×ATR
_SWING_TRAIL_ID = "SWING_TRAIL"
_SWING_TRAIL_ATR_MULT = Decimal("2.0")
_SWING_TRAIL_ESTIMATED_RR = Decimal("3.0")


class ExitProfileCandidateGenerator:
    """Generates valid exit profile candidates from ATR, trailing, and swing policies.

    Args:
        min_stop_pips:       Minimum SL distance from entry in pips.  The actual
                             minimum is ``max(min_stop_pips, spread_pips + 1)``
                             to prevent immediate stop-out due to spread at fill.
        min_tp_pips:         Minimum TP distance for fixed-TP profiles.
        min_rr:              Minimum risk-reward ratio for fixed-TP profiles.
        swing_lookback:      Number of recent bars to scan for swing levels.
        swing_buffer_pips:   Buffer (in pips) added beyond the swing level.
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

    def generate(self, context: ExitProfileContext) -> list[ExitProfileCandidate]:
        """Generate all valid exit profile candidates for the given entry context.

        Args:
            context: Current signal context (ATR, spread, recent bars, …).

        Returns:
            List of valid candidates sorted by R:R descending.
            Empty list if no valid candidate can be constructed.
        """
        candidates: list[ExitProfileCandidate] = []

        # ── Fixed-TP ATR profiles ─────────────────────────────────────────────
        for spec in _FIXED_TP_SPECS:
            cand = self._make_atr_fixed_tp(spec, context)
            if cand is not None:
                candidates.append(cand)

        # ── Trailing ATR profiles ─────────────────────────────────────────────
        for spec in _TRAILING_SPECS:
            cand = self._make_atr_trailing(spec, context)
            if cand is not None:
                candidates.append(cand)

        # ── Hybrid ATR profiles ───────────────────────────────────────────────
        for spec in _HYBRID_SPECS:
            cand = self._make_atr_hybrid(spec, context)
            if cand is not None:
                candidates.append(cand)

        # ── Swing-based profiles ──────────────────────────────────────────────
        swing_sl = self._find_swing_sl(context)
        if swing_sl is not None:
            # Fixed-TP swing
            cand = self._make_swing_fixed_tp(context, swing_sl)
            if cand is not None:
                candidates.append(cand)
            # Trailing swing
            cand = self._make_swing_trailing(context, swing_sl)
            if cand is not None:
                candidates.append(cand)

        # Sort by R:R descending; within equal R:R prefer ATR over swing
        candidates.sort(
            key=lambda c: (
                c.risk_reward,
                1 if c.method in {"atr", "trailing"} else 0,
            ),
            reverse=True,
        )
        return candidates

    # ── ATR fixed-TP ─────────────────────────────────────────────────────────

    def _make_atr_fixed_tp(
        self, spec: _FixedTpSpec, ctx: ExitProfileContext
    ) -> ExitProfileCandidate | None:
        sl_dist = ctx.atr * Decimal(str(spec.sl_atr_mult))
        tp_dist = ctx.atr * Decimal(str(spec.tp_atr_mult))
        if ctx.side == OrderSide.BUY:
            sl_price = ctx.entry_price - sl_dist
            tp_price = ctx.entry_price + tp_dist
        else:
            sl_price = ctx.entry_price + sl_dist
            tp_price = ctx.entry_price - tp_dist
        return self._build_fixed_tp(
            profile_id=spec.profile_id,
            method="atr",
            ctx=ctx,
            sl_price=sl_price,
            tp_price=tp_price,
        )

    # ── ATR trailing ─────────────────────────────────────────────────────────

    def _make_atr_trailing(
        self, spec: _TrailingSpec, ctx: ExitProfileContext
    ) -> ExitProfileCandidate | None:
        sl_dist = ctx.atr * Decimal(str(spec.sl_atr_mult))
        if ctx.side == OrderSide.BUY:
            sl_price = ctx.entry_price - sl_dist
        else:
            sl_price = ctx.entry_price + sl_dist
        return self._build_trailing(
            profile_id=spec.profile_id,
            method="trailing",
            ctx=ctx,
            sl_price=sl_price,
            trailing_atr_mult=Decimal(str(spec.trailing_atr_mult)),
            estimated_rr=Decimal(str(spec.estimated_rr)),
        )

    # ── ATR hybrid ───────────────────────────────────────────────────────────

    def _make_atr_hybrid(
        self, spec: _HybridSpec, ctx: ExitProfileContext
    ) -> ExitProfileCandidate | None:
        sl_dist = ctx.atr * Decimal(str(spec.sl_atr_mult))
        tp_dist = ctx.atr * Decimal(str(spec.tp_atr_mult))
        if ctx.side == OrderSide.BUY:
            sl_price = ctx.entry_price - sl_dist
            tp_price = ctx.entry_price + tp_dist
        else:
            sl_price = ctx.entry_price + sl_dist
            tp_price = ctx.entry_price - tp_dist
        return self._build_hybrid(
            profile_id=spec.profile_id,
            method="atr",
            ctx=ctx,
            sl_price=sl_price,
            tp_price=tp_price,
            trailing_atr_mult=Decimal(str(spec.trailing_atr_mult)),
        )

    # ── Swing helpers ─────────────────────────────────────────────────────────

    def _find_swing_sl(self, ctx: ExitProfileContext) -> Decimal | None:
        """Identify the nearest significant swing level for the stop-loss.

        BUY: lowest low in the lookback window minus a buffer pip.
        SELL: highest high in the lookback window plus a buffer pip.

        Returns None when there is insufficient bar history.
        """
        lookback = self._swing_lookback
        buffer = self._swing_buffer_pips * ctx.pip_size

        if ctx.side == OrderSide.BUY:
            if len(ctx.recent_lows) < 3:
                return None
            lows = (
                ctx.recent_lows[-lookback:]
                if len(ctx.recent_lows) >= lookback
                else ctx.recent_lows
            )
            return min(lows) - buffer
        else:
            if len(ctx.recent_highs) < 3:
                return None
            highs = (
                ctx.recent_highs[-lookback:]
                if len(ctx.recent_highs) >= lookback
                else ctx.recent_highs
            )
            return max(highs) + buffer

    def _make_swing_fixed_tp(
        self, ctx: ExitProfileContext, swing_sl: Decimal
    ) -> ExitProfileCandidate | None:
        sl_dist = abs(ctx.entry_price - swing_sl)
        tp_dist = sl_dist * Decimal(str(_SWING_FIXED_TP_RR))
        if ctx.side == OrderSide.BUY:
            tp_price = ctx.entry_price + tp_dist
        else:
            tp_price = ctx.entry_price - tp_dist
        return self._build_fixed_tp(
            profile_id=_SWING_FIXED_TP_ID,
            method="swing",
            ctx=ctx,
            sl_price=swing_sl,
            tp_price=tp_price,
        )

    def _make_swing_trailing(
        self, ctx: ExitProfileContext, swing_sl: Decimal
    ) -> ExitProfileCandidate | None:
        return self._build_trailing(
            profile_id=_SWING_TRAIL_ID,
            method="swing",
            ctx=ctx,
            sl_price=swing_sl,
            trailing_atr_mult=_SWING_TRAIL_ATR_MULT,
            estimated_rr=_SWING_TRAIL_ESTIMATED_RR,
        )

    # ── Validation & construction ─────────────────────────────────────────────

    def _build_fixed_tp(
        self,
        profile_id: str,
        method: str,
        ctx: ExitProfileContext,
        sl_price: Decimal,
        tp_price: Decimal,
    ) -> ExitProfileCandidate | None:
        """Validate and construct a fixed-TP profile; None if validation fails."""
        if not self._sl_valid(ctx, sl_price):
            return None
        if ctx.side == OrderSide.BUY and tp_price <= ctx.entry_price:
            return None
        if ctx.side == OrderSide.SELL and tp_price >= ctx.entry_price:
            return None

        sl_pips = ctx.price_to_pips(abs(ctx.entry_price - sl_price))
        tp_pips = ctx.price_to_pips(abs(ctx.entry_price - tp_price))

        if not self._sl_pips_valid(sl_pips, ctx):
            return None
        if tp_pips < self._min_tp_pips:
            return None

        rr = (tp_pips / sl_pips).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if rr < self._min_rr:
            return None

        return ExitProfileCandidate(
            profile_id=profile_id,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            risk_reward=rr,
            method=method,
            exit_mode="fixed_tp",
            trailing_atr_mult=None,
        )

    def _build_trailing(
        self,
        profile_id: str,
        method: str,
        ctx: ExitProfileContext,
        sl_price: Decimal,
        trailing_atr_mult: Decimal,
        estimated_rr: Decimal,
    ) -> ExitProfileCandidate | None:
        """Validate and construct a trailing-stop profile; None if invalid."""
        if not self._sl_valid(ctx, sl_price):
            return None
        sl_pips = ctx.price_to_pips(abs(ctx.entry_price - sl_price))
        if not self._sl_pips_valid(sl_pips, ctx):
            return None

        return ExitProfileCandidate(
            profile_id=profile_id,
            sl_price=sl_price,
            tp_price=None,
            sl_pips=sl_pips,
            tp_pips=None,
            risk_reward=estimated_rr,
            method=method,
            exit_mode="trailing_stop",
            trailing_atr_mult=trailing_atr_mult,
        )

    def _build_hybrid(
        self,
        profile_id: str,
        method: str,
        ctx: ExitProfileContext,
        sl_price: Decimal,
        tp_price: Decimal,
        trailing_atr_mult: Decimal,
    ) -> ExitProfileCandidate | None:
        """Validate and construct a hybrid profile; None if invalid."""
        if not self._sl_valid(ctx, sl_price):
            return None
        if ctx.side == OrderSide.BUY and tp_price <= ctx.entry_price:
            return None
        if ctx.side == OrderSide.SELL and tp_price >= ctx.entry_price:
            return None

        sl_pips = ctx.price_to_pips(abs(ctx.entry_price - sl_price))
        tp_pips = ctx.price_to_pips(abs(ctx.entry_price - tp_price))

        if not self._sl_pips_valid(sl_pips, ctx):
            return None
        if tp_pips < self._min_tp_pips:
            return None

        rr = (tp_pips / sl_pips).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if rr < self._min_rr:
            return None

        return ExitProfileCandidate(
            profile_id=profile_id,
            sl_price=sl_price,
            tp_price=tp_price,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            risk_reward=rr,
            method=method,
            exit_mode="hybrid",
            trailing_atr_mult=trailing_atr_mult,
        )

    # ── Shared validation helpers ─────────────────────────────────────────────

    def _sl_valid(self, ctx: ExitProfileContext, sl_price: Decimal) -> bool:
        """SL must be on the correct side of the entry price."""
        if ctx.side == OrderSide.BUY and sl_price >= ctx.entry_price:
            return False
        if ctx.side == OrderSide.SELL and sl_price <= ctx.entry_price:
            return False
        return True

    def _sl_pips_valid(self, sl_pips: Decimal, ctx: ExitProfileContext) -> bool:
        """SL distance must exceed the spread-adjusted minimum and be positive."""
        if sl_pips <= 0:
            return False
        effective_min = self._min_stop_pips + Decimal(
            str(max(ctx.spread_pips, 0.0))
        )
        return sl_pips >= effective_min
