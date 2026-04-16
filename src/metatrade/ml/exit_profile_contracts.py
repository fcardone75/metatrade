"""Domain contracts for the exit profile selection module.

Three immutable dataclasses:

ExitProfileCandidate  — one concrete exit profile with initial SL and optional TP.
ExitProfileContext    — input context for candidate generation / selection.
SelectedExitProfile   — the chosen candidate plus selector metadata.

Key design decisions
--------------------
tp_price is Optional.  When None, no fixed take-profit is placed at entry —
instead the ExitEngine's trailing / dynamic rules are the primary exit
mechanism.  This supports profiles like pure trailing-stop strategies where
locking a TP would cap the upside prematurely.

exit_mode signals to the ExitEngine how to configure rule weights:
  "fixed_tp"      — TP is the primary target; trailing / time rules secondary.
  "trailing_stop" — No fixed TP; trailing_stop rule is the primary exit.
  "hybrid"        — TP acts as a cap; trailing_stop is also active.

trailing_atr_mult is an optional hint passed to the ExitEngine trailing_stop
rule so it knows how tightly (or loosely) to ratchet the stop.  It has no
effect for "fixed_tp" profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from metatrade.core.enums import OrderSide

_VALID_EXIT_MODES = frozenset({"fixed_tp", "trailing_stop", "hybrid"})
_VALID_METHODS = frozenset({"atr", "swing", "trailing"})


@dataclass(frozen=True)
class ExitProfileCandidate:
    """One candidate exit profile produced by ExitProfileCandidateGenerator.

    Attributes:
        profile_id:        Machine-readable identifier, e.g. ``"ATR_1_2"``.
        sl_price:          Absolute stop-loss price (Decimal, always set).
        tp_price:          Absolute take-profit price (Decimal), or ``None``
                           when there is no fixed TP and the ExitEngine manages
                           the exit dynamically (trailing / time-based).
        sl_pips:           SL distance from entry in pips (always > 0).
        tp_pips:           TP distance from entry in pips; ``None`` when
                           ``tp_price`` is ``None``.
        risk_reward:       Estimated R:R.  For fixed-TP profiles this is exact
                           (tp_pips / sl_pips).  For trailing / hybrid profiles
                           this is an empirical estimate based on backtested
                           average outcome.
        method:            Derivation method: ``"atr"``, ``"swing"``, or
                           ``"trailing"`` (pure trailing, no fixed TP).
        exit_mode:         How the ExitEngine should manage the trade after
                           entry: ``"fixed_tp"`` | ``"trailing_stop"`` |
                           ``"hybrid"``.
        trailing_atr_mult: ATR multiplier hint for the ExitEngine
                           ``trailing_stop`` rule.  ``None`` for fixed-TP
                           profiles where trailing is not the primary driver.
    """

    profile_id: str
    sl_price: Decimal
    tp_price: Decimal | None
    sl_pips: Decimal
    tp_pips: Decimal | None
    risk_reward: Decimal
    method: str
    exit_mode: str
    trailing_atr_mult: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("ExitProfileCandidate.profile_id cannot be empty")
        if self.sl_pips <= 0:
            raise ValueError(
                f"ExitProfileCandidate.sl_pips must be > 0, got {self.sl_pips}"
            )
        if self.tp_pips is not None and self.tp_pips <= 0:
            raise ValueError(
                f"ExitProfileCandidate.tp_pips must be > 0 when set, "
                f"got {self.tp_pips}"
            )
        if self.risk_reward <= 0:
            raise ValueError(
                f"ExitProfileCandidate.risk_reward must be > 0, "
                f"got {self.risk_reward}"
            )
        if self.exit_mode not in _VALID_EXIT_MODES:
            raise ValueError(
                f"ExitProfileCandidate.exit_mode must be one of "
                f"{sorted(_VALID_EXIT_MODES)}, got {self.exit_mode!r}"
            )
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"ExitProfileCandidate.method must be one of "
                f"{sorted(_VALID_METHODS)}, got {self.method!r}"
            )
        # Consistency: trailing_stop / hybrid profiles should carry a hint
        # (not enforced as a hard error — callers may omit for hybrid profiles
        # that rely solely on the fixed TP cap).


@dataclass(frozen=True)
class ExitProfileContext:
    """Input context for exit profile candidate generation and selection.

    Built once per signal and passed to both the generator and the selector.
    All fields needed to validate, compute, and rank candidates are included.

    Attributes:
        symbol:          Forex symbol (e.g. ``"EURUSD"``).
        side:            ``OrderSide.BUY`` or ``OrderSide.SELL``.
        entry_price:     Intended entry price (bar close).
        atr:             ATR(14) in price units at signal time (> 0).
        spread_pips:     Current bid-ask spread in pips (>= 0).
        account_balance: Account balance in account currency.
        timestamp_utc:   UTC timestamp of the bar that triggered the signal.
        pip_digits:      Decimal places for 1 pip: 4 for EURUSD, 2 for USDJPY.
        recent_highs:    Bar high prices (oldest first, newest last, ≤ 200 bars).
        recent_lows:     Bar low prices (oldest first, newest last, ≤ 200 bars).
    """

    symbol: str
    side: OrderSide
    entry_price: Decimal
    atr: Decimal
    spread_pips: float
    account_balance: Decimal
    timestamp_utc: datetime
    pip_digits: int = 4
    # Tuples are immutable and hashable — safe in frozen dataclasses.
    recent_highs: tuple[Decimal, ...] = field(default_factory=tuple)
    recent_lows: tuple[Decimal, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.atr <= 0:
            raise ValueError(
                f"ExitProfileContext.atr must be > 0, got {self.atr}"
            )
        if self.spread_pips < 0:
            raise ValueError(
                f"ExitProfileContext.spread_pips must be >= 0, "
                f"got {self.spread_pips}"
            )
        if self.pip_digits < 1:
            raise ValueError(
                f"ExitProfileContext.pip_digits must be >= 1, "
                f"got {self.pip_digits}"
            )

    @property
    def pip_size(self) -> Decimal:
        """Price equivalent of 1 pip (0.0001 for EURUSD, 0.01 for USDJPY)."""
        return Decimal("10") ** -self.pip_digits

    def price_to_pips(self, distance: Decimal) -> Decimal:
        """Convert an absolute price distance to pips (1 d.p.)."""
        return (abs(distance) / self.pip_size).quantize(Decimal("0.1"))


@dataclass(frozen=True)
class SelectedExitProfile:
    """Output of ExitProfileSelector: the chosen profile plus selector metadata.

    Attributes:
        candidate:    The selected ``ExitProfileCandidate``.
        confidence:   Selector confidence ∈ [0, 1].  Always 1.0 for
                      deterministic fallback; model predicted probability for ML.
        method:       ``"ml_model"`` or ``"deterministic_fallback"``.
        reason:       Human-readable explanation of the selection.
        alternatives: Other valid candidates that were not selected
                      (useful for observability and debugging).
    """

    candidate: ExitProfileCandidate
    confidence: float
    method: str
    reason: str
    alternatives: tuple[ExitProfileCandidate, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"SelectedExitProfile.confidence must be in [0,1], "
                f"got {self.confidence}"
            )
        if not self.method:
            raise ValueError("SelectedExitProfile.method cannot be empty")
        if not self.reason:
            raise ValueError("SelectedExitProfile.reason cannot be empty")
