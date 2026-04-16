"""Domain contracts for the intelligent SL/TP policy selection module.

Three immutable dataclasses:

SlTpCandidate      — one concrete (SL, TP) pair with metadata.
SlTpContext        — input context for candidate generation / selection.
SelectedSlTpPolicy — the chosen candidate plus selector metadata.

All prices are Decimal.  Pips are Decimal with one decimal place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from metatrade.core.enums import OrderSide


@dataclass(frozen=True)
class SlTpCandidate:
    """One concrete SL/TP pair produced by SlTpCandidateGenerator.

    Attributes:
        policy_id:   Machine-readable identifier, e.g. ``"ATR_1_2"``.
        sl_price:    Absolute stop-loss price (Decimal).
        tp_price:    Absolute take-profit price (Decimal).
        sl_pips:     SL distance from entry in pips (always > 0).
        tp_pips:     TP distance from entry in pips (always > 0).
        risk_reward: tp_pips / sl_pips (e.g. ``Decimal("2.0")`` for 1:2).
        method:      Derivation method: ``"atr"`` or ``"swing"``.
    """

    policy_id: str
    sl_price: Decimal
    tp_price: Decimal
    sl_pips: Decimal
    tp_pips: Decimal
    risk_reward: Decimal
    method: str

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("SlTpCandidate.policy_id cannot be empty")
        if self.sl_pips <= 0:
            raise ValueError(f"SlTpCandidate.sl_pips must be > 0, got {self.sl_pips}")
        if self.tp_pips <= 0:
            raise ValueError(f"SlTpCandidate.tp_pips must be > 0, got {self.tp_pips}")
        if self.risk_reward <= 0:
            raise ValueError(
                f"SlTpCandidate.risk_reward must be > 0, got {self.risk_reward}"
            )


@dataclass(frozen=True)
class SlTpContext:
    """Input context for SL/TP candidate generation and policy selection.

    The context is built once per signal and passed to both the generator
    and the selector.  All fields needed to validate, compute, and rank
    candidates are included here.

    Attributes:
        symbol:          Forex symbol (e.g. ``"EURUSD"``).
        side:            ``OrderSide.BUY`` or ``OrderSide.SELL``.
        entry_price:     Intended entry price (bar close).
        atr:             ATR(14) in price units at signal time (> 0).
        spread_pips:     Current bid-ask spread in pips (>= 0).
        account_balance: Account balance in account currency.
        timestamp_utc:   UTC timestamp of the bar that triggered the signal.
        pip_digits:      Decimal places for 1 pip: 4 for EURUSD, 2 for USDJPY.
        recent_highs:    Bar high prices (oldest first, newest last, <= 200 bars).
        recent_lows:     Bar low prices (oldest first, newest last, <= 200 bars).
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
            raise ValueError(f"SlTpContext.atr must be > 0, got {self.atr}")
        if self.spread_pips < 0:
            raise ValueError(
                f"SlTpContext.spread_pips must be >= 0, got {self.spread_pips}"
            )
        if self.pip_digits < 1:
            raise ValueError(
                f"SlTpContext.pip_digits must be >= 1, got {self.pip_digits}"
            )

    @property
    def pip_size(self) -> Decimal:
        """Price equivalent of 1 pip (e.g. 0.0001 for EURUSD, 0.01 for USDJPY)."""
        return Decimal("10") ** -self.pip_digits

    def price_to_pips(self, distance: Decimal) -> Decimal:
        """Convert an absolute price distance to pips (1 d.p.)."""
        return (abs(distance) / self.pip_size).quantize(Decimal("0.1"))


@dataclass(frozen=True)
class SelectedSlTpPolicy:
    """Output of SlTpPolicySelector: the chosen candidate plus selector metadata.

    Attributes:
        candidate:    The selected ``SlTpCandidate``.
        confidence:   Selector confidence in [0, 1].  Always 1.0 for
                      deterministic fallback; model predicted probability for ML.
        method:       ``"ml_model"`` or ``"deterministic_fallback"``.
        reason:       Human-readable explanation of the selection.
        alternatives: Other valid candidates that were not selected
                      (useful for observability and debugging).
    """

    candidate: SlTpCandidate
    confidence: float
    method: str
    reason: str
    alternatives: tuple[SlTpCandidate, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"SelectedSlTpPolicy.confidence must be in [0,1], got {self.confidence}"
            )
        if not self.method:
            raise ValueError("SelectedSlTpPolicy.method cannot be empty")
        if not self.reason:
            raise ValueError("SelectedSlTpPolicy.reason cannot be empty")
