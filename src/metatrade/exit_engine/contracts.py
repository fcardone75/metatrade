"""Exit engine data contracts.

PositionContext  — snapshot of an open position + current market state.
ExitSignal       — output of a single exit rule evaluation.
ExitDecision     — aggregated decision from all rules (what the engine returns).
TradeOutcome     — post-close record used to update rule reputations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class ExitAction(str, Enum):
    """Action the exit engine recommends for an open position."""
    HOLD          = "HOLD"
    CLOSE_PARTIAL = "CLOSE_PARTIAL"
    CLOSE_FULL    = "CLOSE_FULL"


class PositionSide(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class PositionContext:
    """All information a rule needs to evaluate an open position.

    Fields:
        position_id         Unique ID (from broker or internal).
        symbol              e.g. "EURUSD".
        side                LONG or SHORT.
        entry_price         Fill price.
        lot_size            Position size in standard lots.
        opened_at_utc       UTC timestamp of entry fill.
        current_price       Latest market price (close of most recent bar).
        current_bar         Most recently closed bar.
        bars_since_entry    All closed bars since the position was opened
                            (oldest first, newest last).  May be empty if the
                            position was just opened this bar.
        peak_favorable_price Highest price seen for LONG, lowest for SHORT,
                             since position opened.  Used by give-back rule.
        entry_atr           ATR value at the moment of entry (for volatility
                             exit and ATR-based trailing stop).  None if not
                             available.
        stop_loss           Current stop-loss price (may be updated by engine).
        take_profit         Original take-profit price (None if not set).
        partial_closes      List of (pips_profit, close_pct) already executed.
    """

    position_id:           str
    symbol:                str
    side:                  PositionSide
    entry_price:           Decimal
    lot_size:              Decimal
    opened_at_utc:         datetime
    current_price:         Decimal
    current_bar:           Any        # Bar — typed as Any to avoid circular import
    bars_since_entry:      list[Any]  # list[Bar]
    peak_favorable_price:  Decimal
    entry_atr:             Decimal | None = None
    stop_loss:             Decimal | None = None
    take_profit:           Decimal | None = None
    partial_closes:        tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.opened_at_utc.tzinfo is None:
            raise ValueError("PositionContext.opened_at_utc must be timezone-aware")
        if self.lot_size <= 0:
            raise ValueError(f"PositionContext.lot_size must be > 0, got {self.lot_size}")
        if self.entry_price <= 0:
            raise ValueError(f"PositionContext.entry_price must be > 0, got {self.entry_price}")

    @property
    def unrealized_pips(self) -> Decimal:
        """Floating P&L in pips (positive = profit)."""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) / Decimal("0.0001")
        return (self.entry_price - self.current_price) / Decimal("0.0001")

    @property
    def bars_held(self) -> int:
        return len(self.bars_since_entry)

    @property
    def peak_profit_pips(self) -> Decimal:
        """Maximum profit in pips seen since entry."""
        if self.side == PositionSide.LONG:
            return (self.peak_favorable_price - self.entry_price) / Decimal("0.0001")
        return (self.entry_price - self.peak_favorable_price) / Decimal("0.0001")


@dataclass(frozen=True)
class ExitSignal:
    """Output of a single exit rule evaluation.

    Fields:
        rule_id     Stable identifier for the rule (e.g. "trailing_stop").
        action      HOLD, CLOSE_PARTIAL, or CLOSE_FULL.
        close_pct   Fraction of the position to close (0.0 = hold, 1.0 = full).
        reason      Human-readable explanation for logs and the dashboard.
        confidence  [0, 1] — how certain the rule is (used in weighted scoring).
        metadata    Optional extra data (indicator values, thresholds hit, etc.).
    """

    rule_id:    str
    action:     ExitAction
    close_pct:  float       # 0.0–1.0
    reason:     str
    confidence: float       # [0, 1]
    metadata:   dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.close_pct <= 1.0:
            raise ValueError(f"ExitSignal.close_pct must be in [0, 1], got {self.close_pct}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"ExitSignal.confidence must be in [0, 1], got {self.confidence}")
        if not self.rule_id:
            raise ValueError("ExitSignal.rule_id cannot be empty")
        if not self.reason:
            raise ValueError("ExitSignal.reason cannot be empty")


@dataclass(frozen=True)
class ExitDecision:
    """Aggregated exit decision produced by the ExitEngine.

    Fields:
        action              Final recommendation.
        close_pct           Fraction to close (1.0 for CLOSE_FULL).
        aggregate_score     Weighted vote score in [0, 100].
        signals             All signals from individual rules.
        explanation         Human-readable summary.
        timestamp_utc       When the decision was produced.
        new_stop_loss       Updated SL price (e.g. from trailing stop),
                            None if unchanged.
    """

    action:          ExitAction
    close_pct:       float
    aggregate_score: float
    signals:         tuple[ExitSignal, ...]
    explanation:     str
    timestamp_utc:   datetime
    new_stop_loss:   Decimal | None = None

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("ExitDecision.timestamp_utc must be timezone-aware")

    @property
    def is_exit(self) -> bool:
        return self.action != ExitAction.HOLD


@dataclass(frozen=True)
class TradeOutcome:
    """Post-close record for updating rule reputations.

    Produced by the caller after a position is fully closed.

    Fields:
        position_id         Links back to PositionContext.
        symbol              Forex pair.
        entry_price         Fill price at open.
        exit_price          Fill price at close.
        side                LONG or SHORT.
        lot_size            Size at entry.
        opened_at_utc       Entry timestamp.
        closed_at_utc       Close timestamp.
        exit_reason         "sl", "tp", "trailing_sl", "engine", "manual".
        signals_emitted     All ExitSignals produced during the trade lifecycle,
                            paired with their bar's close price at the time they
                            were emitted.  List of (ExitSignal, price_at_signal).
        pnl_pips            Final realized P&L in pips.
        pip_value_per_lot   For converting pips to currency.
    """

    position_id:       str
    symbol:            str
    entry_price:       Decimal
    exit_price:        Decimal
    side:              PositionSide
    lot_size:          Decimal
    opened_at_utc:     datetime
    closed_at_utc:     datetime
    exit_reason:       str
    signals_emitted:   tuple[tuple[ExitSignal, Decimal], ...]  # (signal, price_when_emitted)
    pnl_pips:          Decimal
    pip_value_per_lot: Decimal = Decimal("10")

    @property
    def pnl_currency(self) -> Decimal:
        return self.pnl_pips * self.lot_size * self.pip_value_per_lot

    @property
    def was_profitable(self) -> bool:
        return self.pnl_pips > 0
