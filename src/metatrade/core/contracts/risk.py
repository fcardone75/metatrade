"""Risk management contracts.

PositionSizeResult  — output of the position sizer
RiskVeto            — reason a trade was blocked by the risk manager
RiskDecision        — approved or vetoed trade, with full context
KillSwitchState     — current kill switch level and activation reason
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from metatrade.core.enums import KillSwitchLevel, OrderSide


@dataclass(frozen=True)
class PositionSizeResult:
    """Output of the position sizing calculation.

    lot_size        Calculated lot size to trade.
    risk_amount     Maximum loss in account currency if SL is hit.
    risk_pct        risk_amount / account_balance (e.g. 0.01 = 1%).
    stop_loss_price Absolute price level for the stop-loss order.
    stop_loss_pips  Distance from entry to SL in pips.
    take_profit_price  Absolute price level for the take-profit (None = no TP).
    pip_value       Value of 1 pip per lot in account currency.
    method          Name of sizing method used ("fixed_fractional", etc.).
    """

    lot_size: Decimal
    risk_amount: Decimal
    risk_pct: float
    stop_loss_price: Decimal
    stop_loss_pips: Decimal
    pip_value: Decimal
    method: str
    take_profit_price: Decimal | None = None

    def __post_init__(self) -> None:
        if self.lot_size <= 0:
            raise ValueError(f"PositionSizeResult.lot_size must be > 0, got {self.lot_size}")
        if self.risk_amount < 0:
            raise ValueError(
                f"PositionSizeResult.risk_amount cannot be negative, got {self.risk_amount}"
            )
        if not 0.0 <= self.risk_pct <= 1.0:
            raise ValueError(
                f"PositionSizeResult.risk_pct must be in [0.0, 1.0], got {self.risk_pct}"
            )
        if self.stop_loss_pips <= 0:
            raise ValueError(
                f"PositionSizeResult.stop_loss_pips must be > 0, got {self.stop_loss_pips}"
            )
        if self.pip_value <= 0:
            raise ValueError(
                f"PositionSizeResult.pip_value must be > 0, got {self.pip_value}"
            )
        if not self.method:
            raise ValueError("PositionSizeResult.method cannot be empty")


@dataclass(frozen=True)
class RiskVeto:
    """Explanation for why the risk manager blocked a trade.

    veto_code       Machine-readable code for filtering/alerting.
                    Examples: "MAX_POSITIONS", "DAILY_LOSS_LIMIT",
                              "SPREAD_TOO_WIDE", "KILL_SWITCH_ACTIVE".
    reason          Human-readable description.
    context         Optional structured data for debugging.
    """

    reason: str
    veto_code: str
    timestamp_utc: datetime
    context: dict = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("RiskVeto.timestamp_utc must be timezone-aware (UTC)")
        if not self.reason:
            raise ValueError("RiskVeto.reason cannot be empty")
        if not self.veto_code:
            raise ValueError("RiskVeto.veto_code cannot be empty")

    def __repr__(self) -> str:
        return f"RiskVeto(code={self.veto_code!r}, reason={self.reason!r})"


@dataclass(frozen=True)
class RiskDecision:
    """Final output of the risk manager for a proposed trade.

    Exactly one of `position_size` or `veto` is set:
    - approved=True  → position_size is set, veto is None
    - approved=False → veto is set, position_size is None
    """

    approved: bool
    symbol: str
    side: OrderSide
    timestamp_utc: datetime
    position_size: PositionSizeResult | None = None
    veto: RiskVeto | None = None

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("RiskDecision.timestamp_utc must be timezone-aware (UTC)")
        if self.approved and self.position_size is None:
            raise ValueError("Approved RiskDecision must have position_size set")
        if self.approved and self.veto is not None:
            raise ValueError("Approved RiskDecision must not have a veto")
        if not self.approved and self.veto is None:
            raise ValueError("Rejected RiskDecision must have veto set")
        if not self.approved and self.position_size is not None:
            raise ValueError("Rejected RiskDecision must not have position_size")
        if not self.symbol:
            raise ValueError("RiskDecision.symbol cannot be empty")

    def __repr__(self) -> str:
        if self.approved:
            return (
                f"RiskDecision(approved=True, "
                f"symbol={self.symbol!r}, "
                f"lots={self.position_size.lot_size})"  # type: ignore[union-attr]
            )
        return (
            f"RiskDecision(approved=False, "
            f"symbol={self.symbol!r}, "
            f"veto={self.veto!r})"
        )


@dataclass(frozen=True)
class KillSwitchState:
    """Current state of the kill switch subsystem."""

    level: KillSwitchLevel
    reason: str
    activated_at_utc: datetime
    activated_by: str   # "system", "manual", or the module/component name

    def __post_init__(self) -> None:
        if self.activated_at_utc.tzinfo is None:
            raise ValueError("KillSwitchState.activated_at_utc must be timezone-aware (UTC)")
        if not self.reason:
            raise ValueError("KillSwitchState.reason cannot be empty")
        if not self.activated_by:
            raise ValueError("KillSwitchState.activated_by cannot be empty")

    @property
    def is_active(self) -> bool:
        return self.level > KillSwitchLevel.NONE

    @property
    def blocks_new_trades(self) -> bool:
        return self.level >= KillSwitchLevel.TRADE_GATE

    @property
    def blocks_all_activity(self) -> bool:
        return self.level >= KillSwitchLevel.EMERGENCY_HALT

    def __repr__(self) -> str:
        return (
            f"KillSwitchState("
            f"level={self.level.name}, "
            f"reason={self.reason!r}, "
            f"by={self.activated_by!r})"
        )
