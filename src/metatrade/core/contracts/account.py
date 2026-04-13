"""AccountState contract.

A snapshot of the broker account at a point in time.
Updated after every fill, every mark-to-market cycle, and on reconnection.

All monetary values are in account currency (typically USD).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from metatrade.core.enums import RunMode


@dataclass(frozen=True)
class AccountState:
    """Immutable snapshot of account state.

    balance         Realized funds (deposits + closed trade PnL).
    equity          balance + unrealized PnL of all open positions.
    free_margin     Capital available to open new positions.
    used_margin     Capital currently locked as margin for open positions.
    timestamp_utc   When this snapshot was taken.
    currency        Account denomination (e.g. "USD").
    open_positions_count  Number of currently open positions.
    run_mode        Active run mode — recorded for audit clarity.
    """

    balance: Decimal
    equity: Decimal
    free_margin: Decimal
    used_margin: Decimal
    timestamp_utc: datetime
    currency: str = "USD"
    open_positions_count: int = 0
    run_mode: RunMode = RunMode.PAPER

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("AccountState.timestamp_utc must be timezone-aware (UTC)")
        if self.balance < 0:
            raise ValueError(f"AccountState.balance cannot be negative, got {self.balance}")
        if self.free_margin < 0:
            raise ValueError(
                f"AccountState.free_margin cannot be negative, got {self.free_margin}"
            )
        if self.used_margin < 0:
            raise ValueError(
                f"AccountState.used_margin cannot be negative, got {self.used_margin}"
            )
        if self.open_positions_count < 0:
            raise ValueError(
                f"AccountState.open_positions_count cannot be negative, "
                f"got {self.open_positions_count}"
            )
        if not self.currency:
            raise ValueError("AccountState.currency cannot be empty")

    # ── Derived metrics ───────────────────────────────────────────────────────

    @property
    def margin_level(self) -> Decimal | None:
        """Margin level as a percentage (equity / used_margin * 100).
        Returns None when no positions are open (used_margin == 0).
        Broker issues margin call when this drops below ~100%.
        """
        if self.used_margin == 0:
            return None
        return (self.equity / self.used_margin) * Decimal("100")

    @property
    def unrealized_pnl(self) -> Decimal:
        """Aggregate unrealized PnL across all open positions."""
        return self.equity - self.balance

    @property
    def drawdown_from_equity(self) -> Decimal:
        """How far equity is below balance (floating loss). Always >= 0."""
        return max(Decimal("0"), self.balance - self.equity)

    def __repr__(self) -> str:
        return (
            f"AccountState("
            f"balance={self.balance}, "
            f"equity={self.equity}, "
            f"free_margin={self.free_margin}, "
            f"mode={self.run_mode.value})"
        )
