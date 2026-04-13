"""Position contract.

A Position represents an open or closed trade on a specific symbol.
Like Order, positions are immutable — state changes produce new objects.

PnL calculations return None when the required prices are unavailable,
rather than raising exceptions, to support partial data scenarios gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from metatrade.core.enums import PositionSide, PositionState


@dataclass(frozen=True)
class Position:
    """A single open or closed trade position."""

    position_id: str
    symbol: str
    side: PositionSide
    lot_size: Decimal
    entry_price: Decimal
    opened_at_utc: datetime
    state: PositionState = PositionState.OPEN
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    current_price: Decimal | None = None
    closed_at_utc: datetime | None = None
    close_price: Decimal | None = None
    swap_accumulated: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    broker_position_id: str | None = None

    def __post_init__(self) -> None:
        if self.opened_at_utc.tzinfo is None:
            raise ValueError("Position.opened_at_utc must be timezone-aware (UTC)")
        if self.closed_at_utc is not None and self.closed_at_utc.tzinfo is None:
            raise ValueError("Position.closed_at_utc must be timezone-aware (UTC)")
        if self.lot_size <= 0:
            raise ValueError(f"Position.lot_size must be > 0, got {self.lot_size}")
        if self.entry_price <= 0:
            raise ValueError(f"Position.entry_price must be > 0, got {self.entry_price}")
        if self.state == PositionState.CLOSED:
            if self.closed_at_utc is None:
                raise ValueError("Closed Position must have closed_at_utc")
            if self.close_price is None:
                raise ValueError("Closed Position must have close_price")
        if self.commission < 0:
            raise ValueError(f"Position.commission cannot be negative, got {self.commission}")

    # ── PnL ───────────────────────────────────────────────────────────────────

    @property
    def unrealized_pnl_pips(self) -> Decimal | None:
        """Unrealized PnL in pips. None if current_price is unavailable."""
        if self.current_price is None:
            return None
        if self.side == PositionSide.LONG:
            return self.current_price - self.entry_price
        return self.entry_price - self.current_price

    @property
    def realized_pnl_pips(self) -> Decimal | None:
        """Realized PnL in pips (net of nothing — caller converts to currency).
        None if position is not closed.
        """
        if self.close_price is None:
            return None
        if self.side == PositionSide.LONG:
            return self.close_price - self.entry_price
        return self.entry_price - self.close_price

    # ── State helpers ─────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self.state == PositionState.OPEN

    @property
    def is_closed(self) -> bool:
        return self.state == PositionState.CLOSED

    def with_current_price(self, price: Decimal) -> Position:
        """Return a new Position with updated current_price (mark-to-market)."""
        return Position(
            position_id=self.position_id,
            symbol=self.symbol,
            side=self.side,
            lot_size=self.lot_size,
            entry_price=self.entry_price,
            opened_at_utc=self.opened_at_utc,
            state=self.state,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            current_price=price,
            closed_at_utc=self.closed_at_utc,
            close_price=self.close_price,
            swap_accumulated=self.swap_accumulated,
            commission=self.commission,
            broker_position_id=self.broker_position_id,
        )

    def close(self, close_price: Decimal, closed_at_utc: datetime) -> Position:
        """Return a new Position in CLOSED state."""
        return Position(
            position_id=self.position_id,
            symbol=self.symbol,
            side=self.side,
            lot_size=self.lot_size,
            entry_price=self.entry_price,
            opened_at_utc=self.opened_at_utc,
            state=PositionState.CLOSED,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            current_price=close_price,
            closed_at_utc=closed_at_utc,
            close_price=close_price,
            swap_accumulated=self.swap_accumulated,
            commission=self.commission,
            broker_position_id=self.broker_position_id,
        )

    def __repr__(self) -> str:
        return (
            f"Position("
            f"id={self.position_id[:8]}..., "
            f"symbol={self.symbol!r}, "
            f"side={self.side.value}, "
            f"lots={self.lot_size}, "
            f"entry={self.entry_price}, "
            f"state={self.state.value})"
        )
