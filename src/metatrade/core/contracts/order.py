"""Order and Fill contracts.

Order  — an instruction to buy or sell, before it is sent to the broker.
Fill   — the confirmed execution record returned by the broker.

An Order is immutable once created. State transitions are represented by
creating a new Order with the updated status (functional update pattern).
This makes the audit trail trivial: every state is a new immutable object.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from metatrade.core.enums import OrderSide, OrderStatus, OrderType


@dataclass(frozen=True)
class Order:
    """An order instruction, from creation through to terminal state.

    order_id        Internal UUID assigned at creation time.
    broker_order_id Assigned by the broker after submission (None until then).
    price           Required for LIMIT and STOP orders; None for MARKET.
    stop_loss       Price level for the stop-loss, in quote currency.
    take_profit     Price level for the take-profit, in quote currency.
    comment         Free text — passed through to the broker if supported.
    """

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    lot_size: Decimal
    timestamp_utc: datetime
    status: OrderStatus = OrderStatus.PENDING
    price: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    slippage_pips: Decimal | None = None
    broker_order_id: str | None = None
    comment: str = ""

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("Order.timestamp_utc must be timezone-aware (UTC)")
        if self.lot_size <= 0:
            raise ValueError(f"Order.lot_size must be > 0, got {self.lot_size}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT):
            if self.price is None:
                raise ValueError(
                    f"Order.price is required for {self.order_type.value} orders"
                )
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError(f"Order.stop_loss must be > 0, got {self.stop_loss}")
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError(f"Order.take_profit must be > 0, got {self.take_profit}")

    @staticmethod
    def new_id() -> str:
        """Generate a new unique order ID."""
        return str(uuid.uuid4())

    def with_status(self, status: OrderStatus, broker_order_id: str | None = None) -> Order:
        """Return a new Order with updated status (functional update)."""
        return Order(
            order_id=self.order_id,
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            lot_size=self.lot_size,
            timestamp_utc=self.timestamp_utc,
            status=status,
            price=self.price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            slippage_pips=self.slippage_pips,
            broker_order_id=broker_order_id or self.broker_order_id,
            comment=self.comment,
        )

    def __repr__(self) -> str:
        return (
            f"Order("
            f"id={self.order_id[:8]}..., "
            f"symbol={self.symbol!r}, "
            f"side={self.side.value}, "
            f"type={self.order_type.value}, "
            f"lots={self.lot_size}, "
            f"status={self.status.value})"
        )


@dataclass(frozen=True)
class Fill:
    """Confirmed execution record returned by the broker.

    Slippage is the difference between the requested price and filled_price,
    expressed in pips. Positive = filled worse than requested (normal).
    Negative = filled better than requested (price improvement, rare).
    """

    order_id: str
    broker_order_id: str
    symbol: str
    side: OrderSide
    filled_price: Decimal
    filled_lots: Decimal
    commission: Decimal
    timestamp_utc: datetime
    slippage_pips: Decimal = Decimal("0")
    swap: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("Fill.timestamp_utc must be timezone-aware (UTC)")
        if self.filled_price <= 0:
            raise ValueError(f"Fill.filled_price must be > 0, got {self.filled_price}")
        if self.filled_lots <= 0:
            raise ValueError(f"Fill.filled_lots must be > 0, got {self.filled_lots}")
        if self.commission < 0:
            raise ValueError(f"Fill.commission cannot be negative, got {self.commission}")

    @property
    def total_cost(self) -> Decimal:
        """Total cost of the fill: commission + swap."""
        return self.commission + self.swap
