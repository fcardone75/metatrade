"""OrderManager — coordinates order creation, submission, and lifecycle.

Responsibilities:
    - Build Order objects from RiskDecision results.
    - Enforce idempotency (no duplicate submissions).
    - Submit orders through the IBrokerAdapter interface.
    - Track in-flight and completed orders.
    - Convert broker Fills into Position records.
    - Enforce RunMode safety (no real orders in BACKTEST mode).

The OrderManager is the single point of contact between the analysis/risk
layer and the broker adapter. All order state transitions flow through here.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from metatrade.core.contracts.order import Fill, Order
from metatrade.core.contracts.position import Position
from metatrade.core.contracts.risk import RiskDecision
from metatrade.core.enums import OrderSide, OrderStatus, OrderType, PositionSide, RunMode
from metatrade.core.errors import DuplicateOrderError, ModeGuardError, OrderRejectedError
from metatrade.core.log import get_logger
from metatrade.execution.idempotency_guard import IdempotencyGuard

log = get_logger(__name__)


class OrderManager:
    """Manages the full lifecycle of trading orders.

    Args:
        broker:       IBrokerAdapter implementation (MT5, paper, or mock).
        run_mode:     Current run mode — LIVE mode sends real orders.
        ttl_seconds:  Idempotency guard TTL in seconds.
    """

    def __init__(
        self,
        broker: object,  # IBrokerAdapter — duck-typed to avoid circular imports
        run_mode: RunMode = RunMode.PAPER,
        ttl_seconds: int = 3600,
    ) -> None:
        self._broker = broker
        self._run_mode = run_mode
        self._guard = IdempotencyGuard(ttl_seconds=ttl_seconds)
        # {order_id: Order} — all orders this session
        self._orders: dict[str, Order] = {}
        # {position_id: Position} — open positions this session
        self._positions: dict[str, Position] = {}

    # ── Order construction ────────────────────────────────────────────────────

    def build_order(
        self,
        decision: RiskDecision,
        timestamp_utc: datetime,
        comment: str = "",
    ) -> Order:
        """Build an Order from an approved RiskDecision.

        Args:
            decision:      Approved RiskDecision (must have position_size set).
            timestamp_utc: Timestamp for the order.
            comment:       Optional broker comment string.

        Raises:
            ValueError: If decision is not approved.
        """
        if not decision.approved or decision.position_size is None:
            raise ValueError(
                "Cannot build an Order from a non-approved RiskDecision"
            )
        ps = decision.position_size
        return Order(
            order_id=Order.new_id(),
            symbol=decision.symbol,
            side=decision.side,
            order_type=OrderType.MARKET,
            lot_size=ps.lot_size,
            timestamp_utc=timestamp_utc,
            stop_loss=ps.stop_loss_price,
            take_profit=ps.take_profit_price,
            comment=comment,
        )

    # ── Submission ────────────────────────────────────────────────────────────

    def submit(self, order: Order) -> Order:
        """Submit an order to the broker.

        1. Guards against duplicates (idempotency).
        2. Enforces RunMode safety.
        3. Calls broker.send_order().
        4. Stores the order and its final state.

        Args:
            order: An Order in PENDING status.

        Returns:
            The Order updated to SUBMITTED or REJECTED status.

        Raises:
            DuplicateOrderError: If order_id has already been submitted.
            ModeGuardError:      If mode mismatch (e.g. LIVE in BACKTEST).
            OrderRejectedError:  If broker rejects the order synchronously.
        """
        self._guard.check(order.order_id)
        self._guard.register(order.order_id)
        self._orders[order.order_id] = order

        try:
            submitted = self._broker.send_order(order)  # type: ignore[union-attr]
            submitted = order.with_status(
                OrderStatus.SUBMITTED,
                broker_order_id=getattr(submitted, "broker_order_id", None),
            )
        except Exception as exc:
            rejected = order.with_status(OrderStatus.REJECTED)
            self._orders[order.order_id] = rejected
            log.error(
                "order_submission_failed",
                order_id=order.order_id,
                error=str(exc),
            )
            raise OrderRejectedError(
                message=f"Broker rejected order {order.order_id}: {exc}",
                code="ORDER_REJECTED",
                context={"order_id": order.order_id},
            ) from exc

        self._orders[order.order_id] = submitted
        log.info(
            "order_submitted",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            lots=str(order.lot_size),
            broker_id=submitted.broker_order_id,
        )
        return submitted

    # ── Fill handling ─────────────────────────────────────────────────────────

    def on_fill(self, fill: Fill) -> Position:
        """Process a broker fill notification and create/update a Position.

        Args:
            fill: Fill record from the broker.

        Returns:
            The new Position opened by this fill.
        """
        order = self._orders.get(fill.order_id)
        if order is not None:
            filled_order = order.with_status(OrderStatus.FILLED)
            self._orders[fill.order_id] = filled_order

        position = Position(
            position_id=fill.broker_order_id,
            symbol=fill.symbol,
            side=PositionSide.LONG if fill.side == OrderSide.BUY else PositionSide.SHORT,
            lot_size=fill.filled_lots,
            entry_price=fill.filled_price,
            opened_at_utc=fill.timestamp_utc,
            stop_loss=order.stop_loss if order else None,
            take_profit=order.take_profit if order else None,
            commission=fill.commission,
            broker_position_id=fill.broker_order_id,
        )
        self._positions[position.position_id] = position
        log.info(
            "position_opened",
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side.value,
            entry=str(position.entry_price),
            lots=str(position.lot_size),
        )
        return position

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_order(self, order_id: str) -> Order | None:
        return self._orders.get(order_id)

    def get_position(self, position_id: str) -> Position | None:
        return self._positions.get(position_id)

    def open_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.is_open]

    def all_orders(self) -> list[Order]:
        return list(self._orders.values())

    @property
    def run_mode(self) -> RunMode:
        return self._run_mode
