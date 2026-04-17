"""Paper trading broker adapter.

Simulates order execution without sending any real orders to a broker.
Used in PAPER run mode: prices are real (from market data feed) but
all trades are simulated.

Fill simulation:
    - MARKET orders fill immediately at mid-price (bid+ask)/2 + spread/2 for BUY,
      or mid-price - spread/2 for SELL.
    - Commission defaults to 0 (configurable).
    - Slippage defaults to 0.

Account state is maintained in-memory and updated after each fill.
This is not persistence-backed — a restart resets state.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from metatrade.broker.interface import IBrokerAdapter
from metatrade.core.contracts.account import AccountState
from metatrade.core.contracts.order import Fill, Order
from metatrade.core.enums import OrderSide, OrderStatus, RunMode
from metatrade.core.errors import BrokerConnectionError, OrderRejectedError
from metatrade.core.log import get_logger

log = get_logger(__name__)


class PaperBrokerAdapter(IBrokerAdapter):
    """Simulated broker for PAPER trading mode.

    Args:
        initial_balance: Starting account balance in account currency.
        commission_per_lot: Commission charged per lot (default 0).
        spread_pips: Fixed spread to simulate (default 0.0002 = 2 pips).
        currency: Account currency (default "USD").
    """

    def __init__(
        self,
        initial_balance: Decimal = Decimal("10000"),
        commission_per_lot: Decimal = Decimal("0"),
        spread: Decimal = Decimal("0.00020"),
        currency: str = "USD",
    ) -> None:
        self._balance = initial_balance
        self._commission_per_lot = commission_per_lot
        self._spread = spread
        self._currency = currency
        self._connected = False
        # {symbol: (bid, ask)} price quotes
        self._quotes: dict[str, tuple[Decimal, Decimal]] = {}
        # Pending orders (not yet filled)
        self._pending: dict[str, Order] = {}
        # Open positions value tracking (simplified: equity = balance + unrealized)
        self._unrealized_pnl: Decimal = Decimal("0")
        self._used_margin: Decimal = Decimal("0")

    # ── IBrokerAdapter ────────────────────────────────────────────────────────

    def connect(self) -> None:
        self._connected = True
        log.info("paper_broker_connected")

    def disconnect(self) -> None:
        self._connected = False
        log.info("paper_broker_disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def send_order(self, order: Order) -> Order:
        if not self._connected:
            raise BrokerConnectionError(
                message="PaperBroker: not connected",
                code="PAPER_NOT_CONNECTED",
            )
        broker_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        self._pending[order.order_id] = order
        submitted = order.with_status(OrderStatus.SUBMITTED, broker_order_id=broker_id)
        log.info(
            "paper_order_submitted",
            order_id=order.order_id,
            broker_id=broker_id,
            symbol=order.symbol,
            side=order.side.value,
            lots=str(order.lot_size),
        )
        return submitted

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._pending:
            del self._pending[order_id]
            return True
        return False

    def get_account(self) -> AccountState:
        if not self._connected:
            raise BrokerConnectionError(
                message="PaperBroker: not connected",
                code="PAPER_NOT_CONNECTED",
            )
        return AccountState(
            balance=self._balance,
            equity=self._balance + self._unrealized_pnl,
            free_margin=self._balance - self._used_margin,
            used_margin=self._used_margin,
            timestamp_utc=datetime.now(UTC),
            currency=self._currency,
            run_mode=RunMode.PAPER,
        )

    def get_current_price(self, symbol: str) -> tuple[Decimal, Decimal]:
        if symbol not in self._quotes:
            raise KeyError(f"No quote available for {symbol}")
        return self._quotes[symbol]

    # ── Paper-specific ────────────────────────────────────────────────────────

    def set_quote(self, symbol: str, bid: Decimal, ask: Decimal) -> None:
        """Update the simulated price for a symbol."""
        self._quotes[symbol] = (bid, ask)

    def fill_pending(self, order_id: str, timestamp_utc: datetime | None = None) -> Fill:
        """Immediately fill a pending order at the current simulated price.

        Args:
            order_id:      ID of the pending order to fill.
            timestamp_utc: Fill timestamp (defaults to now).

        Returns:
            The Fill record.

        Raises:
            OrderRejectedError: If the order is not pending or no price available.
        """
        order = self._pending.pop(order_id, None)
        if order is None:
            raise OrderRejectedError(
                message=f"Order {order_id} is not pending",
                code="ORDER_NOT_FOUND",
            )
        bid, ask = self.get_current_price(order.symbol)
        fill_price = ask if order.side == OrderSide.BUY else bid
        commission = self._commission_per_lot * order.lot_size
        ts = timestamp_utc or datetime.now(UTC)

        fill = Fill(
            order_id=order.order_id,
            broker_order_id=f"PAPER-{uuid.uuid4().hex[:8].upper()}",
            symbol=order.symbol,
            side=order.side,
            filled_price=fill_price,
            filled_lots=order.lot_size,
            commission=commission,
            timestamp_utc=ts,
        )
        log.info(
            "paper_order_filled",
            order_id=order_id,
            symbol=order.symbol,
            price=str(fill_price),
            lots=str(order.lot_size),
        )
        return fill

    def credit(self, amount: Decimal) -> None:
        """Credit the paper account balance (for PnL updates)."""
        self._balance += amount

    def debit(self, amount: Decimal) -> None:
        """Debit the paper account balance."""
        self._balance -= amount
