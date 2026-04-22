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
from metatrade.core.contracts.position import Position
from metatrade.core.enums import OrderSide, OrderStatus, PositionSide, RunMode
from metatrade.core.errors import BrokerConnectionError, OrderRejectedError
from metatrade.core.log import get_logger

log = get_logger(__name__)

_PIP_SIZE = Decimal("0.0001")


class PaperBrokerAdapter(IBrokerAdapter):
    """Simulated broker for PAPER trading mode.

    Args:
        initial_balance:    Starting account balance in account currency.
        commission_per_lot: Commission charged per lot (default 0).
        spread:             Fixed spread to simulate (default 0.0002 = 2 pips).
        currency:           Account currency (default "USD").
        pip_value_per_lot:  USD value of one pip per standard lot (default 10).
    """

    def __init__(
        self,
        initial_balance: Decimal = Decimal("10000"),
        commission_per_lot: Decimal = Decimal("0"),
        spread: Decimal = Decimal("0.00020"),
        currency: str = "USD",
        pip_value_per_lot: Decimal = Decimal("10"),
    ) -> None:
        self._balance = initial_balance
        self._commission_per_lot = commission_per_lot
        self._spread = spread
        self._currency = currency
        self._pip_value = pip_value_per_lot
        self._connected = False
        # {symbol: (bid, ask)} price quotes
        self._quotes: dict[str, tuple[Decimal, Decimal]] = {}
        # Pending orders (not yet filled)
        self._pending: dict[str, Order] = {}
        # Open positions tracked by position_id
        self._open_positions: dict[str, Position] = {}
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
        unrealized = self._compute_unrealized_pnl()
        return AccountState(
            balance=self._balance,
            equity=self._balance + unrealized,
            free_margin=self._balance - self._used_margin,
            used_margin=self._used_margin,
            timestamp_utc=datetime.now(UTC),
            currency=self._currency,
            open_positions_count=len(self._open_positions),
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

        Also creates an open Position that will be monitored for SL/TP.

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

        broker_oid = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        fill = Fill(
            order_id=order.order_id,
            broker_order_id=broker_oid,
            symbol=order.symbol,
            side=order.side,
            filled_price=fill_price,
            filled_lots=order.lot_size,
            commission=commission,
            timestamp_utc=ts,
        )

        pos_side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
        position = Position(
            position_id=broker_oid,
            symbol=order.symbol,
            side=pos_side,
            lot_size=order.lot_size,
            entry_price=fill_price,
            opened_at_utc=ts,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            commission=commission,
            broker_position_id=broker_oid,
        )
        self._open_positions[broker_oid] = position

        # Deduct commission from balance immediately
        self._balance -= commission

        log.info(
            "paper_order_filled",
            order_id=order_id,
            symbol=order.symbol,
            price=str(fill_price),
            lots=str(order.lot_size),
            sl=str(order.stop_loss),
            tp=str(order.take_profit),
        )
        return fill

    def check_sl_tp(
        self,
        bar_high: Decimal,
        bar_low: Decimal,
        bar_close: Decimal,
        now_utc: datetime,
    ) -> list[tuple[Position, str, Decimal]]:
        """Check all open positions for SL/TP hits on this bar.

        Evaluates whether the bar's high/low reached any position's SL or TP.
        Closes hit positions, updates balance, and returns the results.

        Returns:
            List of (closed_position, exit_reason, pnl_currency) tuples.
            exit_reason is "sl" or "tp".
        """
        closed: list[tuple[Position, str, Decimal]] = []

        for pos_id, pos in list(self._open_positions.items()):
            exit_price: Decimal | None = None
            exit_reason: str | None = None

            if pos.side == PositionSide.LONG:
                if pos.stop_loss is not None and bar_low <= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "sl"
                elif pos.take_profit is not None and bar_high >= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "tp"
            else:  # SHORT
                if pos.stop_loss is not None and bar_high >= pos.stop_loss:
                    exit_price = pos.stop_loss
                    exit_reason = "sl"
                elif pos.take_profit is not None and bar_low <= pos.take_profit:
                    exit_price = pos.take_profit
                    exit_reason = "tp"

            if exit_price is not None and exit_reason is not None:
                closed_pos = pos.close(exit_price, now_utc)
                del self._open_positions[pos_id]
                pnl = self._compute_pnl(pos.side, pos.entry_price, exit_price, pos.lot_size)
                self._balance += pnl
                log.info(
                    "paper_position_closed",
                    position_id=pos_id,
                    symbol=pos.symbol,
                    side=pos.side.value,
                    reason=exit_reason,
                    entry=str(pos.entry_price),
                    exit=str(exit_price),
                    pnl=str(pnl),
                )
                closed.append((closed_pos, exit_reason, pnl))

        return closed

    def get_open_positions_list(self) -> list[Position]:
        """Return all currently open positions."""
        return list(self._open_positions.values())

    def credit(self, amount: Decimal) -> None:
        """Credit the paper account balance (for external PnL updates)."""
        self._balance += amount

    def debit(self, amount: Decimal) -> None:
        """Debit the paper account balance."""
        self._balance -= amount

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_pnl(
        self,
        side: PositionSide,
        entry: Decimal,
        exit_price: Decimal,
        lot_size: Decimal,
    ) -> Decimal:
        """PnL = pip_count × lot_size × pip_value_per_lot."""
        raw_diff = (exit_price - entry) if side == PositionSide.LONG else (entry - exit_price)
        pip_count = raw_diff / _PIP_SIZE
        return pip_count * lot_size * self._pip_value

    def _compute_unrealized_pnl(self) -> Decimal:
        total = Decimal("0")
        for pos in self._open_positions.values():
            quote = self._quotes.get(pos.symbol)
            if quote is None:
                continue
            bid, ask = quote
            current = bid if pos.side == PositionSide.SHORT else ask
            total += self._compute_pnl(pos.side, pos.entry_price, current, pos.lot_size)
        return total
