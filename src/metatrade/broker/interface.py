"""Broker adapter interface.

All broker implementations must satisfy this contract.
The rest of the system only ever talks to IBrokerAdapter, never directly
to MT5 or any other concrete broker library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal

from metatrade.core.contracts.account import AccountState
from metatrade.core.contracts.order import Order


class IBrokerAdapter(ABC):
    """Contract for all broker implementations.

    Implementations:
        MT5BrokerAdapter   — live/paper trading via MetaTrader 5 (Windows only).
        PaperBrokerAdapter — simulated fills at market price, no real orders.
        MockBrokerAdapter  — deterministic fills for unit tests.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the broker.

        Raises:
            BrokerConnectionError: If the connection fails.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Close the broker connection cleanly."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if currently connected."""

    @abstractmethod
    def send_order(self, order: Order) -> Order:
        """Submit an order to the broker.

        Args:
            order: An Order in PENDING status.

        Returns:
            Order updated with broker_order_id and SUBMITTED status.

        Raises:
            OrderRejectedError:     If the broker rejects the order.
            BrokerConnectionError:  If the connection is lost.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Request cancellation of a pending order.

        Returns:
            True if the cancellation was accepted, False if the order was
            already filled or already cancelled.
        """

    @abstractmethod
    def get_account(self) -> AccountState:
        """Fetch current account state from the broker.

        Returns:
            Snapshot of balance, equity, margin.

        Raises:
            BrokerConnectionError: If not connected.
        """

    @abstractmethod
    def get_current_price(self, symbol: str) -> tuple[Decimal, Decimal]:
        """Return the current (bid, ask) for a symbol.

        Raises:
            BrokerConnectionError: If not connected.
            DataError:             If no quote is available for the symbol.
        """

    # ── Optional position management (not abstract — live adapter implements) ──

    def get_open_positions(self, symbol: str | None = None) -> list[dict]:
        """Return open positions.  Default: empty list (paper/mock adapters)."""
        return []

    def modify_sl_tp(
        self,
        ticket: int,
        symbol: str,
        new_sl: float | None,
        new_tp: float | None,
    ) -> bool:
        """Modify SL/TP of an open position.  Default: no-op (returns False)."""
        return False

    def close_position_by_ticket(
        self,
        ticket: int,
        symbol: str,
        volume: float,
        is_buy: bool,
    ) -> bool:
        """Close (fully or partially) a position by ticket.  Default: no-op."""
        return False

    def get_latest_tick(self, symbol: str) -> tuple[int, float, float] | None:
        """Return (time_msc, bid, ask) for the latest tick on *symbol*.

        ``time_msc`` is milliseconds since epoch — callers compare the value
        to detect whether a new tick arrived since the last call.
        Returns None when not connected or no tick is available.
        """
        return None
