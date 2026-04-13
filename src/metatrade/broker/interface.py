"""Broker adapter interface.

All broker implementations must satisfy this contract.
The rest of the system only ever talks to IBrokerAdapter, never directly
to MT5 or any other concrete broker library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal

from metatrade.core.contracts.account import AccountState
from metatrade.core.contracts.order import Fill, Order


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
