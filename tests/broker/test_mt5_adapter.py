"""Tests for MT5BrokerAdapter (offline — verifies behaviour without MT5)."""

from __future__ import annotations

import pytest

from metatrade.core.errors import BrokerConnectionError
from metatrade.broker.mt5_adapter import MT5BrokerAdapter


class TestMT5AdapterOffline:
    """Tests that run without MetaTrader5 installed.

    The adapter uses lazy import — operations that don't require the library
    can be tested normally on any OS.
    """

    def test_not_connected_initially(self) -> None:
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        assert not adapter.is_connected()

    def test_connect_raises_on_missing_mt5(self) -> None:
        """On macOS/Linux (no MT5), connect() raises BrokerConnectionError."""
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        try:
            import MetaTrader5  # noqa: F401
            pytest.skip("MetaTrader5 is installed — skipping offline test")
        except ImportError:
            pass
        with pytest.raises(BrokerConnectionError, match="not available"):
            adapter.connect()

    def test_send_order_raises_when_not_connected(self) -> None:
        from metatrade.core.contracts.order import Order
        from metatrade.core.enums import OrderSide, OrderStatus, OrderType
        from datetime import datetime, timezone
        from decimal import Decimal

        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        order = Order(
            order_id=Order.new_id(),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            lot_size=Decimal("0.10"),
            timestamp_utc=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        )
        with pytest.raises(BrokerConnectionError):
            adapter.send_order(order)

    def test_get_account_raises_when_not_connected(self) -> None:
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        with pytest.raises(BrokerConnectionError):
            adapter.get_account()

    def test_get_current_price_raises_when_not_connected(self) -> None:
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        with pytest.raises(BrokerConnectionError):
            adapter.get_current_price("EURUSD")

    def test_cancel_order_returns_false_when_not_connected(self) -> None:
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        # cancel_order doesn't check connection — just logs and returns False
        result = adapter.cancel_order("some-order-id")
        assert result is False
