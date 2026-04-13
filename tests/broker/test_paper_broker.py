"""Tests for PaperBrokerAdapter."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.order import Order
from metatrade.core.enums import OrderSide, OrderStatus, OrderType, RunMode
from metatrade.core.errors import BrokerConnectionError, OrderRejectedError
from metatrade.broker.paper_broker import PaperBrokerAdapter

UTC = timezone.utc
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def make_order(**kwargs) -> Order:
    defaults = dict(
        order_id=Order.new_id(),
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        lot_size=Decimal("0.50"),
        timestamp_utc=T0,
    )
    defaults.update(kwargs)
    return Order(**defaults)


class TestPaperBrokerConnect:
    def test_not_connected_initially(self) -> None:
        broker = PaperBrokerAdapter()
        assert not broker.is_connected()

    def test_connect_marks_connected(self) -> None:
        broker = PaperBrokerAdapter()
        broker.connect()
        assert broker.is_connected()

    def test_disconnect_marks_disconnected(self) -> None:
        broker = PaperBrokerAdapter()
        broker.connect()
        broker.disconnect()
        assert not broker.is_connected()

    def test_send_order_requires_connection(self) -> None:
        broker = PaperBrokerAdapter()
        with pytest.raises(BrokerConnectionError):
            broker.send_order(make_order())

    def test_get_account_requires_connection(self) -> None:
        broker = PaperBrokerAdapter()
        with pytest.raises(BrokerConnectionError):
            broker.get_account()


class TestPaperBrokerOrders:
    def _connected(self, **kwargs) -> PaperBrokerAdapter:
        b = PaperBrokerAdapter(**kwargs)
        b.connect()
        return b

    def test_send_order_returns_submitted(self) -> None:
        broker = self._connected()
        order = make_order()
        result = broker.send_order(order)
        assert result.status == OrderStatus.SUBMITTED

    def test_send_order_assigns_broker_id(self) -> None:
        broker = self._connected()
        result = broker.send_order(make_order())
        assert result.broker_order_id is not None
        assert result.broker_order_id.startswith("PAPER-")

    def test_cancel_pending_order(self) -> None:
        broker = self._connected()
        order = make_order()
        broker.send_order(order)
        cancelled = broker.cancel_order(order.order_id)
        assert cancelled

    def test_cancel_unknown_order_returns_false(self) -> None:
        broker = self._connected()
        assert not broker.cancel_order("unknown-id")

    def test_fill_pending_at_ask_for_buy(self) -> None:
        broker = self._connected()
        broker.set_quote("EURUSD", bid=Decimal("1.09990"), ask=Decimal("1.10010"))
        order = make_order(side=OrderSide.BUY)
        result = broker.send_order(order)
        fill = broker.fill_pending(order.order_id, T0)
        assert fill.filled_price == Decimal("1.10010")

    def test_fill_pending_at_bid_for_sell(self) -> None:
        broker = self._connected()
        broker.set_quote("EURUSD", bid=Decimal("1.09990"), ask=Decimal("1.10010"))
        order = make_order(side=OrderSide.SELL)
        broker.send_order(order)
        fill = broker.fill_pending(order.order_id, T0)
        assert fill.filled_price == Decimal("1.09990")

    def test_fill_non_pending_raises(self) -> None:
        broker = self._connected()
        with pytest.raises(OrderRejectedError):
            broker.fill_pending("nonexistent-id", T0)

    def test_fill_sets_correct_lots(self) -> None:
        broker = self._connected()
        broker.set_quote("EURUSD", bid=Decimal("1.09990"), ask=Decimal("1.10010"))
        order = make_order(lot_size=Decimal("1.20"))
        broker.send_order(order)
        fill = broker.fill_pending(order.order_id, T0)
        assert fill.filled_lots == Decimal("1.20")

    def test_commission_per_lot_applied(self) -> None:
        broker = self._connected(commission_per_lot=Decimal("5.00"))
        broker.set_quote("EURUSD", bid=Decimal("1.09990"), ask=Decimal("1.10010"))
        order = make_order(lot_size=Decimal("2.0"))
        broker.send_order(order)
        fill = broker.fill_pending(order.order_id, T0)
        assert fill.commission == Decimal("10.00")


class TestPaperBrokerAccount:
    def test_initial_balance(self) -> None:
        broker = PaperBrokerAdapter(initial_balance=Decimal("50000"))
        broker.connect()
        account = broker.get_account()
        assert account.balance == Decimal("50000")

    def test_run_mode_is_paper(self) -> None:
        broker = PaperBrokerAdapter()
        broker.connect()
        account = broker.get_account()
        assert account.run_mode == RunMode.PAPER

    def test_currency_preserved(self) -> None:
        broker = PaperBrokerAdapter(currency="EUR")
        broker.connect()
        assert broker.get_account().currency == "EUR"

    def test_credit_increases_balance(self) -> None:
        broker = PaperBrokerAdapter(initial_balance=Decimal("10000"))
        broker.connect()
        broker.credit(Decimal("500"))
        assert broker.get_account().balance == Decimal("10500")

    def test_debit_decreases_balance(self) -> None:
        broker = PaperBrokerAdapter(initial_balance=Decimal("10000"))
        broker.connect()
        broker.debit(Decimal("200"))
        assert broker.get_account().balance == Decimal("9800")


class TestPaperBrokerPrices:
    def test_get_price_returns_bid_ask(self) -> None:
        broker = PaperBrokerAdapter()
        broker.connect()
        broker.set_quote("EURUSD", Decimal("1.09990"), Decimal("1.10010"))
        bid, ask = broker.get_current_price("EURUSD")
        assert bid == Decimal("1.09990")
        assert ask == Decimal("1.10010")

    def test_get_price_unknown_symbol_raises(self) -> None:
        broker = PaperBrokerAdapter()
        broker.connect()
        with pytest.raises(KeyError):
            broker.get_current_price("UNKNOWN")
