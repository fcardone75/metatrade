"""Tests for core/contracts/order.py."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from metatrade.core.contracts.order import Fill, Order
from metatrade.core.enums import OrderSide, OrderStatus, OrderType

UTC = UTC
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
T1 = datetime(2024, 1, 15, 11, 0, tzinfo=UTC)


def make_order(**kwargs) -> Order:
    defaults = dict(
        order_id=Order.new_id(),
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        lot_size=Decimal("0.10"),
        timestamp_utc=T0,
    )
    defaults.update(kwargs)
    return Order(**defaults)


class TestOrder:
    def test_valid_market_order(self) -> None:
        order = make_order()
        assert order.status == OrderStatus.PENDING
        assert order.price is None

    def test_new_id_is_unique(self) -> None:
        ids = {Order.new_id() for _ in range(100)}
        assert len(ids) == 100

    def test_lot_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="lot_size"):
            make_order(lot_size=Decimal("0"))

    def test_lot_size_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="lot_size"):
            make_order(lot_size=Decimal("-0.1"))

    def test_limit_order_requires_price(self) -> None:
        with pytest.raises(ValueError, match="price is required"):
            make_order(order_type=OrderType.LIMIT, price=None)

    def test_stop_order_requires_price(self) -> None:
        with pytest.raises(ValueError, match="price is required"):
            make_order(order_type=OrderType.STOP, price=None)

    def test_stop_limit_order_requires_price(self) -> None:
        with pytest.raises(ValueError, match="price is required"):
            make_order(order_type=OrderType.STOP_LIMIT, price=None)

    def test_limit_order_with_price(self) -> None:
        order = make_order(order_type=OrderType.LIMIT, price=Decimal("1.10000"))
        assert order.price == Decimal("1.10000")

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            make_order(timestamp_utc=datetime(2024, 1, 1))

    def test_negative_stop_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="stop_loss"):
            make_order(stop_loss=Decimal("-1.0"))

    def test_negative_take_profit_raises(self) -> None:
        with pytest.raises(ValueError, match="take_profit"):
            make_order(take_profit=Decimal("-1.0"))

    def test_with_status_creates_new_object(self) -> None:
        original = make_order()
        updated = original.with_status(OrderStatus.SUBMITTED, broker_order_id="broker-001")
        assert updated.status == OrderStatus.SUBMITTED
        assert updated.broker_order_id == "broker-001"
        assert original.status == OrderStatus.PENDING  # unchanged

    def test_with_status_preserves_all_fields(self) -> None:
        original = make_order(
            stop_loss=Decimal("1.09000"),
            take_profit=Decimal("1.11000"),
            comment="test trade",
        )
        updated = original.with_status(OrderStatus.FILLED)
        assert updated.stop_loss == original.stop_loss
        assert updated.take_profit == original.take_profit
        assert updated.comment == original.comment
        assert updated.order_id == original.order_id

    def test_frozen(self) -> None:
        order = make_order()
        with pytest.raises(Exception):
            order.lot_size = Decimal("1.0")  # type: ignore[misc]

    def test_repr_contains_key_info(self) -> None:
        order = make_order()
        r = repr(order)
        assert "EURUSD" in r
        assert "BUY" in r
        assert "PENDING" in r


class TestFill:
    def make_fill(self, **kwargs) -> Fill:
        defaults = dict(
            order_id="order-123",
            broker_order_id="broker-456",
            symbol="EURUSD",
            side=OrderSide.BUY,
            filled_price=Decimal("1.10002"),
            filled_lots=Decimal("0.10"),
            commission=Decimal("0.70"),
            timestamp_utc=T0,
        )
        defaults.update(kwargs)
        return Fill(**defaults)

    def test_valid_fill(self) -> None:
        fill = self.make_fill()
        assert fill.total_cost == Decimal("0.70")  # commission + swap(0)

    def test_total_cost_with_swap(self) -> None:
        fill = self.make_fill(commission=Decimal("0.70"), swap=Decimal("-0.50"))
        assert fill.total_cost == Decimal("0.20")

    def test_negative_filled_price_raises(self) -> None:
        with pytest.raises(ValueError, match="filled_price"):
            self.make_fill(filled_price=Decimal("-1.1"))

    def test_zero_filled_lots_raises(self) -> None:
        with pytest.raises(ValueError, match="filled_lots"):
            self.make_fill(filled_lots=Decimal("0"))

    def test_negative_commission_raises(self) -> None:
        with pytest.raises(ValueError, match="commission"):
            self.make_fill(commission=Decimal("-1"))

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            self.make_fill(timestamp_utc=datetime(2024, 1, 1))

    def test_slippage_defaults_to_zero(self) -> None:
        fill = self.make_fill()
        assert fill.slippage_pips == Decimal("0")

    def test_frozen(self) -> None:
        fill = self.make_fill()
        with pytest.raises(Exception):
            fill.filled_price = Decimal("1.20000")  # type: ignore[misc]
