"""Tests for OrderManager."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.order import Fill, Order
from metatrade.core.contracts.position import Position
from metatrade.core.contracts.risk import RiskDecision, PositionSizeResult
from metatrade.core.enums import OrderSide, OrderStatus, OrderType, PositionSide, RunMode
from metatrade.core.errors import DuplicateOrderError, OrderRejectedError
from metatrade.execution.order_manager import OrderManager

UTC = timezone.utc
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def make_size_result(**kwargs) -> PositionSizeResult:
    defaults = dict(
        lot_size=Decimal("0.50"),
        risk_amount=Decimal("100.00"),
        risk_pct=0.01,
        stop_loss_price=Decimal("1.09800"),
        stop_loss_pips=Decimal("20.0"),
        pip_value=Decimal("10.0"),
        method="fixed_fractional",
        take_profit_price=Decimal("1.10400"),
    )
    defaults.update(kwargs)
    return PositionSizeResult(**defaults)


def make_decision(**kwargs) -> RiskDecision:
    defaults = dict(
        approved=True,
        symbol="EURUSD",
        side=OrderSide.BUY,
        timestamp_utc=T0,
        position_size=make_size_result(),
    )
    defaults.update(kwargs)
    return RiskDecision(**defaults)


class FakeBroker:
    """Minimal broker that always succeeds."""

    def __init__(self, broker_order_id: str = "B001") -> None:
        self._broker_id = broker_order_id
        self.submitted: list[Order] = []

    def send_order(self, order: Order) -> Order:
        self.submitted.append(order)
        return order.with_status(OrderStatus.SUBMITTED, broker_order_id=self._broker_id)


class FailingBroker:
    """Broker that always rejects."""

    def send_order(self, order: Order) -> Order:
        raise RuntimeError("Broker refused: insufficient margin")


class TestOrderManagerBuild:
    def test_build_order_from_decision(self) -> None:
        manager = OrderManager(broker=FakeBroker())
        order = manager.build_order(make_decision(), T0)
        assert order.symbol == "EURUSD"
        assert order.side == OrderSide.BUY
        assert order.lot_size == Decimal("0.50")
        assert order.order_type == OrderType.MARKET

    def test_build_order_sets_sl_tp(self) -> None:
        manager = OrderManager(broker=FakeBroker())
        order = manager.build_order(make_decision(), T0)
        assert order.stop_loss == Decimal("1.09800")
        assert order.take_profit == Decimal("1.10400")

    def test_build_order_raises_on_non_approved_decision(self) -> None:
        from metatrade.core.contracts.risk import RiskVeto
        veto = RiskVeto(reason="spread too wide", veto_code="SPREAD_TOO_WIDE", timestamp_utc=T0)
        rejected = RiskDecision(
            approved=False,
            symbol="EURUSD",
            side=OrderSide.BUY,
            timestamp_utc=T0,
            veto=veto,
        )
        manager = OrderManager(broker=FakeBroker())
        with pytest.raises(ValueError):
            manager.build_order(rejected, T0)

    def test_order_id_is_unique(self) -> None:
        manager = OrderManager(broker=FakeBroker())
        o1 = manager.build_order(make_decision(), T0)
        o2 = manager.build_order(make_decision(), T0)
        assert o1.order_id != o2.order_id


class TestOrderManagerSubmit:
    def test_submit_returns_submitted_order(self) -> None:
        broker = FakeBroker(broker_order_id="BRK-001")
        manager = OrderManager(broker=broker)
        order = manager.build_order(make_decision(), T0)
        result = manager.submit(order)
        assert result.status == OrderStatus.SUBMITTED
        assert result.broker_order_id == "BRK-001"

    def test_submit_stores_order(self) -> None:
        manager = OrderManager(broker=FakeBroker())
        order = manager.build_order(make_decision(), T0)
        manager.submit(order)
        assert manager.get_order(order.order_id) is not None

    def test_duplicate_submission_raises(self) -> None:
        manager = OrderManager(broker=FakeBroker())
        order = manager.build_order(make_decision(), T0)
        manager.submit(order)
        with pytest.raises(DuplicateOrderError):
            manager.submit(order)

    def test_broker_failure_raises_order_rejected(self) -> None:
        manager = OrderManager(broker=FailingBroker())
        order = manager.build_order(make_decision(), T0)
        with pytest.raises(OrderRejectedError):
            manager.submit(order)

    def test_rejected_order_stored_with_rejected_status(self) -> None:
        manager = OrderManager(broker=FailingBroker())
        order = manager.build_order(make_decision(), T0)
        with pytest.raises(OrderRejectedError):
            manager.submit(order)
        stored = manager.get_order(order.order_id)
        assert stored is not None
        assert stored.status == OrderStatus.REJECTED

    def test_broker_receives_correct_order(self) -> None:
        broker = FakeBroker()
        manager = OrderManager(broker=broker)
        order = manager.build_order(make_decision(), T0)
        manager.submit(order)
        assert len(broker.submitted) == 1
        assert broker.submitted[0].order_id == order.order_id


class TestOrderManagerFill:
    def _make_fill(self, order_id: str) -> Fill:
        return Fill(
            order_id=order_id,
            broker_order_id="BRK-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            filled_price=Decimal("1.10010"),
            filled_lots=Decimal("0.50"),
            commission=Decimal("2.50"),
            timestamp_utc=T0,
        )

    def test_fill_creates_position(self) -> None:
        broker = FakeBroker(broker_order_id="BRK-001")
        manager = OrderManager(broker=broker)
        order = manager.build_order(make_decision(), T0)
        manager.submit(order)
        position = manager.on_fill(self._make_fill(order.order_id))
        assert isinstance(position, Position)
        assert position.symbol == "EURUSD"
        assert position.side == PositionSide.LONG
        assert position.entry_price == Decimal("1.10010")

    def test_fill_marks_order_filled(self) -> None:
        broker = FakeBroker(broker_order_id="BRK-001")
        manager = OrderManager(broker=broker)
        order = manager.build_order(make_decision(), T0)
        manager.submit(order)
        manager.on_fill(self._make_fill(order.order_id))
        stored = manager.get_order(order.order_id)
        assert stored.status == OrderStatus.FILLED

    def test_position_tracked_in_open_positions(self) -> None:
        broker = FakeBroker(broker_order_id="BRK-001")
        manager = OrderManager(broker=broker)
        order = manager.build_order(make_decision(), T0)
        manager.submit(order)
        manager.on_fill(self._make_fill(order.order_id))
        assert len(manager.open_positions()) == 1

    def test_sell_fill_creates_short_position(self) -> None:
        broker = FakeBroker(broker_order_id="BRK-002")
        manager = OrderManager(broker=broker)
        decision = make_decision(
            side=OrderSide.SELL,
            position_size=make_size_result(
                stop_loss_price=Decimal("1.10200"),
                take_profit_price=Decimal("1.09600"),
            ),
        )
        order = manager.build_order(decision, T0)
        manager.submit(order)
        fill = Fill(
            order_id=order.order_id,
            broker_order_id="BRK-002",
            symbol="EURUSD",
            side=OrderSide.SELL,
            filled_price=Decimal("1.09990"),
            filled_lots=Decimal("0.50"),
            commission=Decimal("2.50"),
            timestamp_utc=T0,
        )
        position = manager.on_fill(fill)
        assert position.side == PositionSide.SHORT


class TestOrderManagerQuery:
    def test_all_orders_returns_all(self) -> None:
        broker = FakeBroker()
        manager = OrderManager(broker=broker)
        o1 = manager.build_order(make_decision(), T0)
        o2 = manager.build_order(make_decision(), T0)
        manager.submit(o1)
        manager.submit(o2)
        assert len(manager.all_orders()) == 2

    def test_get_order_returns_none_for_unknown(self) -> None:
        manager = OrderManager(broker=FakeBroker())
        assert manager.get_order("nonexistent") is None

    def test_run_mode_accessible(self) -> None:
        manager = OrderManager(broker=FakeBroker(), run_mode=RunMode.PAPER)
        assert manager.run_mode == RunMode.PAPER
