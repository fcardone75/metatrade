"""Tests for core/contracts/position.py."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from metatrade.core.contracts.position import Position
from metatrade.core.enums import PositionSide, PositionState

UTC = UTC
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
T1 = datetime(2024, 1, 15, 11, 0, tzinfo=UTC)


def make_position(**kwargs) -> Position:
    defaults = dict(
        position_id="pos-001",
        symbol="EURUSD",
        side=PositionSide.LONG,
        lot_size=Decimal("0.10"),
        entry_price=Decimal("1.10000"),
        opened_at_utc=T0,
    )
    defaults.update(kwargs)
    return Position(**defaults)


class TestPosition:
    def test_valid_long_position(self) -> None:
        pos = make_position()
        assert pos.is_open
        assert not pos.is_closed

    def test_valid_short_position(self) -> None:
        pos = make_position(side=PositionSide.SHORT)
        assert pos.is_open

    def test_unrealized_pnl_long_profit(self) -> None:
        pos = make_position(
            side=PositionSide.LONG,
            entry_price=Decimal("1.10000"),
            current_price=Decimal("1.10500"),
        )
        assert pos.unrealized_pnl_pips == Decimal("0.00500")

    def test_unrealized_pnl_long_loss(self) -> None:
        pos = make_position(
            side=PositionSide.LONG,
            entry_price=Decimal("1.10000"),
            current_price=Decimal("1.09500"),
        )
        assert pos.unrealized_pnl_pips == Decimal("-0.00500")

    def test_unrealized_pnl_short_profit(self) -> None:
        pos = make_position(
            side=PositionSide.SHORT,
            entry_price=Decimal("1.10000"),
            current_price=Decimal("1.09500"),
        )
        assert pos.unrealized_pnl_pips == Decimal("0.00500")

    def test_unrealized_pnl_none_when_no_current_price(self) -> None:
        pos = make_position()
        assert pos.unrealized_pnl_pips is None

    def test_realized_pnl_none_when_open(self) -> None:
        pos = make_position()
        assert pos.realized_pnl_pips is None

    def test_close_position(self) -> None:
        pos = make_position()
        closed = pos.close(close_price=Decimal("1.10500"), closed_at_utc=T1)
        assert closed.is_closed
        assert closed.close_price == Decimal("1.10500")
        assert closed.closed_at_utc == T1
        assert closed.realized_pnl_pips == Decimal("0.00500")

    def test_close_short_position(self) -> None:
        pos = make_position(
            side=PositionSide.SHORT,
            entry_price=Decimal("1.10000"),
        )
        closed = pos.close(close_price=Decimal("1.09500"), closed_at_utc=T1)
        assert closed.realized_pnl_pips == Decimal("0.00500")

    def test_with_current_price(self) -> None:
        pos = make_position()
        updated = pos.with_current_price(Decimal("1.10300"))
        assert updated.current_price == Decimal("1.10300")
        assert pos.current_price is None  # original unchanged

    def test_closed_state_requires_closed_at(self) -> None:
        with pytest.raises(ValueError, match="closed_at_utc"):
            make_position(
                state=PositionState.CLOSED,
                close_price=Decimal("1.10500"),
                closed_at_utc=None,
            )

    def test_closed_state_requires_close_price(self) -> None:
        with pytest.raises(ValueError, match="close_price"):
            make_position(
                state=PositionState.CLOSED,
                closed_at_utc=T1,
                close_price=None,
            )

    def test_zero_lot_size_raises(self) -> None:
        with pytest.raises(ValueError, match="lot_size"):
            make_position(lot_size=Decimal("0"))

    def test_negative_entry_price_raises(self) -> None:
        with pytest.raises(ValueError, match="entry_price"):
            make_position(entry_price=Decimal("-1.1"))

    def test_naive_opened_at_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            make_position(opened_at_utc=datetime(2024, 1, 1))

    def test_naive_closed_at_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            make_position(
                state=PositionState.CLOSED,
                closed_at_utc=datetime(2024, 1, 2),  # naive
                close_price=Decimal("1.105"),
            )

    def test_negative_commission_raises(self) -> None:
        with pytest.raises(ValueError, match="commission"):
            make_position(commission=Decimal("-1"))

    def test_frozen(self) -> None:
        pos = make_position()
        with pytest.raises(Exception):
            pos.lot_size = Decimal("1.0")  # type: ignore[misc]

    def test_repr_contains_key_info(self) -> None:
        pos = make_position()
        r = repr(pos)
        assert "EURUSD" in r
        assert "LONG" in r
        assert "OPEN" in r
