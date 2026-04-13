"""Tests for core/contracts/account.py."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.account import AccountState
from metatrade.core.enums import RunMode

UTC = timezone.utc
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def make_account(**kwargs) -> AccountState:
    defaults = dict(
        balance=Decimal("10000"),
        equity=Decimal("10000"),
        free_margin=Decimal("10000"),
        used_margin=Decimal("0"),
        timestamp_utc=T0,
    )
    defaults.update(kwargs)
    return AccountState(**defaults)


class TestAccountState:
    def test_valid_construction(self) -> None:
        acc = make_account()
        assert acc.balance == Decimal("10000")
        assert acc.currency == "USD"
        assert acc.run_mode == RunMode.PAPER

    def test_margin_level_none_when_no_positions(self) -> None:
        acc = make_account(used_margin=Decimal("0"))
        assert acc.margin_level is None

    def test_margin_level_with_positions(self) -> None:
        acc = make_account(
            equity=Decimal("10000"),
            used_margin=Decimal("5000"),
        )
        assert acc.margin_level == Decimal("200")  # 200%

    def test_unrealized_pnl_zero_when_flat(self) -> None:
        acc = make_account(balance=Decimal("10000"), equity=Decimal("10000"))
        assert acc.unrealized_pnl == Decimal("0")

    def test_unrealized_pnl_positive(self) -> None:
        acc = make_account(balance=Decimal("10000"), equity=Decimal("10200"))
        assert acc.unrealized_pnl == Decimal("200")

    def test_unrealized_pnl_negative(self) -> None:
        acc = make_account(balance=Decimal("10000"), equity=Decimal("9800"))
        assert acc.unrealized_pnl == Decimal("-200")

    def test_drawdown_from_equity_when_in_profit(self) -> None:
        acc = make_account(balance=Decimal("10000"), equity=Decimal("10200"))
        assert acc.drawdown_from_equity == Decimal("0")

    def test_drawdown_from_equity_when_in_loss(self) -> None:
        acc = make_account(balance=Decimal("10000"), equity=Decimal("9500"))
        assert acc.drawdown_from_equity == Decimal("500")

    def test_negative_balance_raises(self) -> None:
        with pytest.raises(ValueError, match="balance"):
            make_account(balance=Decimal("-1"))

    def test_negative_free_margin_raises(self) -> None:
        with pytest.raises(ValueError, match="free_margin"):
            make_account(free_margin=Decimal("-1"))

    def test_negative_used_margin_raises(self) -> None:
        with pytest.raises(ValueError, match="used_margin"):
            make_account(used_margin=Decimal("-1"))

    def test_negative_open_positions_raises(self) -> None:
        with pytest.raises(ValueError, match="open_positions_count"):
            make_account(open_positions_count=-1)

    def test_empty_currency_raises(self) -> None:
        with pytest.raises(ValueError, match="currency"):
            make_account(currency="")

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            make_account(timestamp_utc=datetime(2024, 1, 1))

    def test_live_mode(self) -> None:
        acc = make_account(run_mode=RunMode.LIVE)
        assert acc.run_mode == RunMode.LIVE

    def test_frozen(self) -> None:
        acc = make_account()
        with pytest.raises(Exception):
            acc.balance = Decimal("99999")  # type: ignore[misc]

    def test_repr(self) -> None:
        acc = make_account()
        r = repr(acc)
        assert "10000" in r
        assert "PAPER" in r
