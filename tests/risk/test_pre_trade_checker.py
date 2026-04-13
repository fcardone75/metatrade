"""Tests for the pre-trade checker."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.account import AccountState
from metatrade.core.enums import KillSwitchLevel, RunMode
from metatrade.risk.config import RiskConfig
from metatrade.risk.kill_switch import KillSwitchManager
from metatrade.risk.pre_trade_checker import PreTradeChecker

UTC = timezone.utc
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def make_account(**kwargs) -> AccountState:
    defaults = dict(
        balance=Decimal("10000"),
        equity=Decimal("10000"),
        free_margin=Decimal("10000"),
        used_margin=Decimal("0"),
        timestamp_utc=T0,
        open_positions_count=0,
    )
    defaults.update(kwargs)
    return AccountState(**defaults)


def make_config(**kwargs) -> RiskConfig:
    defaults = dict(
        max_open_positions=5,
        max_spread_pips=3.0,
        min_free_margin_pct=0.20,
        daily_loss_limit_pct=0.05,
    )
    defaults.update(kwargs)
    return RiskConfig(**defaults)


class TestPreTradeChecker:
    def test_no_veto_when_all_clear(self) -> None:
        checker = PreTradeChecker(make_config())
        veto = checker.check(make_account(), T0)
        assert veto is None

    def test_kill_switch_blocks_trade(self) -> None:
        ks = KillSwitchManager()
        ks.activate(KillSwitchLevel.TRADE_GATE, "test", ts=T0)
        checker = PreTradeChecker(make_config(), kill_switch=ks)
        veto = checker.check(make_account(), T0)
        assert veto is not None
        assert veto.veto_code == "KILL_SWITCH_ACTIVE"

    def test_max_positions_veto(self) -> None:
        cfg = make_config(max_open_positions=3)
        checker = PreTradeChecker(cfg)
        account = make_account(open_positions_count=3)
        veto = checker.check(account, T0)
        assert veto is not None
        assert veto.veto_code == "MAX_POSITIONS"

    def test_max_positions_unlimited_when_zero(self) -> None:
        cfg = make_config(max_open_positions=0)
        checker = PreTradeChecker(cfg)
        account = make_account(open_positions_count=100)
        veto = checker.check(account, T0)
        assert veto is None

    def test_daily_loss_limit_veto(self) -> None:
        cfg = make_config(daily_loss_limit_pct=0.05)
        checker = PreTradeChecker(cfg)
        # balance=10000, equity=9400 → drawdown=600 > limit=500
        account = make_account(balance=Decimal("10000"), equity=Decimal("9400"))
        veto = checker.check(account, T0)
        assert veto is not None
        assert veto.veto_code == "DAILY_LOSS_LIMIT"

    def test_daily_loss_under_limit_passes(self) -> None:
        cfg = make_config(daily_loss_limit_pct=0.05)
        checker = PreTradeChecker(cfg)
        # drawdown=400 < limit=500
        account = make_account(balance=Decimal("10000"), equity=Decimal("9600"))
        veto = checker.check(account, T0)
        assert veto is None

    def test_spread_too_wide_veto(self) -> None:
        cfg = make_config(max_spread_pips=3.0)
        checker = PreTradeChecker(cfg)
        veto = checker.check(make_account(), T0, spread_pips=5.0)
        assert veto is not None
        assert veto.veto_code == "SPREAD_TOO_WIDE"

    def test_spread_within_limit_passes(self) -> None:
        cfg = make_config(max_spread_pips=3.0)
        checker = PreTradeChecker(cfg)
        veto = checker.check(make_account(), T0, spread_pips=2.5)
        assert veto is None

    def test_spread_check_skipped_when_none(self) -> None:
        cfg = make_config(max_spread_pips=3.0)
        checker = PreTradeChecker(cfg)
        veto = checker.check(make_account(), T0, spread_pips=None)
        assert veto is None

    def test_spread_check_disabled_when_zero(self) -> None:
        cfg = make_config(max_spread_pips=0.0)
        checker = PreTradeChecker(cfg)
        veto = checker.check(make_account(), T0, spread_pips=100.0)
        assert veto is None

    def test_insufficient_margin_veto(self) -> None:
        cfg = make_config(min_free_margin_pct=0.20)
        checker = PreTradeChecker(cfg)
        # balance=10000, required free_margin=2000, but only 1000 available
        account = make_account(
            balance=Decimal("10000"),
            free_margin=Decimal("1000"),
        )
        veto = checker.check(account, T0)
        assert veto is not None
        assert veto.veto_code == "INSUFFICIENT_MARGIN"

    def test_kill_switch_takes_priority_over_other_checks(self) -> None:
        ks = KillSwitchManager()
        ks.activate(KillSwitchLevel.EMERGENCY_HALT, "emergency", ts=T0)
        cfg = make_config(max_open_positions=0)  # no position limit
        checker = PreTradeChecker(cfg, kill_switch=ks)
        veto = checker.check(make_account(), T0)
        assert veto.veto_code == "KILL_SWITCH_ACTIVE"
