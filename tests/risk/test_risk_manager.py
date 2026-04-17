"""Integration tests for the RiskManager."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from metatrade.core.contracts.account import AccountState
from metatrade.core.enums import KillSwitchLevel, OrderSide
from metatrade.risk.config import RiskConfig
from metatrade.risk.manager import RiskManager

UTC = UTC
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


def make_config(**kwargs) -> RiskConfig:
    defaults = dict(
        max_risk_pct=0.01,
        pip_value_per_lot=Decimal("10.0"),
        max_open_positions=5,
        max_spread_pips=3.0,
        min_free_margin_pct=0.10,
        daily_loss_limit_pct=0.05,
        auto_kill_drawdown_pct=0.10,
    )
    defaults.update(kwargs)
    return RiskConfig(**defaults)


class TestRiskManager:
    def test_approved_trade(self) -> None:
        manager = RiskManager(make_config())
        decision = manager.evaluate(
            symbol="EURUSD",
            side=OrderSide.BUY,
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            account=make_account(),
            timestamp_utc=T0,
        )
        assert decision.approved
        assert decision.position_size is not None
        assert decision.veto is None

    def test_vetoed_by_spread(self) -> None:
        manager = RiskManager(make_config(max_spread_pips=2.0))
        decision = manager.evaluate(
            symbol="EURUSD",
            side=OrderSide.BUY,
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            account=make_account(),
            timestamp_utc=T0,
            spread_pips=5.0,
        )
        assert not decision.approved
        assert decision.veto is not None
        assert decision.veto.veto_code == "SPREAD_TOO_WIDE"

    def test_vetoed_by_kill_switch(self) -> None:
        manager = RiskManager(make_config())
        manager.kill_switch.activate(KillSwitchLevel.TRADE_GATE, "manual stop", ts=T0)
        decision = manager.evaluate(
            symbol="EURUSD",
            side=OrderSide.BUY,
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            account=make_account(),
            timestamp_utc=T0,
        )
        assert not decision.approved
        assert decision.veto.veto_code == "KILL_SWITCH_ACTIVE"

    def test_auto_kill_triggers_on_drawdown(self) -> None:
        # Disable daily_loss_limit so only auto-kill fires for the drawdown
        cfg = make_config(auto_kill_drawdown_pct=0.10, daily_loss_limit_pct=0.99)
        manager = RiskManager(cfg)
        # drawdown = 10000 - 8900 = 1100 ≥ 10% of 10000 = 1000
        account = make_account(
            balance=Decimal("10000"),
            equity=Decimal("8900"),
            free_margin=Decimal("8900"),
        )
        # First call — this triggers auto-kill but still processes the trade
        # because check() runs before auto-kill
        decision = manager.evaluate(
            symbol="EURUSD",
            side=OrderSide.BUY,
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            account=account,
            timestamp_utc=T0,
        )
        # After evaluation, kill switch should be SESSION_GATE
        assert manager.kill_switch.level == KillSwitchLevel.SESSION_GATE
        # The current evaluation was already approved before auto-kill fired
        # (auto-kill fires after pre-trade checks pass)
        assert decision.approved

    def test_auto_kill_blocks_next_trade_after_trigger(self) -> None:
        cfg = make_config(auto_kill_drawdown_pct=0.10, daily_loss_limit_pct=0.99)
        manager = RiskManager(cfg)
        # Large drawdown (11%) — above auto_kill threshold but below daily_loss_limit
        account = make_account(
            balance=Decimal("10000"),
            equity=Decimal("8900"),
            free_margin=Decimal("8900"),
        )
        # First trade triggers auto-kill
        manager.evaluate(
            symbol="EURUSD",
            side=OrderSide.BUY,
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            account=account,
            timestamp_utc=T0,
        )
        # Second trade should be blocked
        decision2 = manager.evaluate(
            symbol="EURUSD",
            side=OrderSide.BUY,
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            account=account,
            timestamp_utc=T0,
        )
        assert not decision2.approved
        assert decision2.veto.veto_code == "KILL_SWITCH_ACTIVE"

    def test_sizing_error_returns_veto(self) -> None:
        manager = RiskManager(make_config())
        # Invalid: BUY but SL above entry
        decision = manager.evaluate(
            symbol="EURUSD",
            side=OrderSide.BUY,
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.10500"),  # above entry — invalid for BUY
            account=make_account(),
            timestamp_utc=T0,
        )
        assert not decision.approved
        assert decision.veto.veto_code == "SIZING_ERROR"

    def test_lot_size_in_result(self) -> None:
        manager = RiskManager(make_config())
        decision = manager.evaluate(
            symbol="EURUSD",
            side=OrderSide.BUY,
            entry_price=Decimal("1.10000"),
            sl_price=Decimal("1.09800"),
            account=make_account(),
            timestamp_utc=T0,
        )
        assert decision.position_size.lot_size == Decimal("0.50")

    def test_symbol_preserved_in_decision(self) -> None:
        manager = RiskManager(make_config())
        decision = manager.evaluate(
            symbol="GBPUSD",
            side=OrderSide.SELL,
            entry_price=Decimal("1.27000"),
            sl_price=Decimal("1.27200"),
            account=make_account(),
            timestamp_utc=T0,
        )
        assert decision.symbol == "GBPUSD"

    def test_risk_config_accessible(self) -> None:
        cfg = make_config()
        manager = RiskManager(cfg)
        assert manager._cfg is cfg
