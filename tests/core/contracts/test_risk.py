"""Tests for core/contracts/risk.py."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.risk import (
    KillSwitchState,
    PositionSizeResult,
    RiskDecision,
    RiskVeto,
)
from metatrade.core.enums import KillSwitchLevel, OrderSide

UTC = timezone.utc
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def make_position_size(**kwargs) -> PositionSizeResult:
    defaults = dict(
        lot_size=Decimal("0.10"),
        risk_amount=Decimal("100"),
        risk_pct=0.01,
        stop_loss_price=Decimal("1.09700"),
        stop_loss_pips=Decimal("30"),
        pip_value=Decimal("10"),
        method="fixed_fractional",
    )
    defaults.update(kwargs)
    return PositionSizeResult(**defaults)


def make_veto(**kwargs) -> RiskVeto:
    defaults = dict(
        reason="Max positions reached",
        veto_code="MAX_POSITIONS",
        timestamp_utc=T0,
    )
    defaults.update(kwargs)
    return RiskVeto(**defaults)


class TestPositionSizeResult:
    def test_valid(self) -> None:
        ps = make_position_size()
        assert ps.lot_size == Decimal("0.10")

    def test_zero_lot_size_raises(self) -> None:
        with pytest.raises(ValueError, match="lot_size"):
            make_position_size(lot_size=Decimal("0"))

    def test_negative_risk_amount_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_amount"):
            make_position_size(risk_amount=Decimal("-1"))

    def test_risk_pct_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_pct"):
            make_position_size(risk_pct=1.5)

    def test_risk_pct_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="risk_pct"):
            make_position_size(risk_pct=-0.01)

    def test_zero_stop_loss_pips_raises(self) -> None:
        with pytest.raises(ValueError, match="stop_loss_pips"):
            make_position_size(stop_loss_pips=Decimal("0"))

    def test_zero_pip_value_raises(self) -> None:
        with pytest.raises(ValueError, match="pip_value"):
            make_position_size(pip_value=Decimal("0"))

    def test_empty_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            make_position_size(method="")

    def test_take_profit_optional(self) -> None:
        ps = make_position_size(take_profit_price=Decimal("1.10300"))
        assert ps.take_profit_price == Decimal("1.10300")

    def test_frozen(self) -> None:
        ps = make_position_size()
        with pytest.raises(Exception):
            ps.lot_size = Decimal("1.0")  # type: ignore[misc]


class TestRiskVeto:
    def test_valid(self) -> None:
        veto = make_veto()
        assert veto.veto_code == "MAX_POSITIONS"

    def test_empty_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            make_veto(reason="")

    def test_empty_code_raises(self) -> None:
        with pytest.raises(ValueError, match="veto_code"):
            make_veto(veto_code="")

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            make_veto(timestamp_utc=datetime(2024, 1, 1))

    def test_context_optional(self) -> None:
        veto = make_veto()
        assert veto.context == {}

    def test_context_with_data(self) -> None:
        veto = make_veto(context={"current": 5, "max": 5})
        assert veto.context["current"] == 5

    def test_repr(self) -> None:
        veto = make_veto()
        assert "MAX_POSITIONS" in repr(veto)

    def test_frozen(self) -> None:
        veto = make_veto()
        with pytest.raises(Exception):
            veto.reason = "other"  # type: ignore[misc]


class TestRiskDecision:
    def test_approved_decision(self) -> None:
        ps = make_position_size()
        rd = RiskDecision(
            approved=True,
            symbol="EURUSD",
            side=OrderSide.BUY,
            timestamp_utc=T0,
            position_size=ps,
        )
        assert rd.approved
        assert rd.position_size is not None
        assert rd.veto is None

    def test_rejected_decision(self) -> None:
        veto = make_veto()
        rd = RiskDecision(
            approved=False,
            symbol="EURUSD",
            side=OrderSide.BUY,
            timestamp_utc=T0,
            veto=veto,
        )
        assert not rd.approved
        assert rd.veto is not None
        assert rd.position_size is None

    def test_approved_without_position_size_raises(self) -> None:
        with pytest.raises(ValueError, match="position_size"):
            RiskDecision(
                approved=True, symbol="EURUSD", side=OrderSide.BUY,
                timestamp_utc=T0, position_size=None,
            )

    def test_approved_with_veto_raises(self) -> None:
        with pytest.raises(ValueError, match="veto"):
            RiskDecision(
                approved=True, symbol="EURUSD", side=OrderSide.BUY,
                timestamp_utc=T0,
                position_size=make_position_size(),
                veto=make_veto(),
            )

    def test_rejected_without_veto_raises(self) -> None:
        with pytest.raises(ValueError, match="veto"):
            RiskDecision(
                approved=False, symbol="EURUSD", side=OrderSide.BUY,
                timestamp_utc=T0, veto=None,
            )

    def test_rejected_with_position_size_raises(self) -> None:
        with pytest.raises(ValueError, match="position_size"):
            RiskDecision(
                approved=False, symbol="EURUSD", side=OrderSide.BUY,
                timestamp_utc=T0,
                veto=make_veto(),
                position_size=make_position_size(),
            )

    def test_empty_symbol_raises(self) -> None:
        with pytest.raises(ValueError, match="symbol"):
            RiskDecision(
                approved=False, symbol="", side=OrderSide.BUY,
                timestamp_utc=T0, veto=make_veto(),
            )

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            RiskDecision(
                approved=False, symbol="EURUSD", side=OrderSide.BUY,
                timestamp_utc=datetime(2024, 1, 1),
                veto=make_veto(),
            )

    def test_repr_approved(self) -> None:
        rd = RiskDecision(
            approved=True, symbol="EURUSD", side=OrderSide.BUY,
            timestamp_utc=T0, position_size=make_position_size(),
        )
        assert "approved=True" in repr(rd)

    def test_repr_rejected(self) -> None:
        rd = RiskDecision(
            approved=False, symbol="EURUSD", side=OrderSide.BUY,
            timestamp_utc=T0, veto=make_veto(),
        )
        assert "approved=False" in repr(rd)


class TestKillSwitchState:
    def make_ks(self, level: KillSwitchLevel = KillSwitchLevel.NONE) -> KillSwitchState:
        return KillSwitchState(
            level=level,
            reason="daily loss limit",
            activated_at_utc=T0,
            activated_by="risk_manager",
        )

    def test_none_level_not_active(self) -> None:
        ks = self.make_ks(KillSwitchLevel.NONE)
        assert not ks.is_active
        assert not ks.blocks_new_trades
        assert not ks.blocks_all_activity

    def test_trade_gate_blocks_trades(self) -> None:
        ks = self.make_ks(KillSwitchLevel.TRADE_GATE)
        assert ks.is_active
        assert ks.blocks_new_trades
        assert not ks.blocks_all_activity

    def test_session_gate_blocks_trades(self) -> None:
        ks = self.make_ks(KillSwitchLevel.SESSION_GATE)
        assert ks.blocks_new_trades
        assert not ks.blocks_all_activity

    def test_emergency_halt_blocks_all(self) -> None:
        ks = self.make_ks(KillSwitchLevel.EMERGENCY_HALT)
        assert ks.blocks_new_trades
        assert ks.blocks_all_activity

    def test_hard_kill_blocks_all(self) -> None:
        ks = self.make_ks(KillSwitchLevel.HARD_KILL)
        assert ks.blocks_all_activity

    def test_empty_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            KillSwitchState(
                level=KillSwitchLevel.NONE, reason="",
                activated_at_utc=T0, activated_by="system",
            )

    def test_empty_activated_by_raises(self) -> None:
        with pytest.raises(ValueError, match="activated_by"):
            KillSwitchState(
                level=KillSwitchLevel.NONE, reason="x",
                activated_at_utc=T0, activated_by="",
            )

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            KillSwitchState(
                level=KillSwitchLevel.NONE, reason="x",
                activated_at_utc=datetime(2024, 1, 1), activated_by="system",
            )

    def test_repr(self) -> None:
        ks = self.make_ks(KillSwitchLevel.EMERGENCY_HALT)
        r = repr(ks)
        assert "EMERGENCY_HALT" in r

    def test_frozen(self) -> None:
        ks = self.make_ks()
        with pytest.raises(Exception):
            ks.level = KillSwitchLevel.HARD_KILL  # type: ignore[misc]
