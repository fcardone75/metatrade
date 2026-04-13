"""Tests for the kill switch manager."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from metatrade.core.contracts.risk import KillSwitchState
from metatrade.core.enums import KillSwitchLevel
from metatrade.core.errors import KillSwitchActiveError
from metatrade.risk.kill_switch import KillSwitchManager

UTC = timezone.utc
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


class TestKillSwitchManager:
    def test_initial_level_is_none(self) -> None:
        ks = KillSwitchManager()
        assert ks.level == KillSwitchLevel.NONE
        assert not ks.is_active()

    def test_activate_trade_gate(self) -> None:
        ks = KillSwitchManager()
        state = ks.activate(KillSwitchLevel.TRADE_GATE, "test", ts=T0)
        assert state.level == KillSwitchLevel.TRADE_GATE
        assert ks.blocks_new_trades()

    def test_activate_session_gate(self) -> None:
        ks = KillSwitchManager()
        ks.activate(KillSwitchLevel.SESSION_GATE, "test", ts=T0)
        assert ks.level == KillSwitchLevel.SESSION_GATE

    def test_activate_emergency_halt_blocks_all(self) -> None:
        ks = KillSwitchManager()
        ks.activate(KillSwitchLevel.EMERGENCY_HALT, "emergency", ts=T0)
        assert ks.blocks_all_activity()

    def test_level_only_escalates(self) -> None:
        ks = KillSwitchManager()
        ks.activate(KillSwitchLevel.SESSION_GATE, "high", ts=T0)
        # Try to activate a lower level — should be ignored
        ks.activate(KillSwitchLevel.TRADE_GATE, "low", ts=T0)
        assert ks.level == KillSwitchLevel.SESSION_GATE

    def test_cannot_activate_none(self) -> None:
        ks = KillSwitchManager()
        with pytest.raises(ValueError):
            ks.activate(KillSwitchLevel.NONE, "reset")

    def test_reset_from_trade_gate(self) -> None:
        ks = KillSwitchManager()
        ks.activate(KillSwitchLevel.TRADE_GATE, "test", ts=T0)
        ks.reset(activated_by="operator", ts=T0)
        assert ks.level == KillSwitchLevel.NONE
        assert not ks.is_active()

    def test_reset_from_hard_kill_raises(self) -> None:
        ks = KillSwitchManager()
        ks.activate(KillSwitchLevel.HARD_KILL, "manual", ts=T0)
        with pytest.raises(KillSwitchActiveError):
            ks.reset()

    def test_force_reset_from_hard_kill(self) -> None:
        ks = KillSwitchManager()
        ks.activate(KillSwitchLevel.HARD_KILL, "manual", ts=T0)
        ks.force_reset(activated_by="operator", ts=T0)
        assert ks.level == KillSwitchLevel.NONE

    def test_assert_trading_allowed_raises_when_active(self) -> None:
        ks = KillSwitchManager()
        ks.activate(KillSwitchLevel.TRADE_GATE, "test", ts=T0)
        with pytest.raises(KillSwitchActiveError):
            ks.assert_trading_allowed()

    def test_assert_trading_allowed_passes_when_inactive(self) -> None:
        ks = KillSwitchManager()
        ks.assert_trading_allowed()  # must not raise

    def test_state_is_immutable_snapshot(self) -> None:
        ks = KillSwitchManager()
        state_before = ks.state
        ks.activate(KillSwitchLevel.TRADE_GATE, "escalate", ts=T0)
        # state_before should still reflect the old level
        assert state_before.level == KillSwitchLevel.NONE
        assert ks.state.level == KillSwitchLevel.TRADE_GATE

    def test_reason_stored_in_state(self) -> None:
        ks = KillSwitchManager()
        ks.activate(KillSwitchLevel.TRADE_GATE, "max drawdown hit", ts=T0)
        assert "max drawdown" in ks.state.reason

    def test_activated_by_stored(self) -> None:
        ks = KillSwitchManager()
        ks.activate(KillSwitchLevel.TRADE_GATE, "test", activated_by="risk_module", ts=T0)
        assert ks.state.activated_by == "risk_module"

    def test_audit_store_called_on_activate(self) -> None:
        calls = []

        class FakeAudit:
            def log_kill_switch(self, level, reason, activated_by, ts):
                calls.append((level, reason))

        ks = KillSwitchManager(audit_store=FakeAudit())
        ks.activate(KillSwitchLevel.TRADE_GATE, "audit test", ts=T0)
        assert len(calls) == 1
        assert calls[0][0] == KillSwitchLevel.TRADE_GATE.value

    def test_audit_store_called_on_reset(self) -> None:
        calls = []

        class FakeAudit:
            def log_kill_switch(self, level, reason, activated_by, ts):
                calls.append(level)

        ks = KillSwitchManager(audit_store=FakeAudit())
        ks.activate(KillSwitchLevel.TRADE_GATE, "x", ts=T0)
        ks.reset(ts=T0)
        assert len(calls) == 2
        assert calls[-1] == KillSwitchLevel.NONE.value
