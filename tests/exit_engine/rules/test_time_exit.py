"""Tests for TimeExitRule."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

from metatrade.exit_engine.config import TimeExitConfig
from metatrade.exit_engine.contracts import ExitAction, PositionContext, PositionSide
from metatrade.exit_engine.rules.time_exit import TimeExitRule


def make_ctx(n_bars: int, bar_hour: int = 12):
    """n_bars: number of bars held; bar_hour: hour of current bar's open time."""
    fake_bar = SimpleNamespace(
        timestamp_utc=datetime(2024, 1, 2, bar_hour, 0, tzinfo=UTC)
    )
    bars = [fake_bar] * n_bars
    return PositionContext(
        position_id="p1",
        symbol="EURUSD",
        side=PositionSide.LONG,
        entry_price=Decimal("1.1000"),
        lot_size=Decimal("0.1"),
        opened_at_utc=datetime(2024, 1, 1, tzinfo=UTC),
        current_price=Decimal("1.1010"),
        current_bar=fake_bar,
        bars_since_entry=bars,
        peak_favorable_price=Decimal("1.1010"),
    )


class TestTimeExitRule:
    def test_disabled_returns_hold(self):
        rule = TimeExitRule(TimeExitConfig(enabled=False))
        ctx = make_ctx(100)
        assert rule.evaluate(ctx).action == ExitAction.HOLD

    def test_under_max_bars_holds(self):
        rule = TimeExitRule(TimeExitConfig(max_holding_bars=48))
        ctx = make_ctx(40)
        assert rule.evaluate(ctx).action == ExitAction.HOLD

    def test_at_max_bars_exits(self):
        rule = TimeExitRule(TimeExitConfig(max_holding_bars=48))
        ctx = make_ctx(48)
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.CLOSE_FULL

    def test_over_max_bars_exits(self):
        rule = TimeExitRule(TimeExitConfig(max_holding_bars=48))
        ctx = make_ctx(100)
        assert rule.evaluate(ctx).action == ExitAction.CLOSE_FULL

    def test_session_end_triggers_exit(self):
        # bar_hour=21, session_end=21 → triggers
        rule = TimeExitRule(TimeExitConfig(max_holding_bars=100, close_on_session_end=True, session_end_hour_utc=21))
        ctx = make_ctx(10, bar_hour=21)
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.CLOSE_FULL

    def test_before_session_end_holds(self):
        rule = TimeExitRule(TimeExitConfig(max_holding_bars=100, close_on_session_end=True, session_end_hour_utc=21))
        ctx = make_ctx(10, bar_hour=15)
        assert rule.evaluate(ctx).action == ExitAction.HOLD

    def test_session_end_disabled(self):
        rule = TimeExitRule(TimeExitConfig(max_holding_bars=100, close_on_session_end=False, session_end_hour_utc=21))
        ctx = make_ctx(10, bar_hour=21)
        assert rule.evaluate(ctx).action == ExitAction.HOLD
