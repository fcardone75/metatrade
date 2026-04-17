"""Tests for PartialExitRule."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from metatrade.exit_engine.config import PartialExitConfig, PartialExitLevel
from metatrade.exit_engine.contracts import ExitAction, PositionContext, PositionSide
from metatrade.exit_engine.rules.partial_exit import PartialExitRule


def make_ctx(current_price: str, partial_closes=()):
    return PositionContext(
        position_id="p1",
        symbol="EURUSD",
        side=PositionSide.LONG,
        entry_price=Decimal("1.1000"),
        lot_size=Decimal("0.1"),
        opened_at_utc=datetime(2024, 1, 1, tzinfo=UTC),
        current_price=Decimal(current_price),
        current_bar=None,
        bars_since_entry=[],
        peak_favorable_price=Decimal(current_price),
        partial_closes=partial_closes,
    )


DEFAULT_CFG = PartialExitConfig(
    enabled=True,
    levels=[
        PartialExitLevel(pips=20.0, close_pct=0.33),
        PartialExitLevel(pips=40.0, close_pct=0.33),
    ],
)


class TestPartialExitRule:
    def test_disabled_returns_hold(self):
        rule = PartialExitRule(PartialExitConfig(enabled=False))
        assert rule.evaluate(make_ctx("1.1030")).action == ExitAction.HOLD

    def test_no_levels_returns_hold(self):
        rule = PartialExitRule(PartialExitConfig(enabled=True, levels=[]))
        assert rule.evaluate(make_ctx("1.1030")).action == ExitAction.HOLD

    def test_below_first_level_holds(self):
        rule = PartialExitRule(DEFAULT_CFG)
        # 10pip < 20pip threshold
        ctx = make_ctx("1.1010")
        assert rule.evaluate(ctx).action == ExitAction.HOLD

    def test_first_level_triggers(self):
        rule = PartialExitRule(DEFAULT_CFG)
        # 25pip ≥ 20pip → CLOSE_PARTIAL at 33%
        ctx = make_ctx("1.1025")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.CLOSE_PARTIAL
        assert sig.close_pct == pytest.approx(0.33)

    def test_first_level_already_executed_skips_to_second(self):
        rule = PartialExitRule(DEFAULT_CFG)
        # First level already done; 45pip ≥ 40pip → second level
        ctx = make_ctx("1.1045", partial_closes=((20.0, 0.33),))
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.CLOSE_PARTIAL
        assert sig.close_pct == pytest.approx(0.33)
        assert "40" in sig.reason

    def test_both_levels_executed_holds(self):
        rule = PartialExitRule(DEFAULT_CFG)
        ctx = make_ctx("1.1060", partial_closes=((20.0, 0.33), (40.0, 0.33)))
        assert rule.evaluate(ctx).action == ExitAction.HOLD

    def test_only_first_level_on_first_bar(self):
        # Even if 45pip profit, only signals the first un-executed level
        rule = PartialExitRule(DEFAULT_CFG)
        ctx = make_ctx("1.1045")  # 45pip, no partial closes yet
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.CLOSE_PARTIAL
        # First level (20 pip) should be signalled, not second
        assert "20" in sig.reason
