"""Tests for GiveBackRule."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.exit_engine.config import GiveBackConfig
from metatrade.exit_engine.contracts import ExitAction, PositionContext, PositionSide
from metatrade.exit_engine.rules.give_back import GiveBackRule


def make_ctx(current_price, peak_price, side=PositionSide.LONG):
    entry = "1.1000"
    return PositionContext(
        position_id="p1",
        symbol="EURUSD",
        side=side,
        entry_price=Decimal(entry),
        lot_size=Decimal("0.1"),
        opened_at_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        current_price=Decimal(current_price),
        current_bar=None,
        bars_since_entry=[],
        peak_favorable_price=Decimal(peak_price),
    )


class TestGiveBackRule:
    def test_disabled_returns_hold(self):
        rule = GiveBackRule(GiveBackConfig(enabled=False))
        ctx = make_ctx("1.1010", "1.1040")
        assert rule.evaluate(ctx).action == ExitAction.HOLD

    def test_no_peak_profit_holds(self):
        # peak == entry → peak_profit_pips == 0
        rule = GiveBackRule(GiveBackConfig(give_back_pct=0.40))
        ctx = make_ctx("1.1000", "1.1000")  # peak = entry
        assert rule.evaluate(ctx).action == ExitAction.HOLD

    def test_no_giveback_holds(self):
        # peak=40pip, give_back=40% → min_keep=24pip
        # current=30pip > 24pip → HOLD
        rule = GiveBackRule(GiveBackConfig(give_back_pct=0.40))
        ctx = make_ctx("1.1030", "1.1040")  # 30pip profit, peak 40pip
        assert rule.evaluate(ctx).action == ExitAction.HOLD

    def test_gives_back_triggers_exit(self):
        # peak=40pip, give_back=40% → min_keep=24pip
        # current=20pip < 24pip → CLOSE_FULL
        rule = GiveBackRule(GiveBackConfig(give_back_pct=0.40))
        ctx = make_ctx("1.1020", "1.1040")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.CLOSE_FULL

    def test_confidence_scales_with_giveback(self):
        # More give-back → higher confidence
        rule = GiveBackRule(GiveBackConfig(give_back_pct=0.30))
        ctx_small = make_ctx("1.1027", "1.1040")  # just crossed threshold
        ctx_large = make_ctx("1.1000", "1.1040")  # 100% give-back (back to entry)
        sig_small = rule.evaluate(ctx_small)
        sig_large = rule.evaluate(ctx_large)
        assert sig_large.confidence >= sig_small.confidence

    def test_short_gives_back_triggers_exit(self):
        # SHORT: entry=1.1000, peak_favorable=1.0960 (40pip profit)
        # give_back=40% → min_keep=24pip → price must stay below 1.0976
        # current=1.0980 → current_pips = 1.1000 - 1.0980 = 20pip < 24pip → exit
        rule = GiveBackRule(GiveBackConfig(give_back_pct=0.40))
        ctx = make_ctx("1.0980", "1.0960", side=PositionSide.SHORT)
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.CLOSE_FULL
