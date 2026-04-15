"""Tests for BreakEvenRule."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.exit_engine.config import BreakEvenConfig
from metatrade.exit_engine.contracts import ExitAction, PositionContext, PositionSide
from metatrade.exit_engine.rules.break_even import BreakEvenRule


def make_long_ctx(current_price="1.1000", stop_loss=None):
    return PositionContext(
        position_id="p1",
        symbol="EURUSD",
        side=PositionSide.LONG,
        entry_price=Decimal("1.1000"),
        lot_size=Decimal("0.1"),
        opened_at_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        current_price=Decimal(current_price),
        current_bar=None,
        bars_since_entry=[],
        peak_favorable_price=Decimal(current_price),
        stop_loss=Decimal(stop_loss) if stop_loss else None,
    )


class TestBreakEvenRule:
    def test_disabled_returns_hold(self):
        rule = BreakEvenRule(BreakEvenConfig(enabled=False))
        sig = rule.evaluate(make_long_ctx("1.1020"))
        assert sig.action == ExitAction.HOLD

    def test_below_trigger_holds(self):
        # trigger_pips=15, profit=10pip → not triggered
        rule = BreakEvenRule(BreakEvenConfig(trigger_pips=15.0, buffer_pips=2.0))
        ctx = make_long_ctx("1.1010")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.HOLD

    def test_at_trigger_returns_hold_with_suggested_sl(self):
        # profit=15pip → trigger hit → HOLD but suggested_sl = entry + buffer = 1.1000 + 0.0002
        rule = BreakEvenRule(BreakEvenConfig(trigger_pips=15.0, buffer_pips=2.0))
        ctx = make_long_ctx("1.1015")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.HOLD  # never directly closes

        sl = rule.suggested_sl(ctx)
        assert sl is not None
        expected = 1.1000 + 0.0002
        assert abs(sl - expected) < 1e-6

    def test_already_at_break_even_skips(self):
        # SL already at entry+buffer → should skip (no change)
        rule = BreakEvenRule(BreakEvenConfig(trigger_pips=15.0, buffer_pips=2.0))
        # SL = 1.1002 = entry + 2pip already
        ctx = make_long_ctx("1.1020", stop_loss="1.1002")
        sl = rule.suggested_sl(ctx)
        # Already at or above break-even → returns None
        assert sl is None

    def test_sl_above_entry_already_skip(self):
        # SL already beyond entry (1.1010 > entry 1.1000) — no update
        rule = BreakEvenRule(BreakEvenConfig(trigger_pips=15.0, buffer_pips=2.0))
        ctx = make_long_ctx("1.1020", stop_loss="1.1010")
        sl = rule.suggested_sl(ctx)
        assert sl is None
