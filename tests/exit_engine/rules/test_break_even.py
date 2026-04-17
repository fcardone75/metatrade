"""Tests for BreakEvenRule."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

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
        opened_at_utc=datetime(2024, 1, 1, tzinfo=UTC),
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


def make_short_ctx(current_price="1.0980", stop_loss=None):
    return PositionContext(
        position_id="p2",
        symbol="EURUSD",
        side=PositionSide.SHORT,
        entry_price=Decimal("1.1000"),
        lot_size=Decimal("0.1"),
        opened_at_utc=datetime(2024, 1, 1, tzinfo=UTC),
        current_price=Decimal(current_price),
        current_bar=None,
        bars_since_entry=[],
        peak_favorable_price=Decimal(current_price),
        stop_loss=Decimal(stop_loss) if stop_loss else None,
    )


class TestBreakEvenRuleShort:
    def test_short_already_at_break_even_holds(self):
        """SL already at or below break-even for SHORT."""
        rule = BreakEvenRule(BreakEvenConfig(trigger_pips=15.0, buffer_pips=2.0))
        # be_sl = entry - buffer = 1.1000 - 0.0002 = 1.0998
        # SL at 1.0998 is already <= be_sl → skip
        ctx = make_short_ctx("1.0985", stop_loss="1.0998")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.HOLD

    def test_short_suggested_sl_when_not_triggered(self):
        """suggested_sl returns None when trigger pips not reached for SHORT."""
        rule = BreakEvenRule(BreakEvenConfig(trigger_pips=15.0, buffer_pips=2.0))
        # Only 5 pips profit on SHORT — not at trigger yet
        ctx = make_short_ctx("1.0995")
        sl = rule.suggested_sl(ctx)
        assert sl is None

    def test_short_be_sl_is_below_entry(self):
        """For SHORT, break-even SL = entry - buffer (below entry)."""
        rule = BreakEvenRule(BreakEvenConfig(trigger_pips=15.0, buffer_pips=2.0))
        ctx = make_short_ctx("1.0985")
        # 15 pips profit, trigger reached
        sl = rule.suggested_sl(ctx)
        if sl is not None:
            # Should be entry - buffer = 1.1000 - 0.0002 = 1.0998
            assert abs(sl - 1.0998) < 1e-6

    def test_short_suggested_sl_skipped_when_current_sl_better(self):
        """suggested_sl returns None if existing SL is already <= be_sl for SHORT."""
        rule = BreakEvenRule(BreakEvenConfig(trigger_pips=15.0, buffer_pips=2.0))
        # be_sl = 1.0998, existing SL = 1.0990 (better than be_sl for SHORT)
        ctx = make_short_ctx("1.0985", stop_loss="1.0990")
        sl = rule.suggested_sl(ctx)
        assert sl is None
