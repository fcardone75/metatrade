"""Tests for TrailingStopRule."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from metatrade.exit_engine.config import TrailingStopConfig
from metatrade.exit_engine.contracts import ExitAction, PositionContext, PositionSide
from metatrade.exit_engine.rules.trailing_stop import TrailingStopRule


def make_long_ctx(
    entry_price="1.1000",
    current_price="1.1030",
    peak="1.1040",
    entry_atr=None,
    stop_loss=None,
    bars=None,
):
    return PositionContext(
        position_id="p1",
        symbol="EURUSD",
        side=PositionSide.LONG,
        entry_price=Decimal(entry_price),
        lot_size=Decimal("0.1"),
        opened_at_utc=datetime(2024, 1, 1, tzinfo=UTC),
        current_price=Decimal(current_price),
        current_bar=None,
        bars_since_entry=bars or [],
        peak_favorable_price=Decimal(peak),
        entry_atr=Decimal(entry_atr) if entry_atr else None,
        stop_loss=Decimal(stop_loss) if stop_loss else None,
    )


def make_short_ctx(
    entry_price="1.1000",
    current_price="1.0970",
    peak="1.0960",
    entry_atr=None,
    stop_loss=None,
):
    return PositionContext(
        position_id="p2",
        symbol="EURUSD",
        side=PositionSide.SHORT,
        entry_price=Decimal(entry_price),
        lot_size=Decimal("0.1"),
        opened_at_utc=datetime(2024, 1, 1, tzinfo=UTC),
        current_price=Decimal(current_price),
        current_bar=None,
        bars_since_entry=[],
        peak_favorable_price=Decimal(peak),
        entry_atr=Decimal(entry_atr) if entry_atr else None,
        stop_loss=Decimal(stop_loss) if stop_loss else None,
    )


class TestTrailingStopRule:
    def test_disabled_returns_hold(self):
        rule = TrailingStopRule(TrailingStopConfig(enabled=False))
        ctx = make_long_ctx(entry_atr="0.0010")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.HOLD

    def test_hold_when_no_atr(self):
        rule = TrailingStopRule(TrailingStopConfig(mode="atr"))
        ctx = make_long_ctx()  # no entry_atr
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.HOLD

    def test_long_above_trailing_sl_holds(self):
        # entry_atr=20pip, mult=2 → distance=40pip
        # peak=1.1040, trailing_sl = 1.1040 - 0.0040 = 1.1000
        # current=1.1030 > 1.1000 → HOLD
        rule = TrailingStopRule(TrailingStopConfig(mode="atr", atr_mult=2.0))
        ctx = make_long_ctx(current_price="1.1030", peak="1.1040", entry_atr="0.0020")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.HOLD

    def test_long_below_trailing_sl_exits(self):
        # peak=1.1040, entry_atr=0.0020, mult=2 → distance=0.0040
        # trailing_sl = 1.1040 - 0.0040 = 1.1000
        # current=1.0998 < 1.1000 → CLOSE_FULL
        rule = TrailingStopRule(TrailingStopConfig(mode="atr", atr_mult=2.0))
        ctx = make_long_ctx(current_price="1.0998", peak="1.1040", entry_atr="0.0020")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.CLOSE_FULL

    def test_short_above_trailing_sl_exits(self):
        # SHORT: peak_favorable=1.0960, entry_atr=0.0020, mult=2 → distance=0.0040
        # trailing_sl = 1.0960 + 0.0040 = 1.1000
        # current=1.1002 > 1.1000 → CLOSE_FULL
        rule = TrailingStopRule(TrailingStopConfig(mode="atr", atr_mult=2.0))
        ctx = make_short_ctx(current_price="1.1002", peak="1.0960", entry_atr="0.0020")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.CLOSE_FULL

    def test_suggested_sl_returned_for_long(self):
        rule = TrailingStopRule(TrailingStopConfig(mode="atr", atr_mult=2.0))
        ctx = make_long_ctx(current_price="1.1030", peak="1.1040", entry_atr="0.0020")
        sl = rule.suggested_sl(ctx)
        assert sl is not None
        # trailing_sl = 1.1040 - 0.0040 = 1.1000
        assert abs(sl - 1.1000) < 0.0001

    def test_fixed_pips_mode(self):
        rule = TrailingStopRule(TrailingStopConfig(mode="fixed_pips", fixed_pips=30.0))
        # peak=1.1040, distance=30pip=0.0030, trailing_sl=1.1010
        # current=1.1005 < 1.1010 → CLOSE_FULL
        ctx = make_long_ctx(current_price="1.1005", peak="1.1040", entry_atr="0.0010")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.CLOSE_FULL

    def test_fixed_pct_mode(self):
        rule = TrailingStopRule(TrailingStopConfig(mode="fixed_pct", fixed_pct=0.003))
        # peak=1.1040, distance=1.1040×0.003=0.003312
        # trailing_sl ≈ 1.1040 - 0.003312 ≈ 1.10069
        # current=1.1000 < trailing_sl → CLOSE_FULL
        ctx = make_long_ctx(current_price="1.1000", peak="1.1040", entry_atr="0.0010")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.CLOSE_FULL

    def test_unknown_mode_returns_hold(self):
        """Unknown mode → _compute_sl returns None → HOLD."""
        rule = TrailingStopRule(TrailingStopConfig(mode="unknown_mode"))
        ctx = make_long_ctx(current_price="1.1020", peak="1.1040", entry_atr="0.0010")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.HOLD

    def test_atr_computed_from_bars(self):
        """With enough bars, ATR is computed from actual bar data (not entry_atr)."""
        from datetime import datetime

        from metatrade.core.contracts.market import Bar

        rule = TrailingStopRule(TrailingStopConfig(mode="atr", atr_period=5, atr_mult=2.0))

        # Build 15 bars with consistent ranges
        bars = []
        for i in range(15):
            close = Decimal(str(1.1000 + i * 0.001))
            high = close + Decimal("0.0005")
            low = close - Decimal("0.0005")
            bars.append(Bar(
                symbol="EURUSD",
                timeframe="M15",
                timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC),
                open=close,
                high=high,
                low=low,
                close=close,
                volume=100,
            ))

        ctx = make_long_ctx(
            entry_price="1.1000",
            current_price="1.1140",
            peak="1.1140",
            entry_atr="0.0010",
            bars=bars,
        )
        sl = rule.suggested_sl(ctx)
        assert sl is not None
        assert isinstance(sl, float)

    def test_short_sl_not_hit_holds(self):
        rule = TrailingStopRule(TrailingStopConfig(mode="atr", atr_mult=2.0))
        # SHORT peak=1.0960, atr=0.0020, mult=2 → sl=1.0960+0.0040=1.1000
        # current=1.0990 < sl=1.1000 → not hit → HOLD
        ctx = make_short_ctx(current_price="1.0990", peak="1.0960", entry_atr="0.0020")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.HOLD
