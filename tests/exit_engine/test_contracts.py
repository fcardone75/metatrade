"""Tests for exit engine contracts (PositionContext, ExitSignal, etc.)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.exit_engine.contracts import (
    ExitAction,
    ExitDecision,
    ExitSignal,
    PositionContext,
    PositionSide,
    TradeOutcome,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def long_ctx():
    return PositionContext(
        position_id="p1",
        symbol="EURUSD",
        side=PositionSide.LONG,
        entry_price=Decimal("1.1000"),
        lot_size=Decimal("0.1"),
        opened_at_utc=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        current_price=Decimal("1.1030"),
        current_bar=None,
        bars_since_entry=[],
        peak_favorable_price=Decimal("1.1040"),
    )


@pytest.fixture
def short_ctx():
    return PositionContext(
        position_id="p2",
        symbol="EURUSD",
        side=PositionSide.SHORT,
        entry_price=Decimal("1.1000"),
        lot_size=Decimal("0.1"),
        opened_at_utc=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc),
        current_price=Decimal("1.0970"),
        current_bar=None,
        bars_since_entry=[],
        peak_favorable_price=Decimal("1.0960"),
    )


# ── PositionContext ────────────────────────────────────────────────────────────

class TestPositionContext:
    def test_unrealized_pips_long(self, long_ctx):
        # 1.1030 - 1.1000 = 0.0030 / 0.0001 = 30 pips
        assert long_ctx.unrealized_pips == Decimal("30")

    def test_unrealized_pips_short(self, short_ctx):
        # 1.1000 - 1.0970 = 0.0030 / 0.0001 = 30 pips
        assert short_ctx.unrealized_pips == Decimal("30")

    def test_peak_profit_pips_long(self, long_ctx):
        # peak 1.1040 - entry 1.1000 = 40 pips
        assert long_ctx.peak_profit_pips == Decimal("40")

    def test_peak_profit_pips_short(self, short_ctx):
        # entry 1.1000 - peak 1.0960 = 40 pips
        assert short_ctx.peak_profit_pips == Decimal("40")

    def test_bars_held(self, long_ctx):
        assert long_ctx.bars_held == 0

    def test_rejects_naive_datetime(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            PositionContext(
                position_id="p",
                symbol="EURUSD",
                side=PositionSide.LONG,
                entry_price=Decimal("1.1"),
                lot_size=Decimal("0.1"),
                opened_at_utc=datetime(2024, 1, 1),  # naive
                current_price=Decimal("1.1"),
                current_bar=None,
                bars_since_entry=[],
                peak_favorable_price=Decimal("1.1"),
            )

    def test_rejects_zero_lot_size(self):
        with pytest.raises(ValueError, match="lot_size"):
            PositionContext(
                position_id="p",
                symbol="EURUSD",
                side=PositionSide.LONG,
                entry_price=Decimal("1.1"),
                lot_size=Decimal("0"),
                opened_at_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
                current_price=Decimal("1.1"),
                current_bar=None,
                bars_since_entry=[],
                peak_favorable_price=Decimal("1.1"),
            )


# ── ExitSignal ────────────────────────────────────────────────────────────────

class TestExitSignal:
    def test_valid_hold(self):
        s = ExitSignal(
            rule_id="test",
            action=ExitAction.HOLD,
            close_pct=0.0,
            reason="ok",
            confidence=0.5,
        )
        assert s.action == ExitAction.HOLD

    def test_rejects_invalid_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            ExitSignal(rule_id="x", action=ExitAction.HOLD, close_pct=0.0,
                       reason="x", confidence=1.5)

    def test_rejects_invalid_close_pct(self):
        with pytest.raises(ValueError, match="close_pct"):
            ExitSignal(rule_id="x", action=ExitAction.HOLD, close_pct=-0.1,
                       reason="x", confidence=0.5)

    def test_rejects_empty_rule_id(self):
        with pytest.raises(ValueError, match="rule_id"):
            ExitSignal(rule_id="", action=ExitAction.HOLD, close_pct=0.0,
                       reason="x", confidence=0.5)

    def test_rejects_empty_reason(self):
        with pytest.raises(ValueError, match="reason"):
            ExitSignal(rule_id="x", action=ExitAction.HOLD, close_pct=0.0,
                       reason="", confidence=0.5)


# ── ExitDecision ──────────────────────────────────────────────────────────────

class TestExitDecision:
    def test_is_exit_false_for_hold(self):
        d = ExitDecision(
            action=ExitAction.HOLD,
            close_pct=0.0,
            aggregate_score=0.0,
            signals=(),
            explanation="hold",
            timestamp_utc=datetime.now(timezone.utc),
        )
        assert not d.is_exit

    def test_is_exit_true_for_close_full(self):
        d = ExitDecision(
            action=ExitAction.CLOSE_FULL,
            close_pct=1.0,
            aggregate_score=80.0,
            signals=(),
            explanation="exit",
            timestamp_utc=datetime.now(timezone.utc),
        )
        assert d.is_exit
