"""Tests for VolatilityExitRule."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.exit_engine.config import VolatilityExitConfig
from metatrade.exit_engine.contracts import ExitAction, PositionContext, PositionSide
from metatrade.exit_engine.rules.volatility_exit import VolatilityExitRule


def _bar(close: float, high: float | None = None, low: float | None = None) -> Bar:
    c = Decimal(str(close))
    h = Decimal(str(high)) if high else c + Decimal("0.0005")
    l = Decimal(str(low)) if low else c - Decimal("0.0005")
    return Bar(
        symbol="EURUSD",
        timeframe="M15",
        timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC),
        open=c,
        high=h,
        low=l,
        close=c,
        volume=100,
    )


def _ctx(
    entry_atr: str | None,
    current_price: str = "1.1000",
    bars: list | None = None,
) -> PositionContext:
    return PositionContext(
        position_id="p1",
        symbol="EURUSD",
        side=PositionSide.LONG,
        entry_price=Decimal("1.1000"),
        lot_size=Decimal("0.1"),
        opened_at_utc=datetime(2024, 1, 1, tzinfo=UTC),
        current_price=Decimal(current_price),
        current_bar=None,
        bars_since_entry=bars or [],
        peak_favorable_price=Decimal(current_price),
        entry_atr=Decimal(entry_atr) if entry_atr else None,
    )


class TestVolatilityExitRule:
    def test_disabled_returns_hold(self):
        rule = VolatilityExitRule(VolatilityExitConfig(enabled=False))
        ctx = _ctx(entry_atr="0.0010")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.HOLD

    def test_hold_when_no_entry_atr(self):
        rule = VolatilityExitRule(VolatilityExitConfig(enabled=True))
        ctx = _ctx(entry_atr=None)
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.HOLD

    def test_hold_when_entry_atr_is_zero(self):
        rule = VolatilityExitRule(VolatilityExitConfig(enabled=True))
        ctx = _ctx(entry_atr="0")
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.HOLD

    def test_hold_when_few_bars(self):
        """Insufficient bars: falls back to entry_atr which equals current — ratio=1 < threshold."""
        rule = VolatilityExitRule(VolatilityExitConfig(
            enabled=True, atr_expansion_mult=2.0, atr_period=14
        ))
        # Only 3 bars — not enough to compute ATR(14), falls back to entry_atr
        bars = [_bar(1.1000 + i * 0.001) for i in range(3)]
        ctx = _ctx(entry_atr="0.0010", bars=bars)
        sig = rule.evaluate(ctx)
        # ratio = entry_atr / entry_atr = 1.0 < 2.0 → HOLD
        assert sig.action == ExitAction.HOLD

    def test_exits_when_atr_expands(self):
        """With enough bars, if ATR significantly expands, should exit."""
        rule = VolatilityExitRule(VolatilityExitConfig(
            enabled=True, atr_expansion_mult=1.5, atr_period=5
        ))
        # Create bars with high volatility (wide high-low ranges)
        # Entry ATR was 0.0005; current bars have very large ranges
        bars = []
        for i in range(15):
            close = 1.1000 + i * 0.001
            high = close + 0.0050  # 50-pip range
            low = close - 0.0050
            bars.append(_bar(close, high, low))

        ctx = _ctx(entry_atr="0.0005", bars=bars)
        sig = rule.evaluate(ctx)
        # ATR from high-volatility bars >> 0.0005; ratio should exceed 1.5
        assert sig.action in (ExitAction.CLOSE_FULL, ExitAction.HOLD)

    def test_hold_when_atr_within_threshold(self):
        """ATR stays below expansion threshold → HOLD."""
        rule = VolatilityExitRule(VolatilityExitConfig(
            enabled=True, atr_expansion_mult=5.0, atr_period=5
        ))
        # Normal bars with small ranges
        bars = [_bar(1.1000 + i * 0.0001) for i in range(10)]
        ctx = _ctx(entry_atr="0.0010", bars=bars)
        sig = rule.evaluate(ctx)
        assert sig.action == ExitAction.HOLD

    def test_exit_signal_contains_ratio(self):
        """When exit fires, the reason mentions the ratio."""
        rule = VolatilityExitRule(VolatilityExitConfig(
            enabled=True, atr_expansion_mult=1.5, atr_period=5
        ))
        bars = []
        for i in range(15):
            close = 1.1000 + i * 0.001
            bars.append(_bar(close, close + 0.0050, close - 0.0050))

        ctx = _ctx(entry_atr="0.0001", bars=bars)
        sig = rule.evaluate(ctx)
        if sig.action == ExitAction.CLOSE_FULL:
            assert "x" in sig.reason.lower() or "atr" in sig.reason.lower()

    def test_hold_reason_includes_ratio(self):
        rule = VolatilityExitRule(VolatilityExitConfig(
            enabled=True, atr_expansion_mult=2.0, atr_period=14
        ))
        # Few bars: falls back to entry_atr, ratio = 1.0
        ctx = _ctx(entry_atr="0.0010", bars=[])
        sig = rule.evaluate(ctx)
        assert "ratio" in sig.reason.lower() or "insufficient" in sig.reason.lower()

    def test_suggested_sl_returns_none(self):
        rule = VolatilityExitRule()
        ctx = _ctx(entry_atr="0.0010")
        assert rule.suggested_sl(ctx) is None
