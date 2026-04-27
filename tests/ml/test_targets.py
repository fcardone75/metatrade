"""Tests for ml/targets.py — MlContinuousTarget and ContinuousTargetBuilder."""

from __future__ import annotations

from decimal import Decimal

import pytest

from metatrade.ml.targets import ContinuousTargetBuilder, MlContinuousTarget

# Reuse bar helpers from the main ML test file
from tests.ml.test_ml import make_bar, make_bars


# ---------------------------------------------------------------------------
# MlContinuousTarget
# ---------------------------------------------------------------------------

class TestMlContinuousTarget:
    def test_fields(self) -> None:
        t = MlContinuousTarget(bar_index=10, r_multiple=1.5, forward_bars=5, atr_value=0.0012)
        assert t.bar_index == 10
        assert t.r_multiple == 1.5
        assert t.forward_bars == 5
        assert t.atr_value == 0.0012

    def test_frozen(self) -> None:
        import dataclasses
        t = MlContinuousTarget(bar_index=0, r_multiple=0.0, forward_bars=5, atr_value=0.001)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            t.r_multiple = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ContinuousTargetBuilder — constructor validation
# ---------------------------------------------------------------------------

class TestContinuousTargetBuilderInit:
    def test_defaults(self) -> None:
        b = ContinuousTargetBuilder()
        assert b.forward_bars == 5
        assert b.atr_period == 14

    def test_forward_bars_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="forward_bars"):
            ContinuousTargetBuilder(forward_bars=0)

    def test_atr_period_one_raises(self) -> None:
        with pytest.raises(ValueError, match="atr_period"):
            ContinuousTargetBuilder(atr_period=1)

    def test_atr_mult_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="atr_mult"):
            ContinuousTargetBuilder(atr_mult=0.0)


# ---------------------------------------------------------------------------
# ContinuousTargetBuilder — build()
# ---------------------------------------------------------------------------

def _trending_bars(n: int, step: float = 0.0002) -> list:
    """Monotonically rising bars — predictable ATR and direction."""
    return make_bars(n, start=1.1000, step=step)


class TestContinuousTargetBuilderBuild:
    def test_empty_when_too_short(self) -> None:
        builder = ContinuousTargetBuilder(forward_bars=5, atr_period=14)
        # Need at least atr_period + 1 + forward_bars = 20 bars
        bars = make_bars(19)
        assert builder.build(bars) == []

    def test_returns_targets(self) -> None:
        builder = ContinuousTargetBuilder(forward_bars=5, atr_period=14)
        bars = make_bars(100)
        targets = builder.build(bars)
        assert len(targets) > 0

    def test_count_correct(self) -> None:
        forward = 5
        period = 14
        n = 100
        builder = ContinuousTargetBuilder(forward_bars=forward, atr_period=period)
        bars = make_bars(n)
        targets = builder.build(bars)
        # eligible: from index (period) to (n - forward - 1) inclusive
        min_i = period        # index of first bar where ATR is available (period+1 bars seen)
        max_i = n - forward - 1
        expected = max_i - min_i + 1
        assert len(targets) == expected

    def test_bar_index_in_range(self) -> None:
        n = 80
        builder = ContinuousTargetBuilder(forward_bars=5, atr_period=14)
        bars = make_bars(n)
        targets = builder.build(bars)
        for t in targets:
            assert 0 <= t.bar_index < n

    def test_forward_bars_preserved(self) -> None:
        builder = ContinuousTargetBuilder(forward_bars=3, atr_period=14)
        bars = make_bars(60)
        targets = builder.build(bars)
        assert all(t.forward_bars == 3 for t in targets)

    def test_atr_value_positive(self) -> None:
        builder = ContinuousTargetBuilder()
        bars = make_bars(100)
        targets = builder.build(bars)
        assert all(t.atr_value > 0 for t in targets)

    def test_r_multiple_sign_bullish(self) -> None:
        # Monotonically rising bars → r_multiple should be positive
        builder = ContinuousTargetBuilder(forward_bars=5, atr_period=14)
        bars = _trending_bars(100, step=0.0010)
        targets = builder.build(bars)
        positives = sum(1 for t in targets if t.r_multiple > 0)
        assert positives > len(targets) * 0.8  # most should be bullish

    def test_r_multiple_sign_bearish(self) -> None:
        # Monotonically falling bars → r_multiple should be negative
        builder = ContinuousTargetBuilder(forward_bars=5, atr_period=14)
        bars = _trending_bars(100, step=-0.0002)
        targets = builder.build(bars)
        negatives = sum(1 for t in targets if t.r_multiple < 0)
        assert negatives > len(targets) * 0.8

    def test_no_targets_beyond_bar_minus_forward(self) -> None:
        n = 50
        forward = 5
        builder = ContinuousTargetBuilder(forward_bars=forward, atr_period=14)
        bars = make_bars(n)
        targets = builder.build(bars)
        # Last bar index must be < n - forward
        if targets:
            assert max(t.bar_index for t in targets) < n - forward

    def test_custom_forward_and_period(self) -> None:
        builder = ContinuousTargetBuilder(forward_bars=2, atr_period=5)
        bars = make_bars(40)
        targets = builder.build(bars)
        assert len(targets) > 0
        assert all(t.forward_bars == 2 for t in targets)
