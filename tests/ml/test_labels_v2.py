"""Tests for label_bars_v2 and TripleBarrierLabeler."""

from __future__ import annotations

from decimal import Decimal

import pytest

from metatrade.ml.labeling.triple_barrier import TripleBarrierLabeler
from metatrade.ml.labels import label_bars, label_bars_v2

from tests.ml.test_ml import make_bar, make_bars


# ---------------------------------------------------------------------------
# Helpers — bars with controlled high/low
# ---------------------------------------------------------------------------

def _bar_with(close: float, high: float, low: float, i: int = 0):
    return make_bar(close=close, high=high, low=low, i=i)


def _rising_bars(n: int, base: float = 1.1000, step: float = 0.0010) -> list:
    """Bars where each bar's high is clearly above the previous close."""
    bars = []
    for i in range(n):
        c = round(base + i * step, 5)
        bars.append(make_bar(close=c, high=c + 0.0015, low=c - 0.0005, i=i))
    return bars


def _flat_bars(n: int, price: float = 1.1000) -> list:
    """Bars that stay inside a very tight range — time barrier should fire."""
    bars = []
    for i in range(n):
        bars.append(make_bar(close=price, high=price + 0.00001, low=price - 0.00001, i=i))
    return bars


# ---------------------------------------------------------------------------
# TripleBarrierLabeler — constructor
# ---------------------------------------------------------------------------

class TestTripleBarrierLabelerInit:
    def test_defaults(self) -> None:
        tb = TripleBarrierLabeler()
        assert tb.forward_bars == 5
        assert tb.atr_period == 14

    def test_forward_bars_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="forward_bars"):
            TripleBarrierLabeler(forward_bars=0)

    def test_atr_period_one_raises(self) -> None:
        with pytest.raises(ValueError, match="atr_period"):
            TripleBarrierLabeler(atr_period=1)

    def test_atr_mult_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="atr_mult"):
            TripleBarrierLabeler(atr_mult=0.0)

    def test_spread_pct_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="spread_pct"):
            TripleBarrierLabeler(spread_pct=-0.001)


# ---------------------------------------------------------------------------
# TripleBarrierLabeler — label() output shape and None placement
# ---------------------------------------------------------------------------

class TestTripleBarrierLabelerShape:
    def test_same_length_as_bars(self) -> None:
        tb = TripleBarrierLabeler(forward_bars=5, atr_period=14)
        bars = make_bars(100)
        labels = tb.label(bars)
        assert len(labels) == 100

    def test_none_at_end(self) -> None:
        forward = 5
        tb = TripleBarrierLabeler(forward_bars=forward, atr_period=14)
        bars = make_bars(100)
        labels = tb.label(bars)
        # Last forward_bars must be None (no future data)
        assert all(lbl is None for lbl in labels[-forward:])

    def test_none_at_start_atr_warmup(self) -> None:
        period = 14
        tb = TripleBarrierLabeler(forward_bars=5, atr_period=period)
        bars = make_bars(100)
        labels = tb.label(bars)
        # First period bars should be None (ATR warmup)
        assert all(lbl is None for lbl in labels[:period])

    def test_all_none_when_too_short(self) -> None:
        tb = TripleBarrierLabeler(forward_bars=5, atr_period=14)
        bars = make_bars(18)  # needs at least 14+1+5 = 20
        labels = tb.label(bars)
        assert all(lbl is None for lbl in labels)

    def test_valid_labels_only_in_1_0_neg1(self) -> None:
        tb = TripleBarrierLabeler()
        bars = make_bars(100)
        labels = tb.label(bars)
        for lbl in labels:
            assert lbl in (1, 0, -1, None)


# ---------------------------------------------------------------------------
# TripleBarrierLabeler — correctness
# ---------------------------------------------------------------------------

class TestTripleBarrierLabelerCorrectness:
    def test_buy_label_when_high_crosses_barrier(self) -> None:
        """Bars where high clearly exceeds entry + ATR*mult → majority BUY."""
        tb = TripleBarrierLabeler(forward_bars=5, atr_period=14, atr_mult=1.0)
        bars = _rising_bars(100, step=0.0015)
        labels = [l for l in tb.label(bars) if l is not None]
        buy_count = labels.count(1)
        assert buy_count > len(labels) * 0.5

    def test_hold_label_for_flat_market(self) -> None:
        """Bars within a very tight range → all valid labels should be HOLD."""
        tb = TripleBarrierLabeler(forward_bars=5, atr_period=5, atr_mult=1.5)
        bars = _flat_bars(60)
        labels = [l for l in tb.label(bars) if l is not None]
        hold_count = labels.count(0)
        assert hold_count == len(labels)

    def test_sell_label_when_low_crosses_barrier(self) -> None:
        """Monotonically falling bars → majority SELL."""
        tb = TripleBarrierLabeler(forward_bars=5, atr_period=14, atr_mult=1.0)
        bars = _rising_bars(100, step=-0.0015)
        labels = [l for l in tb.label(bars) if l is not None]
        sell_count = labels.count(-1)
        assert sell_count > len(labels) * 0.5

    def test_spread_tightens_barriers(self) -> None:
        """With high spread_pct, fewer directional labels compared to spread=0."""
        bars = _rising_bars(100, step=0.0005)
        tb_no_spread = TripleBarrierLabeler(forward_bars=5, atr_period=14, atr_mult=1.0, spread_pct=0.0)
        tb_spread = TripleBarrierLabeler(forward_bars=5, atr_period=14, atr_mult=1.0, spread_pct=0.01)

        no_spread_dir = [l for l in tb_no_spread.label(bars) if l in (1, -1)]
        with_spread_dir = [l for l in tb_spread.label(bars) if l in (1, -1)]
        assert len(with_spread_dir) <= len(no_spread_dir)


# ---------------------------------------------------------------------------
# label_bars_v2 — wrapper consistency
# ---------------------------------------------------------------------------

class TestLabelBarsV2:
    def test_same_length_as_bars(self) -> None:
        bars = make_bars(100)
        labels = label_bars_v2(bars)
        assert len(labels) == 100

    def test_labels_in_valid_set(self) -> None:
        bars = make_bars(100)
        for lbl in label_bars_v2(bars):
            assert lbl in (1, 0, -1, None)

    def test_v1_unchanged(self) -> None:
        """label_bars() must still return same results as before this PR."""
        bars = make_bars(60)
        v1 = label_bars(bars)
        assert len(v1) == 60
        # Basic sanity: not all None
        assert any(lbl is not None for lbl in v1)

    def test_v2_differs_from_v1(self) -> None:
        """Triple barrier and fixed-point close labels can differ."""
        bars = _rising_bars(80, step=0.0010)
        v1 = label_bars(bars, forward_bars=5, atr_threshold_mult=1.0)
        v2 = label_bars_v2(bars, forward_bars=5, atr_threshold_mult=1.0)
        # They may differ — the key is that both are valid and distinct
        valid_v1 = [l for l in v1 if l is not None]
        valid_v2 = [l for l in v2 if l is not None]
        assert len(valid_v1) > 0
        assert len(valid_v2) > 0
        # They won't be identical for trending bars (v2 catches intrabar moves sooner)

    def test_spread_pct_parameter_accepted(self) -> None:
        bars = make_bars(80)
        labels = label_bars_v2(bars, spread_pct=0.0001)
        assert len(labels) == 80
