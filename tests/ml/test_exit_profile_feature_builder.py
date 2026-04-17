"""Tests for ExitProfileFeatureBuilder."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from metatrade.core.enums import OrderSide
from metatrade.ml.exit_profile_contracts import ExitProfileContext
from metatrade.ml.exit_profile_feature_builder import ExitProfileFeatureBuilder


def _ts(hour: int = 10) -> datetime:
    return datetime(2024, 1, 15, hour, 0, tzinfo=UTC)


def _ctx(**kwargs) -> ExitProfileContext:
    highs = tuple(Decimal("1.10") + Decimal("0.0001") * i for i in range(30))
    lows  = tuple(Decimal("1.09") + Decimal("0.0001") * i for i in range(30))
    defaults = dict(
        symbol="EURUSD",
        side=OrderSide.BUY,
        entry_price=Decimal("1.10000"),
        atr=Decimal("0.0010"),
        spread_pips=1.5,
        account_balance=Decimal("10000"),
        timestamp_utc=_ts(),
        pip_digits=4,
        recent_highs=highs,
        recent_lows=lows,
    )
    defaults.update(kwargs)
    return ExitProfileContext(**defaults)


class TestExitProfileFeatureBuilder:
    def setup_method(self):
        self.builder = ExitProfileFeatureBuilder()

    def test_returns_all_feature_names(self):
        feats = self.builder.build(_ctx())
        for name in ExitProfileFeatureBuilder.FEATURE_NAMES:
            assert name in feats

    def test_all_values_are_finite(self):
        feats = self.builder.build(_ctx())
        for k, v in feats.items():
            assert math.isfinite(v), f"Feature {k} = {v} is not finite"

    def test_atr_pips_positive(self):
        feats = self.builder.build(_ctx())
        assert feats["atr_pips"] > 0

    def test_atr_pips_value(self):
        ctx = _ctx(atr=Decimal("0.0010"), pip_digits=4)
        feats = self.builder.build(ctx)
        assert abs(feats["atr_pips"] - 10.0) < 0.001

    def test_spread_pips_passthrough(self):
        feats = self.builder.build(_ctx(spread_pips=2.5))
        assert feats["spread_pips"] == pytest.approx(2.5)

    def test_is_buy_flag(self):
        buy_feats  = self.builder.build(_ctx(side=OrderSide.BUY))
        sell_feats = self.builder.build(_ctx(side=OrderSide.SELL))
        assert buy_feats["is_buy"] == 1.0
        assert sell_feats["is_buy"] == 0.0

    def test_hour_range(self):
        for h in [0, 6, 12, 18, 23]:
            feats = self.builder.build(_ctx(timestamp_utc=_ts(hour=h)))
            assert feats["hour"] == float(h)

    def test_hour_sin_cos_bounded(self):
        feats = self.builder.build(_ctx())
        assert -1.0 <= feats["hour_sin"] <= 1.0
        assert -1.0 <= feats["hour_cos"] <= 1.0

    def test_day_of_week_range(self):
        feats = self.builder.build(_ctx())
        # 2024-01-15 is a Monday (weekday 0)
        assert feats["day_of_week"] == 0.0

    def test_build_vector_length(self):
        vec = self.builder.build_vector(_ctx())
        assert len(vec) == len(ExitProfileFeatureBuilder.FEATURE_NAMES)

    def test_build_vector_order_matches_feature_names(self):
        ctx = _ctx()
        feats = self.builder.build(ctx)
        vec = self.builder.build_vector(ctx)
        expected = [feats[k] for k in ExitProfileFeatureBuilder.FEATURE_NAMES]
        assert vec == expected

    def test_no_bar_history_fallback(self):
        ctx = _ctx(recent_highs=(), recent_lows=())
        feats = self.builder.build(ctx)
        # recent_range_pips falls back to atr; should still be finite and > 0
        assert math.isfinite(feats["recent_range_pips"])
        assert feats["recent_range_pips"] >= 0.0

    def test_spread_atr_ratio(self):
        # spread=1.0 pips, atr=10.0 pips → ratio ≈ 0.1
        ctx = _ctx(atr=Decimal("0.0010"), spread_pips=1.0)
        feats = self.builder.build(ctx)
        assert abs(feats["spread_atr_ratio"] - 0.1) < 0.01
