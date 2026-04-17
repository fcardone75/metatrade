"""Tests for ExitProfileCandidateGenerator."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from metatrade.core.enums import OrderSide
from metatrade.ml.exit_profile_candidate_generator import ExitProfileCandidateGenerator
from metatrade.ml.exit_profile_contracts import ExitProfileCandidate, ExitProfileContext


def _ts() -> datetime:
    return datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def _ctx(side: OrderSide = OrderSide.BUY, **kwargs) -> ExitProfileContext:
    highs = tuple(Decimal("1.10") + Decimal("0.0001") * i for i in range(30))
    lows  = tuple(Decimal("1.09") + Decimal("0.0001") * i for i in range(30))
    defaults = dict(
        symbol="EURUSD",
        side=side,
        entry_price=Decimal("1.10000"),
        atr=Decimal("0.0010"),
        spread_pips=1.0,
        account_balance=Decimal("10000"),
        timestamp_utc=_ts(),
        pip_digits=4,
        recent_highs=highs,
        recent_lows=lows,
    )
    defaults.update(kwargs)
    return ExitProfileContext(**defaults)


class TestExitProfileCandidateGeneratorBasic:
    def setup_method(self):
        self.gen = ExitProfileCandidateGenerator(min_stop_pips=5.0)

    def test_returns_non_empty_list(self):
        candidates = self.gen.generate(_ctx())
        assert len(candidates) > 0

    def test_all_candidates_valid_types(self):
        for c in self.gen.generate(_ctx()):
            assert isinstance(c, ExitProfileCandidate)

    def test_sorted_by_rr_descending(self):
        candidates = self.gen.generate(_ctx())
        rrs = [c.risk_reward for c in candidates]
        assert rrs == sorted(rrs, reverse=True)

    def test_buy_sl_below_entry(self):
        for c in self.gen.generate(_ctx(side=OrderSide.BUY)):
            assert c.sl_price < Decimal("1.10000")

    def test_sell_sl_above_entry(self):
        for c in self.gen.generate(_ctx(side=OrderSide.SELL)):
            assert c.sl_price > Decimal("1.10000")

    def test_buy_tp_above_entry_for_fixed_tp(self):
        for c in self.gen.generate(_ctx(side=OrderSide.BUY)):
            if c.tp_price is not None:
                assert c.tp_price > Decimal("1.10000")

    def test_sell_tp_below_entry_for_fixed_tp(self):
        for c in self.gen.generate(_ctx(side=OrderSide.SELL)):
            if c.tp_price is not None:
                assert c.tp_price < Decimal("1.10000")

    def test_sl_pips_above_min(self):
        gen = ExitProfileCandidateGenerator(min_stop_pips=5.0)
        for c in gen.generate(_ctx(spread_pips=0.0)):
            assert c.sl_pips >= Decimal("5.0")

    def test_sl_pips_above_spread_adjusted_min(self):
        gen = ExitProfileCandidateGenerator(min_stop_pips=5.0)
        spread = 2.0
        for c in gen.generate(_ctx(spread_pips=spread)):
            assert float(c.sl_pips) >= 5.0 + spread

    def test_contains_fixed_tp_profiles(self):
        candidates = self.gen.generate(_ctx())
        modes = {c.exit_mode for c in candidates}
        assert "fixed_tp" in modes

    def test_contains_trailing_stop_profiles(self):
        candidates = self.gen.generate(_ctx())
        modes = {c.exit_mode for c in candidates}
        assert "trailing_stop" in modes

    def test_contains_hybrid_profiles(self):
        candidates = self.gen.generate(_ctx())
        modes = {c.exit_mode for c in candidates}
        assert "hybrid" in modes

    def test_trailing_profiles_have_no_tp_price(self):
        for c in self.gen.generate(_ctx()):
            if c.exit_mode == "trailing_stop":
                assert c.tp_price is None
                assert c.tp_pips is None

    def test_trailing_profiles_have_trailing_atr_mult(self):
        for c in self.gen.generate(_ctx()):
            if c.exit_mode == "trailing_stop":
                assert c.trailing_atr_mult is not None
                assert c.trailing_atr_mult > 0

    def test_fixed_tp_profiles_have_tp_price(self):
        for c in self.gen.generate(_ctx()):
            if c.exit_mode == "fixed_tp":
                assert c.tp_price is not None
                assert c.tp_pips is not None

    def test_swing_candidates_present_with_bar_history(self):
        candidates = self.gen.generate(_ctx())
        ids = {c.profile_id for c in candidates}
        assert "SWING_2R" in ids or "SWING_TRAIL" in ids

    def test_no_swing_without_bar_history(self):
        ctx = _ctx(recent_highs=(), recent_lows=())
        candidates = self.gen.generate(ctx)
        ids = {c.profile_id for c in candidates}
        assert "SWING_2R" not in ids
        assert "SWING_TRAIL" not in ids

    def test_empty_result_when_min_stop_too_large(self):
        # ATR=0.001 → max SL dist ≈ 15 pips; min_stop=100 filters everything
        gen = ExitProfileCandidateGenerator(min_stop_pips=100.0)
        candidates = gen.generate(_ctx(recent_highs=(), recent_lows=()))
        assert candidates == []

    def test_rr_field_correct_for_fixed_tp(self):
        for c in self.gen.generate(_ctx()):
            if c.exit_mode == "fixed_tp" and c.tp_pips and c.sl_pips:
                expected_rr = (c.tp_pips / c.sl_pips).quantize(Decimal("0.01"))
                assert c.risk_reward == expected_rr

    def test_sell_side_generates_candidates(self):
        candidates = self.gen.generate(_ctx(side=OrderSide.SELL))
        assert len(candidates) > 0


class TestExitProfileGeneratorSwingDetail:
    def test_swing_sl_buy_below_recent_lows(self):
        lows = tuple(Decimal("1.095") + Decimal("0.0001") * i for i in range(30))
        highs = tuple(Decimal("1.100") + Decimal("0.0001") * i for i in range(30))
        ctx = _ctx(recent_highs=highs, recent_lows=lows)
        gen = ExitProfileCandidateGenerator(min_stop_pips=1.0)
        candidates = gen.generate(ctx)
        swing = next((c for c in candidates if c.profile_id == "SWING_2R"), None)
        if swing is not None:
            assert swing.sl_price < min(lows)

    def test_swing_sl_sell_above_recent_highs(self):
        lows = tuple(Decimal("1.09") + Decimal("0.0001") * i for i in range(30))
        highs = tuple(Decimal("1.099") + Decimal("0.0001") * i for i in range(30))
        ctx = _ctx(side=OrderSide.SELL, recent_highs=highs, recent_lows=lows)
        gen = ExitProfileCandidateGenerator(min_stop_pips=1.0)
        candidates = gen.generate(ctx)
        swing = next((c for c in candidates if c.profile_id == "SWING_2R"), None)
        if swing is not None:
            assert swing.sl_price > max(highs)
