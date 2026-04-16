"""Tests for ExitProfileCandidate, ExitProfileContext, SelectedExitProfile."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.core.enums import OrderSide
from metatrade.ml.exit_profile_contracts import (
    ExitProfileCandidate,
    ExitProfileContext,
    SelectedExitProfile,
)


def _ts() -> datetime:
    return datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc)


def _ctx(**kwargs) -> ExitProfileContext:
    defaults = dict(
        symbol="EURUSD",
        side=OrderSide.BUY,
        entry_price=Decimal("1.10000"),
        atr=Decimal("0.0010"),
        spread_pips=1.0,
        account_balance=Decimal("10000"),
        timestamp_utc=_ts(),
        pip_digits=4,
    )
    defaults.update(kwargs)
    return ExitProfileContext(**defaults)


def _candidate(**kwargs) -> ExitProfileCandidate:
    defaults = dict(
        profile_id="ATR_1_2",
        sl_price=Decimal("1.09900"),
        tp_price=Decimal("1.10200"),
        sl_pips=Decimal("10.0"),
        tp_pips=Decimal("20.0"),
        risk_reward=Decimal("2.00"),
        method="atr",
        exit_mode="fixed_tp",
        trailing_atr_mult=None,
    )
    defaults.update(kwargs)
    return ExitProfileCandidate(**defaults)


# ── ExitProfileCandidate ──────────────────────────────────────────────────────

class TestExitProfileCandidate:
    def test_valid_fixed_tp(self):
        c = _candidate()
        assert c.profile_id == "ATR_1_2"
        assert c.exit_mode == "fixed_tp"
        assert c.tp_price is not None
        assert c.trailing_atr_mult is None

    def test_valid_trailing_stop(self):
        c = _candidate(
            profile_id="ATR_TRAIL_TIGHT",
            tp_price=None,
            tp_pips=None,
            exit_mode="trailing_stop",
            trailing_atr_mult=Decimal("1.5"),
            method="trailing",
        )
        assert c.tp_price is None
        assert c.tp_pips is None
        assert c.trailing_atr_mult == Decimal("1.5")

    def test_valid_hybrid(self):
        c = _candidate(
            profile_id="ATR_HYBRID",
            exit_mode="hybrid",
            trailing_atr_mult=Decimal("1.0"),
        )
        assert c.exit_mode == "hybrid"
        assert c.tp_price is not None

    def test_empty_profile_id_raises(self):
        with pytest.raises(ValueError, match="profile_id"):
            _candidate(profile_id="")

    def test_sl_pips_zero_raises(self):
        with pytest.raises(ValueError, match="sl_pips"):
            _candidate(sl_pips=Decimal("0"))

    def test_sl_pips_negative_raises(self):
        with pytest.raises(ValueError, match="sl_pips"):
            _candidate(sl_pips=Decimal("-1"))

    def test_tp_pips_zero_raises(self):
        with pytest.raises(ValueError, match="tp_pips"):
            _candidate(tp_pips=Decimal("0"))

    def test_risk_reward_zero_raises(self):
        with pytest.raises(ValueError, match="risk_reward"):
            _candidate(risk_reward=Decimal("0"))

    def test_invalid_exit_mode_raises(self):
        with pytest.raises(ValueError, match="exit_mode"):
            _candidate(exit_mode="unknown_mode")

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            _candidate(method="magic")

    def test_trailing_null_tp_pips_allowed(self):
        # tp_pips=None is valid for trailing profiles
        c = _candidate(
            profile_id="TRAIL",
            tp_price=None,
            tp_pips=None,
            exit_mode="trailing_stop",
            method="trailing",
        )
        assert c.tp_pips is None


# ── ExitProfileContext ────────────────────────────────────────────────────────

class TestExitProfileContext:
    def test_valid_context(self):
        ctx = _ctx()
        assert ctx.pip_size == Decimal("0.0001")
        assert ctx.price_to_pips(Decimal("0.0010")) == Decimal("10.0")

    def test_jpy_pip_size(self):
        ctx = _ctx(pip_digits=2)
        assert ctx.pip_size == Decimal("0.01")
        assert ctx.price_to_pips(Decimal("0.10")) == Decimal("10.0")

    def test_atr_zero_raises(self):
        with pytest.raises(ValueError, match="atr"):
            _ctx(atr=Decimal("0"))

    def test_atr_negative_raises(self):
        with pytest.raises(ValueError, match="atr"):
            _ctx(atr=Decimal("-0.001"))

    def test_spread_negative_raises(self):
        with pytest.raises(ValueError, match="spread_pips"):
            _ctx(spread_pips=-0.5)

    def test_pip_digits_zero_raises(self):
        with pytest.raises(ValueError, match="pip_digits"):
            _ctx(pip_digits=0)

    def test_price_to_pips_symmetric(self):
        ctx = _ctx()
        assert ctx.price_to_pips(Decimal("-0.0010")) == Decimal("10.0")

    def test_recent_bars_default_empty(self):
        ctx = _ctx()
        assert ctx.recent_highs == ()
        assert ctx.recent_lows == ()

    def test_recent_bars_set(self):
        ctx = _ctx(
            recent_highs=(Decimal("1.101"), Decimal("1.102")),
            recent_lows=(Decimal("1.099"), Decimal("1.100")),
        )
        assert len(ctx.recent_highs) == 2


# ── SelectedExitProfile ───────────────────────────────────────────────────────

class TestSelectedExitProfile:
    def test_valid_selection(self):
        c = _candidate()
        s = SelectedExitProfile(
            candidate=c,
            confidence=0.85,
            method="ml_model",
            reason="ml_predicted:ATR_1_2",
        )
        assert s.confidence == 0.85
        assert s.alternatives == ()

    def test_with_alternatives(self):
        c1 = _candidate(profile_id="ATR_1_2")
        c2 = _candidate(
            profile_id="ATR_1_5_3",
            sl_price=Decimal("1.09850"),
            sl_pips=Decimal("15.0"),
            tp_pips=Decimal("45.0"),
            tp_price=Decimal("1.10450"),
            risk_reward=Decimal("3.00"),
        )
        s = SelectedExitProfile(
            candidate=c1,
            confidence=1.0,
            method="deterministic_fallback",
            reason="best_rr:2.0:ATR_1_2",
            alternatives=(c2,),
        )
        assert len(s.alternatives) == 1

    def test_confidence_above_1_raises(self):
        c = _candidate()
        with pytest.raises(ValueError, match="confidence"):
            SelectedExitProfile(
                candidate=c, confidence=1.1, method="m", reason="r"
            )

    def test_confidence_below_0_raises(self):
        c = _candidate()
        with pytest.raises(ValueError, match="confidence"):
            SelectedExitProfile(
                candidate=c, confidence=-0.1, method="m", reason="r"
            )

    def test_empty_method_raises(self):
        c = _candidate()
        with pytest.raises(ValueError, match="method"):
            SelectedExitProfile(
                candidate=c, confidence=0.5, method="", reason="r"
            )

    def test_empty_reason_raises(self):
        c = _candidate()
        with pytest.raises(ValueError, match="reason"):
            SelectedExitProfile(
                candidate=c, confidence=0.5, method="m", reason=""
            )
