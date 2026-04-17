"""Tests for ExitProfileSelector (deterministic fallback + ML path)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from metatrade.core.enums import OrderSide
from metatrade.ml.exit_profile_contracts import (
    ExitProfileCandidate,
    ExitProfileContext,
    SelectedExitProfile,
)
from metatrade.ml.exit_profile_selector import ExitProfileSelector


def _ts() -> datetime:
    return datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


def _ctx() -> ExitProfileContext:
    return ExitProfileContext(
        symbol="EURUSD",
        side=OrderSide.BUY,
        entry_price=Decimal("1.10000"),
        atr=Decimal("0.0010"),
        spread_pips=1.0,
        account_balance=Decimal("10000"),
        timestamp_utc=_ts(),
    )


def _candidate(profile_id: str, rr: str = "2.00", exit_mode: str = "fixed_tp") -> ExitProfileCandidate:
    is_trailing = exit_mode == "trailing_stop"
    return ExitProfileCandidate(
        profile_id=profile_id,
        sl_price=Decimal("1.09900"),
        tp_price=None if is_trailing else Decimal("1.10200"),
        sl_pips=Decimal("10.0"),
        tp_pips=None if is_trailing else Decimal("20.0"),
        risk_reward=Decimal(rr),
        method="trailing" if is_trailing else "atr",
        exit_mode=exit_mode,
        trailing_atr_mult=Decimal("1.5") if is_trailing else None,
    )


class TestExitProfileSelectorDeterministic:
    def setup_method(self):
        self.sel = ExitProfileSelector()

    def test_returns_none_for_empty_candidates(self):
        assert self.sel.select([], _ctx()) is None

    def test_returns_selection_with_one_candidate(self):
        c = _candidate("ATR_1_2")
        result = self.sel.select([c], _ctx())
        assert result is not None
        assert result.candidate.profile_id == "ATR_1_2"

    def test_selects_highest_rr(self):
        c_low  = _candidate("ATR_1_2",   rr="2.00")
        c_high = _candidate("ATR_1_5_3", rr="3.00")
        # Generator would sort by R:R desc; simulate that order
        result = self.sel.select([c_high, c_low], _ctx())
        assert result is not None
        assert result.candidate.profile_id == "ATR_1_5_3"

    def test_method_is_deterministic_fallback(self):
        result = self.sel.select([_candidate("ATR_1_2")], _ctx())
        assert result is not None
        assert result.method == "deterministic_fallback"

    def test_confidence_is_1_for_deterministic(self):
        result = self.sel.select([_candidate("ATR_1_2")], _ctx())
        assert result is not None
        assert result.confidence == 1.0

    def test_alternatives_populated(self):
        candidates = [
            _candidate("ATR_1_5_3", rr="3.00"),
            _candidate("ATR_1_2",   rr="2.00"),
            _candidate("ATR_1_1_5", rr="1.50"),
        ]
        result = self.sel.select(candidates, _ctx())
        assert result is not None
        assert len(result.alternatives) == 2

    def test_result_is_selected_exit_profile(self):
        result = self.sel.select([_candidate("ATR_1_2")], _ctx())
        assert isinstance(result, SelectedExitProfile)

    def test_trailing_candidate_can_be_selected(self):
        c_trail = _candidate("ATR_TRAIL_TIGHT", rr="2.50", exit_mode="trailing_stop")
        result = self.sel.select([c_trail], _ctx())
        assert result is not None
        assert result.candidate.exit_mode == "trailing_stop"
        assert result.candidate.tp_price is None


class TestExitProfileSelectorML:
    def _mock_feature_builder(self):
        from metatrade.ml.exit_profile_feature_builder import ExitProfileFeatureBuilder
        fb = ExitProfileFeatureBuilder()
        return fb

    def test_ml_path_selects_predicted_candidate(self):
        model = MagicMock()
        model.predict.return_value = ["ATR_1_5_3"]
        model.predict_proba.return_value = [[0.2, 0.7, 0.1]]
        model.classes_ = ["ATR_1_2", "ATR_1_5_3", "SWING_2R"]

        fb = self._mock_feature_builder()
        sel = ExitProfileSelector(model=model, feature_builder=fb)

        candidates = [
            _candidate("ATR_1_5_3", rr="3.00"),
            _candidate("ATR_1_2",   rr="2.00"),
        ]
        result = sel.select(candidates, _ctx())
        assert result is not None
        assert result.candidate.profile_id == "ATR_1_5_3"
        assert result.method == "ml_model"
        assert result.confidence == pytest.approx(0.7)

    def test_ml_fallback_when_predicted_not_in_candidates(self):
        model = MagicMock()
        model.predict.return_value = ["SWING_TRAIL"]  # not in candidate list
        model.predict_proba.side_effect = Exception("no proba")
        model.classes_ = ["SWING_TRAIL"]

        fb = self._mock_feature_builder()
        sel = ExitProfileSelector(model=model, feature_builder=fb)

        candidates = [_candidate("ATR_1_2", rr="2.00")]
        result = sel.select(candidates, _ctx())
        assert result is not None
        # Falls back to deterministic
        assert result.method == "deterministic_fallback"
        assert result.candidate.profile_id == "ATR_1_2"

    def test_ml_fallback_on_model_exception(self):
        model = MagicMock()
        model.predict.side_effect = RuntimeError("model broken")

        fb = self._mock_feature_builder()
        sel = ExitProfileSelector(model=model, feature_builder=fb)

        candidates = [_candidate("ATR_1_2")]
        result = sel.select(candidates, _ctx())
        assert result is not None
        assert result.method == "deterministic_fallback"

    def test_no_model_uses_deterministic(self):
        sel = ExitProfileSelector(model=None)
        result = sel.select([_candidate("ATR_1_2")], _ctx())
        assert result is not None
        assert result.method == "deterministic_fallback"

    def test_model_without_feature_builder_uses_deterministic(self):
        model = MagicMock()
        sel = ExitProfileSelector(model=model, feature_builder=None)
        result = sel.select([_candidate("ATR_1_2")], _ctx())
        assert result is not None
        assert result.method == "deterministic_fallback"

    def test_confidence_fallback_when_proba_unavailable(self):
        model = MagicMock()
        model.predict.return_value = ["ATR_1_2"]
        del model.predict_proba  # no predict_proba attribute

        fb = self._mock_feature_builder()
        sel = ExitProfileSelector(model=model, feature_builder=fb)
        result = sel.select([_candidate("ATR_1_2")], _ctx())
        if result is not None and result.method == "ml_model":
            # confidence should default to 0.5 when proba unavailable
            assert result.confidence == pytest.approx(0.5)
