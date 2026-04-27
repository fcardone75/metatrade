"""Tests for ml/prediction.py — build_ml_prediction."""

from __future__ import annotations

import numpy as np
import pytest

from metatrade.ml.contracts import MlPrediction
from metatrade.ml.prediction import build_ml_prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _probas(*values: float) -> np.ndarray:
    arr = np.array(values, dtype=float)
    return arr / arr.sum()  # normalise


# ---------------------------------------------------------------------------
# No calibration — raw probas
# ---------------------------------------------------------------------------

class TestBuildMlPredictionRaw:
    def test_returns_ml_prediction(self) -> None:
        p = _probas(0.2, 0.5, 0.3)
        result = build_ml_prediction(
            raw_direction=0,
            probas=p,
            classes=[-1, 0, 1],
        )
        assert isinstance(result, MlPrediction)

    def test_direction_from_argmax(self) -> None:
        # HOLD (index 1, class 0) has highest prob
        p = _probas(0.1, 0.7, 0.2)
        result = build_ml_prediction(
            raw_direction=0,
            probas=p,
            classes=[-1, 0, 1],
        )
        assert result.direction == 0  # HOLD

    def test_direction_buy(self) -> None:
        # BUY (index 2, class 1) has highest prob
        p = _probas(0.1, 0.1, 0.8)
        result = build_ml_prediction(
            raw_direction=1,
            probas=p,
            classes=[-1, 0, 1],
        )
        assert result.direction == 1

    def test_confidence_is_max_proba(self) -> None:
        p = _probas(0.1, 0.6, 0.3)
        result = build_ml_prediction(raw_direction=0, probas=p, classes=[-1, 0, 1])
        assert abs(result.confidence - p.max()) < 1e-9

    def test_p_buy_p_hold_p_sell_set(self) -> None:
        p = np.array([0.2, 0.5, 0.3])  # already sums to 1
        result = build_ml_prediction(raw_direction=0, probas=p, classes=[-1, 0, 1])
        assert result.p_sell is not None
        assert result.p_hold is not None
        assert result.p_buy is not None
        assert abs(result.p_sell - 0.2) < 1e-9
        assert abs(result.p_hold - 0.5) < 1e-9
        assert abs(result.p_buy - 0.3) < 1e-9

    def test_p_buy_p_hold_p_sell_sum_to_one(self) -> None:
        p = _probas(0.2, 0.5, 0.3)
        result = build_ml_prediction(raw_direction=0, probas=p, classes=[-1, 0, 1])
        total = (result.p_buy or 0.0) + (result.p_hold or 0.0) + (result.p_sell or 0.0)
        assert abs(total - 1.0) < 1e-6

    def test_confidence_margin_is_top1_minus_top2(self) -> None:
        p = np.array([0.1, 0.6, 0.3])
        result = build_ml_prediction(raw_direction=0, probas=p, classes=[-1, 0, 1])
        assert result.confidence_margin is not None
        expected = 0.6 - 0.3
        assert abs(result.confidence_margin - expected) < 1e-9

    def test_raw_direction_preserved(self) -> None:
        # raw_direction = -1 (SELL class, index 0); probas favour BUY but raw is SELL
        p = _probas(0.05, 0.2, 0.75)
        result = build_ml_prediction(raw_direction=-1, probas=p, classes=[-1, 0, 1])
        assert result.raw_direction == -1


# ---------------------------------------------------------------------------
# XGBoost remap scenario
# ---------------------------------------------------------------------------

class TestBuildMlPredictionXgboostRemap:
    def test_buy_remap(self) -> None:
        # XGBoost: classes [0,1,2] → domain [-1,0,1]
        inv_remap = {0: -1, 1: 0, 2: 1}
        # col 2 (class 2 → BUY) has highest prob
        p = np.array([0.1, 0.1, 0.8])
        result = build_ml_prediction(
            raw_direction=2,
            probas=p,
            classes=[0, 1, 2],
            inv_label_remap=inv_remap,
        )
        assert result.direction == 1   # BUY
        assert result.raw_direction == 2
        assert result.p_buy is not None and abs(result.p_buy - 0.8) < 1e-9
        assert result.p_sell is not None and abs(result.p_sell - 0.1) < 1e-9
        assert result.p_hold is not None and abs(result.p_hold - 0.1) < 1e-9

    def test_sell_remap(self) -> None:
        inv_remap = {0: -1, 1: 0, 2: 1}
        p = np.array([0.7, 0.2, 0.1])  # col 0 = SELL
        result = build_ml_prediction(
            raw_direction=0,
            probas=p,
            classes=[0, 1, 2],
            inv_label_remap=inv_remap,
        )
        assert result.direction == -1  # SELL
        assert result.p_sell is not None and abs(result.p_sell - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# Calibrated probas
# ---------------------------------------------------------------------------

class TestBuildMlPredictionCalibrated:
    def test_direction_from_calibrated_argmax(self) -> None:
        # Raw probas: HOLD wins
        raw_p = np.array([0.1, 0.6, 0.3])
        # Calibrated: BUY wins after calibration
        cal_p = np.array([0.05, 0.25, 0.70])
        result = build_ml_prediction(
            raw_direction=0,   # raw argmax → HOLD
            probas=raw_p,
            classes=[-1, 0, 1],
            calibrated_probas=cal_p,
        )
        assert result.direction == 1   # BUY from calibrated
        assert result.raw_direction == 0  # raw still preserved

    def test_confidence_from_calibrated(self) -> None:
        raw_p = np.array([0.1, 0.6, 0.3])
        cal_p = np.array([0.05, 0.25, 0.70])
        result = build_ml_prediction(
            raw_direction=0,
            probas=raw_p,
            classes=[-1, 0, 1],
            calibrated_probas=cal_p,
        )
        assert abs(result.confidence - 0.70) < 1e-9

    def test_p_buy_from_calibrated(self) -> None:
        raw_p = np.array([0.1, 0.6, 0.3])
        cal_p = np.array([0.05, 0.25, 0.70])
        result = build_ml_prediction(
            raw_direction=0,
            probas=raw_p,
            classes=[-1, 0, 1],
            calibrated_probas=cal_p,
        )
        assert result.p_buy is not None and abs(result.p_buy - 0.70) < 1e-9
        assert result.p_hold is not None and abs(result.p_hold - 0.25) < 1e-9
        assert result.p_sell is not None and abs(result.p_sell - 0.05) < 1e-9

    def test_calibrated_consistency_when_no_shift(self) -> None:
        # raw_direction must be a value present in classes (1 = BUY, at index 2)
        p = np.array([0.1, 0.1, 0.8])
        result_raw = build_ml_prediction(raw_direction=1, probas=p, classes=[-1, 0, 1])
        result_cal = build_ml_prediction(
            raw_direction=1, probas=p, classes=[-1, 0, 1], calibrated_probas=p
        )
        assert result_raw.direction == result_cal.direction
        assert abs(result_raw.confidence - result_cal.confidence) < 1e-9
