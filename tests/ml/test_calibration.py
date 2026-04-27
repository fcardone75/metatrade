"""Tests for ml/calibration.py — ProbabilityCalibrator."""

from __future__ import annotations

import numpy as np
import pytest

from metatrade.ml.calibration import ProbabilityCalibrator, _MIN_CALIBRATION_SAMPLES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLASSES = [-1, 0, 1]  # SELL, HOLD, BUY — domain label order


def _make_probas(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(alpha=[1.0, 2.0, 1.0], size=n)
    return raw.astype(float)


def _make_labels(n: int, classes: list[int] = _CLASSES, seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed)
    return [classes[i] for i in rng.integers(0, len(classes), size=n)]


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestProbabilityCalibratorInit:
    def test_valid_isotonic(self) -> None:
        cal = ProbabilityCalibrator("isotonic")
        assert cal.method == "isotonic"
        assert not cal.is_fitted

    def test_valid_sigmoid(self) -> None:
        cal = ProbabilityCalibrator("sigmoid")
        assert cal.method == "sigmoid"

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            ProbabilityCalibrator("random_forest")

    def test_classes_empty_before_fit(self) -> None:
        assert ProbabilityCalibrator().classes_ == []


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestProbabilityCalibratorFit:
    def test_is_fitted_after_fit(self) -> None:
        cal = ProbabilityCalibrator()
        n = _MIN_CALIBRATION_SAMPLES
        cal.fit(_make_probas(n), _make_labels(n), _CLASSES)
        assert cal.is_fitted

    def test_classes_stored_after_fit(self) -> None:
        cal = ProbabilityCalibrator()
        n = _MIN_CALIBRATION_SAMPLES
        cal.fit(_make_probas(n), _make_labels(n), _CLASSES)
        assert cal.classes_ == _CLASSES

    def test_too_few_samples_raises(self) -> None:
        cal = ProbabilityCalibrator()
        n = _MIN_CALIBRATION_SAMPLES - 1
        with pytest.raises(ValueError, match="at least"):
            cal.fit(_make_probas(n), _make_labels(n), _CLASSES)

    def test_label_length_mismatch_raises(self) -> None:
        cal = ProbabilityCalibrator()
        probas = _make_probas(50)
        labels = _make_labels(40)
        with pytest.raises(ValueError, match="rows but labels"):
            cal.fit(probas, labels, _CLASSES)

    def test_column_count_mismatch_raises(self) -> None:
        cal = ProbabilityCalibrator()
        probas = _make_probas(50)  # 3 columns
        with pytest.raises(ValueError, match="columns"):
            cal.fit(probas, _make_labels(50), [-1, 0])  # 2 classes

    def test_sigmoid_method_fits(self) -> None:
        cal = ProbabilityCalibrator("sigmoid")
        n = _MIN_CALIBRATION_SAMPLES
        cal.fit(_make_probas(n), _make_labels(n), _CLASSES)
        assert cal.is_fitted


# ---------------------------------------------------------------------------
# calibrate()
# ---------------------------------------------------------------------------

class TestProbabilityCalibratorCalibrate:
    def _fitted(self, method: str = "isotonic", n: int = 100) -> ProbabilityCalibrator:
        cal = ProbabilityCalibrator(method)
        cal.fit(_make_probas(n), _make_labels(n), _CLASSES)
        return cal

    def test_output_shape(self) -> None:
        cal = self._fitted()
        probas = _make_probas(20, seed=99)
        out = cal.calibrate(probas)
        assert out.shape == probas.shape

    def test_rows_sum_to_one(self) -> None:
        cal = self._fitted()
        probas = _make_probas(50, seed=7)
        out = cal.calibrate(probas)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(50), atol=1e-6)

    def test_values_in_zero_one(self) -> None:
        cal = self._fitted()
        out = cal.calibrate(_make_probas(50))
        assert (out >= 0).all() and (out <= 1.0 + 1e-9).all()

    def test_sigmoid_rows_sum_to_one(self) -> None:
        cal = self._fitted("sigmoid")
        out = cal.calibrate(_make_probas(30, seed=3))
        np.testing.assert_allclose(out.sum(axis=1), np.ones(30), atol=1e-6)

    def test_not_fitted_raises(self) -> None:
        cal = ProbabilityCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.calibrate(_make_probas(5))


# ---------------------------------------------------------------------------
# ece()
# ---------------------------------------------------------------------------

class TestProbabilityCalibratorEce:
    def test_ece_in_range(self) -> None:
        cal = ProbabilityCalibrator()
        n = 200
        probas = _make_probas(n)
        labels = _make_labels(n)
        cal.fit(probas, labels, _CLASSES)
        ece = cal.ece(probas, labels)
        assert 0.0 <= ece <= 1.0

    def test_ece_float(self) -> None:
        cal = ProbabilityCalibrator()
        n = 100
        probas = _make_probas(n)
        labels = _make_labels(n)
        cal.fit(probas, labels, _CLASSES)
        assert isinstance(cal.ece(probas, labels), float)


# ---------------------------------------------------------------------------
# serialize / deserialize
# ---------------------------------------------------------------------------

class TestProbabilityCalibratorSerialization:
    def _fitted(self, n: int = 100) -> ProbabilityCalibrator:
        cal = ProbabilityCalibrator()
        cal.fit(_make_probas(n), _make_labels(n), _CLASSES)
        return cal

    def test_round_trip_is_fitted(self) -> None:
        cal = self._fitted()
        restored = ProbabilityCalibrator.deserialize(cal.serialize())
        assert restored.is_fitted

    def test_round_trip_method(self) -> None:
        cal = self._fitted()
        restored = ProbabilityCalibrator.deserialize(cal.serialize())
        assert restored.method == cal.method

    def test_round_trip_classes(self) -> None:
        cal = self._fitted()
        restored = ProbabilityCalibrator.deserialize(cal.serialize())
        assert restored.classes_ == cal.classes_

    def test_round_trip_calibrate_consistent(self) -> None:
        cal = self._fitted()
        probas = _make_probas(20, seed=55)
        restored = ProbabilityCalibrator.deserialize(cal.serialize())
        np.testing.assert_allclose(
            cal.calibrate(probas),
            restored.calibrate(probas),
            atol=1e-9,
        )

    def test_serialize_unfitted_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Cannot serialize"):
            ProbabilityCalibrator().serialize()
