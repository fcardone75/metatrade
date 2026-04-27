"""Post-hoc probability calibrator for the ML classifier.

Fits one isotonic (or sigmoid) regressor per class using a one-vs-rest
approach on a separate holdout set.  The calibrator is trained on raw
model probabilities and adjusts them so that the stated confidence better
reflects the empirical accuracy (lower Expected Calibration Error).

Calibration does NOT change the training set — it requires a separate
held-out split to avoid leakage.  ECE is the primary quality metric:
target ECE < 0.05 on the same holdout.
"""

from __future__ import annotations

import pickle

import numpy as np

from metatrade.core.log import get_logger

log = get_logger(__name__)

_VALID_METHODS = ("isotonic", "sigmoid")
_MIN_CALIBRATION_SAMPLES = 30


class ProbabilityCalibrator:
    """Post-hoc calibrator for a 3-class trading signal classifier.

    Fits one regressor per class (one-vs-rest) on a held-out set of
    raw model probabilities and true labels.  After fitting, ``calibrate()``
    maps raw probas to calibrated ones and renormalises each row to sum to 1.

    Args:
        method: Calibration algorithm — ``"isotonic"`` (default, non-parametric,
                needs ≥ 30 samples) or ``"sigmoid"`` (Platt scaling, parametric).
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {_VALID_METHODS}, got {method!r}"
            )
        self._method = method
        self._calibrators: dict[int, object] = {}  # raw_class → fitted regressor
        self._classes: list[int] = []
        self._is_fitted = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def method(self) -> str:
        return self._method

    @property
    def classes_(self) -> list[int]:
        """Raw model class labels in the order used during fit."""
        return list(self._classes)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        probas: np.ndarray,
        labels: list[int],
        classes: list[int],
    ) -> None:
        """Fit calibrators on holdout class probabilities and true labels.

        Args:
            probas:  Raw class probabilities, shape ``(n_samples, len(classes))``.
            labels:  True class labels for each sample; elements must be in
                     ``classes``.
            classes: Ordered list of class values matching the columns of
                     ``probas`` (e.g. ``[-1, 0, 1]`` or ``[0, 1, 2]`` for XGBoost).

        Raises:
            ValueError: If probas / labels lengths mismatch, column count
                        doesn't match classes, or fewer than
                        ``_MIN_CALIBRATION_SAMPLES`` samples are provided.
        """
        n = probas.shape[0]
        if n != len(labels):
            raise ValueError(
                f"probas has {n} rows but labels has {len(labels)} elements"
            )
        if probas.shape[1] != len(classes):
            raise ValueError(
                f"probas has {probas.shape[1]} columns but len(classes)={len(classes)}"
            )
        if n < _MIN_CALIBRATION_SAMPLES:
            raise ValueError(
                f"Need at least {_MIN_CALIBRATION_SAMPLES} samples for calibration, got {n}"
            )

        self._classes = list(classes)
        calibrators: dict[int, object] = {}

        for i, cls in enumerate(classes):
            y_binary = np.array([1.0 if lbl == cls else 0.0 for lbl in labels])
            p_col = probas[:, i].astype(float)

            if self._method == "isotonic":
                from sklearn.isotonic import IsotonicRegression  # type: ignore[import]
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(p_col, y_binary)
            else:
                from sklearn.linear_model import LogisticRegression  # type: ignore[import]
                cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
                cal.fit(p_col.reshape(-1, 1), y_binary)

            calibrators[cls] = cal

        self._calibrators = calibrators
        self._is_fitted = True
        log.info(
            "calibrator_fitted",
            method=self._method,
            n_samples=n,
            classes=classes,
        )

    # ── Calibrate ─────────────────────────────────────────────────────────────

    def calibrate(self, probas: np.ndarray) -> np.ndarray:
        """Return calibrated and renormalised probabilities.

        Args:
            probas: Raw class probabilities, shape ``(n_samples, len(classes_))``.

        Returns:
            Calibrated probabilities of the same shape; each row sums to 1.

        Raises:
            RuntimeError: If the calibrator has not been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ProbabilityCalibrator has not been fitted. Call fit() first."
            )

        calibrated = np.empty_like(probas, dtype=float)
        for i, cls in enumerate(self._classes):
            cal = self._calibrators[cls]
            p_col = probas[:, i].astype(float)
            if self._method == "isotonic":
                from sklearn.isotonic import IsotonicRegression  # type: ignore[import]
                assert isinstance(cal, IsotonicRegression)
                calibrated[:, i] = cal.predict(p_col)
            else:
                from sklearn.linear_model import LogisticRegression  # type: ignore[import]
                assert isinstance(cal, LogisticRegression)
                calibrated[:, i] = cal.predict_proba(p_col.reshape(-1, 1))[:, 1]

        # Renormalise rows so they sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return calibrated / row_sums

    # ── ECE ───────────────────────────────────────────────────────────────────

    def ece(
        self,
        probas: np.ndarray,
        labels: list[int],
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE) after calibration.

        Uses the maximum-confidence class for binning (multi-class ECE as in
        Guo et al., 2017).  A lower value means better calibration; target < 0.05.

        Args:
            probas:  Raw class probabilities, shape ``(n, len(classes_))``.
            labels:  True domain labels for each sample.
            n_bins:  Number of equal-width bins in [0, 1].

        Returns:
            ECE ∈ [0, 1].
        """
        cal = self.calibrate(probas)
        confidences = cal.max(axis=1)
        pred_idx = cal.argmax(axis=1)
        predicted = np.array([self._classes[i] for i in pred_idx])
        correct = (predicted == np.array(labels)).astype(float)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(labels)
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (confidences > lo) & (confidences <= hi)
            count = mask.sum()
            if count == 0:
                continue
            acc = correct[mask].mean()
            avg_conf = confidences[mask].mean()
            ece += (count / n) * abs(acc - avg_conf)

        return float(ece)

    # ── Serialization ─────────────────────────────────────────────────────────

    def serialize(self) -> bytes:
        """Serialize the fitted calibrator to bytes for persistence."""
        if not self._is_fitted:
            raise RuntimeError(
                "Cannot serialize an unfitted ProbabilityCalibrator."
            )
        payload = {
            "method": self._method,
            "classes": self._classes,
            "calibrators": self._calibrators,
            "is_fitted": self._is_fitted,
        }
        return pickle.dumps(payload)

    @classmethod
    def deserialize(cls, data: bytes) -> ProbabilityCalibrator:
        """Reconstruct a calibrator from serialized bytes.

        Args:
            data: Bytes produced by ``serialize()``.

        Returns:
            Fitted ``ProbabilityCalibrator`` instance.
        """
        payload: dict[str, object] = pickle.loads(data)  # noqa: S301
        method = str(payload.get("method") or "isotonic")
        instance = cls(method=method)
        classes_raw = payload.get("classes")
        instance._classes = list(classes_raw) if isinstance(classes_raw, list) else []
        cals = payload.get("calibrators")
        instance._calibrators = dict(cals) if isinstance(cals, dict) else {}
        instance._is_fitted = bool(payload.get("is_fitted", False))
        return instance
