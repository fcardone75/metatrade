"""ML classifier for trading signal generation.

Pluggable backend — selected via ``MLConfig.backend`` or the ``backend``
constructor argument (CLI ``--backend`` takes precedence over env):

    histgbm   HistGradientBoostingClassifier (scikit-learn) — default, always available
    lightgbm  LGBMClassifier (lightgbm) — faster, leaf-wise growth, better on tabular data
    xgboost   XGBClassifier (xgboost) — robust alternative

All backends expose the same sklearn-compatible API.  If the requested backend
is not installed, a warning is logged and histgbm is used as fallback.

The classifier outputs a 3-class prediction:
    1  = BUY
   -1  = SELL
    0  = HOLD

Replaces the model by calling ``fit()`` — thread-safe because it reassigns
the internal reference atomically.
"""

from __future__ import annotations

import pickle
import subprocess
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from metatrade.core.log import get_logger
from metatrade.ml.calibration import ProbabilityCalibrator
from metatrade.ml.config import MLConfig
from metatrade.ml.contracts import MlFeatureSchema, MlPrediction
from metatrade.ml.features import MIN_FEATURE_BARS, FeatureVector
from metatrade.ml.prediction import build_ml_prediction

log = get_logger(__name__)

_VALID_BACKENDS = {"histgbm", "lightgbm", "xgboost", "catboost"}
# Minimum free GPU memory (MB) required to enable catboost GPU mode.
# Below this threshold we silently fall back to CPU to avoid a hard OOM abort.
_MIN_CATBOOST_GPU_MB = 2048

# Backends that handle class_weight internally via constructor param.
# For all others, balanced sample_weight is computed and passed to fit().
# CatBoost uses auto_class_weights="Balanced" in its constructor, so it must
# be listed here to avoid double-weighting (constructor + sample_weight in fit).
_NATIVE_CLASS_WEIGHT_BACKENDS = {"histgbm", "lightgbm", "catboost"}


def _free_gpu_mb() -> int | None:
    """Return free GPU memory in MB for GPU 0 via nvidia-smi, or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def _balanced_sample_weights(labels: list[int]) -> list[float]:
    """Compute per-sample weights for balanced class distribution."""
    counts: dict[int, int] = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    n = len(labels)
    n_classes = len(counts)
    weight_for: dict[int, float] = {
        cls: n / (n_classes * cnt) for cls, cnt in counts.items()
    }
    return [weight_for[lbl] for lbl in labels]


def _build_estimator(config: MLConfig, backend: str) -> Any:
    """Instantiate the sklearn-compatible estimator for the given backend.

    Falls back to histgbm with a warning if the library is not installed.
    """
    backend = backend.lower()
    if backend not in _VALID_BACKENDS:
        log.warning("ml_backend_unknown", backend=backend, fallback="histgbm")
        backend = "histgbm"

    if backend == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore[import]
            lgb_kwargs: dict[str, Any] = dict(
                n_estimators=config.max_iter,
                max_depth=config.max_depth if config.max_depth else -1,
                num_leaves=config.num_leaves,
                learning_rate=config.learning_rate,
                min_child_samples=config.min_child_samples,
                random_state=config.random_seed,
                class_weight=config.class_weight,
                verbosity=-1,
            )
            if config.use_gpu:
                lgb_kwargs["device"] = "gpu"
                log.info("ml_gpu_enabled", backend="lightgbm")
            return lgb.LGBMClassifier(**lgb_kwargs)
        except ImportError:
            log.warning("ml_backend_lightgbm_not_installed", fallback="histgbm")
            backend = "histgbm"

    if backend == "xgboost":
        try:
            import xgboost as xgb  # type: ignore[import]
            xgb_kwargs: dict[str, Any] = dict(
                n_estimators=config.max_iter,
                max_depth=config.max_depth if config.max_depth else 6,
                max_leaves=config.num_leaves,
                learning_rate=config.learning_rate,
                min_child_weight=config.min_child_samples,
                random_state=config.random_seed,
                eval_metric="mlogloss",
                verbosity=0,
            )
            if config.use_gpu:
                xgb_kwargs["tree_method"] = "hist"
                xgb_kwargs["device"] = "cuda"
                log.info("ml_gpu_enabled", backend="xgboost")
            return xgb.XGBClassifier(**xgb_kwargs)
        except ImportError:
            log.warning("ml_backend_xgboost_not_installed", fallback="histgbm")
            backend = "histgbm"

    if backend == "catboost":
        try:
            from catboost import CatBoostClassifier  # type: ignore[import]
            cb_kwargs: dict[str, Any] = dict(
                iterations=config.max_iter,
                depth=config.max_depth if config.max_depth else 6,
                learning_rate=config.learning_rate,
                min_data_in_leaf=config.min_child_samples,
                random_seed=config.random_seed,
                auto_class_weights="Balanced" if config.class_weight == "balanced" else None,
                verbose=0,
            )
            if config.use_gpu:
                free_mb = _free_gpu_mb()
                if free_mb is not None and free_mb < _MIN_CATBOOST_GPU_MB:
                    log.warning(
                        "ml_catboost_gpu_skipped_low_memory",
                        free_gpu_mb=free_mb,
                        min_required_mb=_MIN_CATBOOST_GPU_MB,
                    )
                else:
                    cb_kwargs["task_type"] = "GPU"
                    log.info("ml_gpu_enabled", backend="catboost", free_gpu_mb=free_mb)
            return CatBoostClassifier(**cb_kwargs)
        except ImportError:
            log.warning("ml_backend_catboost_not_installed", fallback="histgbm")
            backend = "histgbm"

    # histgbm (default / fallback)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_samples_leaf=config.min_child_samples,
        random_state=config.random_seed,
        class_weight=config.class_weight,
    )


@dataclass
class ClassifierMetrics:
    """Training/evaluation metrics for a fitted classifier."""

    accuracy: float
    n_samples: int
    feature_importances: dict[str, float] = field(default_factory=dict)
    class_distribution: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-safe dict for persistence in MlModelArtifacts."""
        return {
            "accuracy": self.accuracy,
            "n_samples": self.n_samples,
            "feature_importances": self.feature_importances,
            "class_distribution": {str(k): v for k, v in self.class_distribution.items()},
        }


class MLClassifier:
    """Pluggable gradient-boosting classifier for signal generation.

    Args:
        config:  MLConfig controlling hyper-parameters and backend default.
        backend: Override the backend for this instance.  Accepted values:
                 ``"histgbm"`` | ``"lightgbm"`` | ``"xgboost"``.
                 When None (default), uses ``config.backend``.
                 CLI ``--backend`` should pass its value here so it takes
                 precedence over the env-level ``ML_BACKEND``.
    """

    def __init__(self, config: MLConfig | None = None, backend: str | None = None) -> None:
        self._config = config or MLConfig()
        self._backend = (backend or self._config.backend).lower()
        self._model: Any = None  # sklearn estimator or None
        self._metrics: ClassifierMetrics | None = None
        self._is_trained = False
        self._feature_names: list[str] = []
        self._vector_class: str = "FeatureVector"
        # XGBoost requires non-negative integer labels; we remap [-1,0,1] → [0,1,2]
        # and store the inverse so predict() can restore domain labels.
        self._inv_label_remap: dict[int, int] = {}
        self._calibrator: ProbabilityCalibrator | None = None

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def metrics(self) -> ClassifierMetrics | None:
        return self._metrics

    def fit(
        self,
        feature_vectors: list[FeatureVector],
        labels: list[int],
    ) -> ClassifierMetrics:
        """Train the classifier on the given data.

        Args:
            feature_vectors: List of FeatureVectors (training features).
            labels:          Corresponding labels (1, -1, or 0).

        Returns:
            ClassifierMetrics from in-sample evaluation.

        Raises:
            ValueError: If samples < min_train_samples or features/labels mismatch.
            ImportError: If scikit-learn is not installed.
        """
        if len(feature_vectors) != len(labels):
            raise ValueError(
                f"feature_vectors ({len(feature_vectors)}) and "
                f"labels ({len(labels)}) must have equal length"
            )
        n = len(feature_vectors)
        if n < self._config.min_train_samples:
            raise ValueError(
                f"Need at least {self._config.min_train_samples} samples, got {n}"
            )

        import pandas as pd

        fv_cls = type(feature_vectors[0]) if feature_vectors else FeatureVector
        self._feature_names = fv_cls.feature_names()
        self._vector_class = fv_cls.__name__
        X = pd.DataFrame(
            [fv.to_list() for fv in feature_vectors],
            columns=self._feature_names,
        )
        y = labels

        # XGBoost rejects negative class labels; remap domain labels to [0, N-1].
        if self._backend == "xgboost":
            unique = sorted(set(labels))
            label_to_idx = {lbl: i for i, lbl in enumerate(unique)}
            self._inv_label_remap = {i: lbl for i, lbl in enumerate(unique)}
            y = [label_to_idx[lbl] for lbl in labels]
        else:
            self._inv_label_remap = {}

        model = _build_estimator(self._config, self._backend)
        log.info("ml_classifier_fit", backend=self._backend, n_samples=len(X))
        use_sample_weight = (
            self._config.class_weight == "balanced"
            and self._backend not in _NATIVE_CLASS_WEIGHT_BACKENDS
        )
        if use_sample_weight:
            sw = _balanced_sample_weights(labels)
            model.fit(X, y, sample_weight=sw)
        else:
            sw = None
            model.fit(X, y)

        preds = model.predict(X)
        # Raw accuracy is used for _is_trained: min_accuracy is calibrated on raw
        # accuracy, and balanced accuracy can be artificially low on HOLD-dominated
        # M1/M5 datasets even when the model genuinely learns BUY/SELL patterns.
        # CatBoost/XGBoost predict() can return numpy scalars that accumulate into
        # an ndarray inside Python's sum(); cast to float to ensure a plain scalar.
        preds_flat = preds.ravel() if hasattr(preds, "ravel") else preds
        raw_accuracy = float(sum(p == lbl for p, lbl in zip(preds_flat, y)) / n)

        if use_sample_weight:
            # Balanced accuracy: mean per-class recall (reported separately for
            # insight into BUY/SELL learning, but NOT used for is_trained gate).
            counts: dict[int, int] = {}
            correct_by_class: dict[int, int] = {}
            for p, lbl in zip(preds_flat, y):
                counts[lbl] = counts.get(lbl, 0) + 1
                if p == lbl:
                    correct_by_class[lbl] = correct_by_class.get(lbl, 0) + 1
            accuracy = sum(
                correct_by_class.get(c, 0) / cnt for c, cnt in counts.items()
            ) / len(counts)
        else:
            accuracy = raw_accuracy

        # HistGBM exposes feature_importances_ via permutation importance
        # or the built-in property (available in sklearn >= 1.2).
        # Use it when available; otherwise record zeros.
        # Resolve names from the actual vector type (supports MultiResFeatureVector).
        fv_cls = type(feature_vectors[0]) if feature_vectors else FeatureVector
        feature_names = fv_cls.feature_names()
        try:
            importances = {
                name: float(imp)
                for name, imp in zip(feature_names, model.feature_importances_)
            }
        except AttributeError:
            importances = {name: 0.0 for name in feature_names}

        # Class distribution
        class_dist: dict[int, int] = {}
        for lbl in y:
            class_dist[lbl] = class_dist.get(lbl, 0) + 1

        self._model = model
        self._metrics = ClassifierMetrics(
            accuracy=accuracy,
            n_samples=n,
            feature_importances=importances,
            class_distribution=class_dist,
        )
        self._is_trained = raw_accuracy >= self._config.min_accuracy
        log.info(
            "ml_classifier_fit_done",
            backend=self._backend,
            in_sample_accuracy=round(raw_accuracy, 4),
            balanced_accuracy=round(accuracy, 4) if use_sample_weight else None,
            is_trained=self._is_trained,
            min_accuracy=self._config.min_accuracy,
        )
        return self._metrics

    def predict_raw(self, feature_vector: FeatureVector) -> MlPrediction:
        """Predict signal direction and confidence, returning a typed MlPrediction.

        When a fitted ``ProbabilityCalibrator`` is attached (via ``calibrate()``),
        the returned ``direction``, ``confidence``, and per-class probabilities
        (``p_buy`` / ``p_hold`` / ``p_sell``) reflect the calibrated output.
        ``raw_direction`` always reflects the uncalibrated argmax.

        Args:
            feature_vector: Extracted features for the current bar.

        Returns:
            ``MlPrediction`` with direction, confidence, raw_direction,
            per-class probabilities and confidence_margin.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._model is None or not self._is_trained:
            raise RuntimeError(
                "MLClassifier has not been trained. Call fit() first."
            )

        import pandas as pd

        X = pd.DataFrame([feature_vector.to_list()], columns=self._feature_names or None)
        probas: np.ndarray = self._model.predict_proba(X)[0]
        raw_classes = [int(c) for c in self._model.classes_]
        raw_direction = raw_classes[int(probas.argmax())]

        calibrated_probas: np.ndarray | None = None
        if self._calibrator is not None and self._calibrator.is_fitted:
            calibrated_probas = self._calibrator.calibrate(probas.reshape(1, -1))[0]

        return build_ml_prediction(
            raw_direction=raw_direction,
            probas=probas,
            classes=raw_classes,
            inv_label_remap=self._inv_label_remap,
            calibrated_probas=calibrated_probas,
        )

    def feature_schema(self) -> MlFeatureSchema:
        """Return the feature schema recorded during the last ``fit()`` call.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if not self._is_trained or not self._feature_names:
            raise RuntimeError(
                "MLClassifier has not been trained. Call fit() first."
            )
        return MlFeatureSchema(
            feature_names=tuple(self._feature_names),
            feature_count=len(self._feature_names),
            min_bars=MIN_FEATURE_BARS,
            vector_class=self._vector_class,
        )

    def calibrate(
        self,
        holdout_vectors: list[FeatureVector],
        holdout_labels: list[int],
        method: str = "isotonic",
    ) -> ProbabilityCalibrator:
        """Fit a post-hoc probability calibrator on a holdout set.

        The calibrator adjusts raw ``predict_proba`` outputs so that the
        stated confidence better matches empirical accuracy (lower ECE).
        After this call, ``predict_raw()`` automatically uses the calibrated
        probabilities.

        Args:
            holdout_vectors: Feature vectors from a held-out split (not used
                             in training — using training data causes leakage).
            holdout_labels:  True domain labels for each holdout vector.
            method:          Calibration algorithm — ``"isotonic"`` or ``"sigmoid"``.

        Returns:
            The fitted ``ProbabilityCalibrator`` instance (also stored
            internally so ``predict_raw()`` picks it up automatically).

        Raises:
            RuntimeError: If the classifier has not been trained yet.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError(
                "MLClassifier has not been trained. Call fit() first."
            )

        import pandas as pd

        X = pd.DataFrame(
            [fv.to_list() for fv in holdout_vectors],
            columns=self._feature_names or None,
        )
        probas: np.ndarray = self._model.predict_proba(X)
        raw_classes = [int(c) for c in self._model.classes_]

        # Map domain labels back to raw class space (required for XGBoost remap)
        if self._inv_label_remap:
            domain_to_raw = {v: k for k, v in self._inv_label_remap.items()}
            raw_labels = [domain_to_raw.get(lbl, lbl) for lbl in holdout_labels]
        else:
            raw_labels = list(holdout_labels)

        calibrator = ProbabilityCalibrator(method=method)
        calibrator.fit(probas, raw_labels, raw_classes)
        self._calibrator = calibrator

        ece = calibrator.ece(probas, raw_labels)
        log.info(
            "classifier_calibrated",
            method=method,
            n_samples=len(holdout_vectors),
            holdout_ece=round(ece, 4),
        )
        return calibrator

    def serialize_calibrator(self) -> bytes | None:
        """Return serialized calibrator bytes, or ``None`` if not fitted."""
        if self._calibrator is None or not self._calibrator.is_fitted:
            return None
        return self._calibrator.serialize()

    def predict(self, feature_vector: FeatureVector) -> tuple[int, float]:
        """Predict signal direction and confidence for a single feature vector.

        Args:
            feature_vector: Extracted features for the current bar.

        Returns:
            (direction, confidence) where:
                direction  ∈ {1, -1, 0}
                confidence ∈ [0.0, 1.0] (max class probability)

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._model is None or not self._is_trained:
            raise RuntimeError(
                "MLClassifier has not been trained. Call fit() first."
            )

        import pandas as pd

        X = pd.DataFrame([feature_vector.to_list()], columns=self._feature_names or None)
        probas = self._model.predict_proba(X)[0]
        classes = self._model.classes_

        # Confidence = probability of the predicted class
        max_idx = int(probas.argmax())
        predicted = int(classes[max_idx])
        direction = self._inv_label_remap.get(predicted, predicted)
        confidence = float(probas[max_idx])

        return direction, confidence

    def serialize(self) -> bytes:
        """Serialize the model and metadata to bytes for storage.

        The payload is a dict containing the raw sklearn model bytes plus
        the feature names and label remap needed for correct inference.
        Legacy callers that expect raw model bytes are handled in ``deserialize()``.

        Returns:
            Pickled dict with keys: ``model``, ``feature_names``,
            ``inv_label_remap``, ``vector_class``.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if self._model is None:
            raise RuntimeError("No trained model to serialize.")
        payload = {
            "model": pickle.dumps(self._model),
            "feature_names": self._feature_names,
            "inv_label_remap": self._inv_label_remap,
            "vector_class": self._vector_class,
        }
        return pickle.dumps(payload)

    @classmethod
    def deserialize(cls, data: bytes, config: MLConfig | None = None) -> MLClassifier:
        """Reconstruct a classifier from serialized bytes.

        Supports both the new dict-payload format (produced by the current
        ``serialize()``) and the legacy format where ``data`` is a direct
        pickle of the sklearn estimator.

        Args:
            data:   Bytes produced by ``serialize()``.
            config: Optional MLConfig (defaults to MLConfig()).

        Returns:
            Fitted MLClassifier instance with feature names restored.
        """
        instance = cls(config)
        payload = pickle.loads(data)  # noqa: S301
        if isinstance(payload, dict) and "model" in payload:
            # New format: dict with metadata
            instance._model = pickle.loads(payload["model"])  # noqa: S301
            instance._feature_names = list(payload.get("feature_names") or [])
            instance._inv_label_remap = {
                int(k): int(v)
                for k, v in (payload.get("inv_label_remap") or {}).items()
            }
            instance._vector_class = str(payload.get("vector_class") or "FeatureVector")
        else:
            # Legacy format: payload is the sklearn estimator directly
            instance._model = payload
            instance._feature_names = []
            instance._inv_label_remap = {}
            instance._vector_class = "FeatureVector"
        instance._is_trained = True
        return instance
