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
from dataclasses import dataclass, field
from typing import Any

from metatrade.core.log import get_logger
from metatrade.ml.config import MLConfig
from metatrade.ml.features import FeatureVector

log = get_logger(__name__)

_VALID_BACKENDS = {"histgbm", "lightgbm", "xgboost"}


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
            return lgb.LGBMClassifier(
                n_estimators=config.max_iter,
                max_depth=config.max_depth if config.max_depth else -1,
                num_leaves=config.num_leaves,
                learning_rate=config.learning_rate,
                min_child_samples=config.min_child_samples,
                random_state=config.random_seed,
                class_weight=config.class_weight,
                verbosity=-1,
            )
        except ImportError:
            log.warning("ml_backend_lightgbm_not_installed", fallback="histgbm")
            backend = "histgbm"

    if backend == "xgboost":
        try:
            import xgboost as xgb  # type: ignore[import]
            return xgb.XGBClassifier(
                n_estimators=config.max_iter,
                max_depth=config.max_depth if config.max_depth else 6,
                max_leaves=config.num_leaves,
                learning_rate=config.learning_rate,
                min_child_weight=config.min_child_samples,
                random_state=config.random_seed,
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0,
            )
        except ImportError:
            log.warning("ml_backend_xgboost_not_installed", fallback="histgbm")
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

        X = [fv.to_list() for fv in feature_vectors]
        y = labels

        model = _build_estimator(self._config, self._backend)
        log.info("ml_classifier_fit", backend=self._backend, n_samples=len(X))
        model.fit(X, y)

        # In-sample accuracy (training metric — not generalisation)
        preds = model.predict(X)
        accuracy = sum(p == lbl for p, lbl in zip(preds, y)) / n

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
        self._is_trained = accuracy >= self._config.min_accuracy
        return self._metrics

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

        X = [feature_vector.to_list()]
        probas = self._model.predict_proba(X)[0]
        classes = self._model.classes_

        # Confidence = probability of the predicted class
        max_idx = int(probas.argmax())
        direction = int(classes[max_idx])
        confidence = float(probas[max_idx])

        return direction, confidence

    def serialize(self) -> bytes:
        """Serialize the model to bytes (via pickle) for storage.

        Returns:
            Pickled bytes of the sklearn model.

        Raises:
            RuntimeError: If the model has not been trained.
        """
        if self._model is None:
            raise RuntimeError("No trained model to serialize.")
        return pickle.dumps(self._model)

    @classmethod
    def deserialize(cls, data: bytes, config: MLConfig | None = None) -> MLClassifier:
        """Reconstruct a classifier from serialized bytes.

        Args:
            data:   Pickled sklearn model bytes.
            config: Optional MLConfig (defaults to MLConfig()).

        Returns:
            Fitted MLClassifier instance.
        """
        instance = cls(config)
        instance._model = pickle.loads(data)  # noqa: S301
        instance._is_trained = True
        return instance
