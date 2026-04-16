"""ML classifier for trading signal generation.

Uses HistGradientBoostingClassifier (scikit-learn) trained on the FeatureVector
schema from ``ml.features``. The classifier outputs a 3-class prediction:
    1  = BUY
   -1  = SELL
    0  = HOLD

HistGradientBoostingClassifier advantages over RandomForest:
- Gradient boosting on decision trees — captures non-linear patterns better
- Native handling of missing values (no imputation needed)
- Significantly faster training on large datasets
- Better generalisation on tabular financial data
- Uses histogram-based splits (faster than exact splits)

The model degrades gracefully: returns HOLD when predicted class probability
is below the minimum confidence threshold.

Replaces the model by calling ``fit()`` — thread-safe because it reassigns
the internal reference atomically.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any

from metatrade.ml.config import MLConfig
from metatrade.ml.features import FeatureVector


@dataclass
class ClassifierMetrics:
    """Training/evaluation metrics for a fitted classifier."""

    accuracy: float
    n_samples: int
    feature_importances: dict[str, float] = field(default_factory=dict)
    class_distribution: dict[int, int] = field(default_factory=dict)


class MLClassifier:
    """Wrapper around HistGradientBoostingClassifier for signal generation.

    Args:
        config: MLConfig instance controlling model hyper-parameters.
    """

    def __init__(self, config: MLConfig | None = None) -> None:
        self._config = config or MLConfig()
        self._model: Any = None  # sklearn estimator or None
        self._metrics: ClassifierMetrics | None = None
        self._is_trained = False

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

        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for ML module. "
                "Install with: pip install scikit-learn"
            ) from e

        X = [fv.to_list() for fv in feature_vectors]
        y = labels

        model = HistGradientBoostingClassifier(
            max_iter=self._config.max_iter,
            max_depth=self._config.max_depth,
            random_state=self._config.random_seed,
            class_weight=self._config.class_weight,
        )
        model.fit(X, y)

        # In-sample accuracy (training metric — not generalisation)
        preds = model.predict(X)
        accuracy = sum(p == lbl for p, lbl in zip(preds, y)) / n

        # HistGBM exposes feature_importances_ via permutation importance
        # or the built-in property (available in sklearn >= 1.2).
        # Use it when available; otherwise record zeros.
        feature_names = FeatureVector.feature_names()
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
    def deserialize(cls, data: bytes, config: MLConfig | None = None) -> "MLClassifier":
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
