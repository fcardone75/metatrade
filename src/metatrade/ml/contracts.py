"""ML domain contracts — typed data transfer objects for the ML pipeline.

These dataclasses define the boundaries between labeling, training,
calibration, inference and persistence. They carry no business logic
and have no dependencies on other metatrade modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MlPrediction:
    """Output of the classifier for a single observation.

    Args:
        direction:         Final domain label: 1=BUY, -1=SELL, 0=HOLD.
                           Derived from calibrated probas when a calibrator is
                           active; otherwise from raw model output.
        confidence:        Max class probability ∈ [0.0, 1.0].  Calibrated
                           when a calibrator is active.
        raw_direction:     Raw class value from the model before inv_label_remap.
                           Equal to ``direction`` for non-XGBoost / uncalibrated
                           inference.
        p_buy:             Probability assigned to BUY (domain label 1).
                           ``None`` when the classifier does not expose per-class
                           probabilities (should not happen in practice).
        p_hold:            Probability assigned to HOLD (domain label 0).
        p_sell:            Probability assigned to SELL (domain label -1).
        confidence_margin: Difference between top-1 and top-2 class probabilities.
                           Large margin → high certainty between the top two classes.
                           ``None`` when fewer than 2 classes exist.
        metadata:          Optional extra fields (model version, feature count, …).
    """

    direction: int
    confidence: float
    raw_direction: int
    p_buy: float | None = None
    p_hold: float | None = None
    p_sell: float | None = None
    confidence_margin: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MlFeatureSchema:
    """Description of the feature vector used for training and inference.

    Captures the exact contract between feature extraction and the model
    so that mismatches (e.g. different FeatureVector subclass or version)
    can be detected at load time.

    Args:
        feature_names: Ordered tuple of feature column names.
        feature_count: Must equal ``len(feature_names)``.
        min_bars:      Minimum bar window required by the feature extractor.
        vector_class:  Fully-qualified or short class name of the FeatureVector
                       used (e.g. ``"FeatureVector"``).
    """

    feature_names: tuple[str, ...]
    feature_count: int
    min_bars: int
    vector_class: str

    def __post_init__(self) -> None:
        if self.feature_count != len(self.feature_names):
            raise ValueError(
                f"feature_count ({self.feature_count}) does not match "
                f"len(feature_names) ({len(self.feature_names)})"
            )


@dataclass(frozen=True)
class MlModelArtifacts:
    """Bundle of everything needed to load and use a trained model.

    Combines the serialized model bytes with all metadata required to
    reconstruct a fully operational classifier without re-training.

    Note: instances are not hashable because ``metrics`` and ``tags`` are
    dicts. Use as a value object only (not as a dict key or set member).

    Args:
        version:        Version string, e.g. ``"v20240601_1430"``.
        symbol:         Symbol the model was trained for.
        created_at:     Unix timestamp of model creation.
        feature_schema: Schema of the feature vector used for training.
        model_bytes:       Pickled model bytes produced by ``MLClassifier.serialize()``.
        metrics:           Training metrics dict (accuracy, n_samples, …).
        calibrator_bytes:  Serialized ``ProbabilityCalibrator`` bytes, or ``None``
                           when no calibrator has been fitted for this model.
        ev_model_bytes:    Serialized ``ExpectedValueTrainer`` bytes, or ``None``
                           when no EV model has been trained for this model.
        tags:              Free-form metadata (backend, holdout_accuracy, …).
    """

    version: str
    symbol: str
    created_at: float
    feature_schema: MlFeatureSchema
    model_bytes: bytes
    metrics: dict[str, object]
    calibrator_bytes: bytes | None = None
    ev_model_bytes: bytes | None = None
    tags: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class MlEvaluationReport:
    """Metrics snapshot for a single fold or holdout evaluation.

    Args:
        fold_id:            0-based fold index (use 0 for holdout).
        n_samples:          Number of samples evaluated.
        accuracy:           Overall accuracy ∈ [0, 1].
        buy_precision:      Precision for class BUY ∈ [0, 1].
        sell_precision:     Precision for class SELL ∈ [0, 1].
        hold_precision:     Precision for class HOLD ∈ [0, 1].
        buy_recall:         Recall for class BUY ∈ [0, 1].
        sell_recall:        Recall for class SELL ∈ [0, 1].
        class_distribution: Map from label (-1, 0, 1) to sample count.
    """

    fold_id: int
    n_samples: int
    accuracy: float
    buy_precision: float
    sell_precision: float
    hold_precision: float
    buy_recall: float
    sell_recall: float
    class_distribution: dict[int, int]

    def __post_init__(self) -> None:
        for name, val in [
            ("accuracy", self.accuracy),
            ("buy_precision", self.buy_precision),
            ("sell_precision", self.sell_precision),
            ("hold_precision", self.hold_precision),
            ("buy_recall", self.buy_recall),
            ("sell_recall", self.sell_recall),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {val}")
