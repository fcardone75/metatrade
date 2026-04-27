"""Tests for the new contracts API on MLClassifier: predict_raw(), feature_schema(), serialize() v2."""

from __future__ import annotations

import pytest

from metatrade.ml.classifier import MLClassifier
from metatrade.ml.contracts import MlFeatureSchema, MlPrediction
from metatrade.ml.features import MIN_FEATURE_BARS, FeatureVector

# Reuse helpers from the main ML test file
from tests.ml.test_ml import _test_cfg, make_labeled_dataset, _make_fv


# ---------------------------------------------------------------------------
# predict_raw()
# ---------------------------------------------------------------------------

class TestPredictRaw:
    def test_returns_ml_prediction(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        fv = _make_fv()
        result = clf.predict_raw(fv)
        assert isinstance(result, MlPrediction)

    def test_direction_in_domain(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        result = clf.predict_raw(_make_fv())
        assert result.direction in {1, -1, 0}

    def test_confidence_in_range(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        result = clf.predict_raw(_make_fv())
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_and_predict_raw_consistent(self) -> None:
        """predict() and predict_raw() must return the same direction and confidence."""
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        fv = _make_fv()
        direction_old, confidence_old = clf.predict(fv)
        result = clf.predict_raw(fv)
        assert result.direction == direction_old
        assert abs(result.confidence - confidence_old) < 1e-9

    def test_untrained_raises(self) -> None:
        clf = MLClassifier(_test_cfg())
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.predict_raw(_make_fv())


# ---------------------------------------------------------------------------
# feature_schema()
# ---------------------------------------------------------------------------

class TestFeatureSchema:
    def test_returns_schema_after_fit(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        schema = clf.feature_schema()
        assert isinstance(schema, MlFeatureSchema)

    def test_feature_count_matches_names(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        schema = clf.feature_schema()
        assert schema.feature_count == len(schema.feature_names)

    def test_feature_names_match_vector(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        schema = clf.feature_schema()
        assert list(schema.feature_names) == FeatureVector.feature_names()

    def test_min_bars_set(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        schema = clf.feature_schema()
        assert schema.min_bars == MIN_FEATURE_BARS

    def test_vector_class_name(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        schema = clf.feature_schema()
        assert schema.vector_class == "FeatureVector"

    def test_untrained_raises(self) -> None:
        clf = MLClassifier(_test_cfg())
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.feature_schema()


# ---------------------------------------------------------------------------
# serialize() / deserialize() — new format
# ---------------------------------------------------------------------------

class TestSerializeDeserialize:
    def test_round_trip_preserves_feature_names(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        data = clf.serialize()
        restored = MLClassifier.deserialize(data)
        assert restored._feature_names == clf._feature_names

    def test_round_trip_preserves_inv_label_remap(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        data = clf.serialize()
        restored = MLClassifier.deserialize(data)
        assert restored._inv_label_remap == clf._inv_label_remap

    def test_round_trip_predict_raw_consistent(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        fv = _make_fv()
        original = clf.predict_raw(fv)
        restored = MLClassifier.deserialize(clf.serialize())
        restored_pred = restored.predict_raw(fv)
        assert restored_pred.direction == original.direction
        assert abs(restored_pred.confidence - original.confidence) < 1e-9

    def test_legacy_format_still_loads(self) -> None:
        """A pickled raw sklearn model (old format) must still deserialize."""
        import pickle
        from sklearn.ensemble import HistGradientBoostingClassifier
        import pandas as pd
        import numpy as np

        # Simulate old-format serialize(): just the raw sklearn model
        model = HistGradientBoostingClassifier(max_iter=5, random_state=42)
        X = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
        y = [0, 1, -1] * 16 + [0, 1]
        model.fit(X, y)
        legacy_bytes = pickle.dumps(model)

        restored = MLClassifier.deserialize(legacy_bytes)
        assert restored.is_trained
        assert restored._feature_names == []
        assert restored._inv_label_remap == {}

    def test_is_trained_after_deserialize(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        restored = MLClassifier.deserialize(clf.serialize())
        assert restored.is_trained

    def test_feature_schema_available_after_deserialize(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        clf.fit(X, y)
        restored = MLClassifier.deserialize(clf.serialize())
        schema = restored.feature_schema()
        assert schema.feature_count == 35


# ---------------------------------------------------------------------------
# ClassifierMetrics.to_dict()
# ---------------------------------------------------------------------------

class TestClassifierMetricsToDict:
    def test_to_dict_keys(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        metrics = clf.fit(X, y)
        d = metrics.to_dict()
        assert "accuracy" in d
        assert "n_samples" in d
        assert "feature_importances" in d
        assert "class_distribution" in d

    def test_to_dict_json_serializable(self) -> None:
        import json
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(200)
        metrics = clf.fit(X, y)
        # Should not raise
        json.dumps(metrics.to_dict())
