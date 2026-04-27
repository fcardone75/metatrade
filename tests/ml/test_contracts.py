"""Tests for ml/contracts.py — MlPrediction, MlFeatureSchema, MlModelArtifacts, MlEvaluationReport."""

from __future__ import annotations

import dataclasses

import pytest

from metatrade.ml.contracts import (
    MlEvaluationReport,
    MlFeatureSchema,
    MlModelArtifacts,
    MlPrediction,
)


# ---------------------------------------------------------------------------
# MlPrediction
# ---------------------------------------------------------------------------

class TestMlPrediction:
    def test_fields(self) -> None:
        p = MlPrediction(direction=1, confidence=0.72, raw_direction=2)
        assert p.direction == 1
        assert p.confidence == 0.72
        assert p.raw_direction == 2
        assert p.metadata == {}

    def test_frozen(self) -> None:
        p = MlPrediction(direction=1, confidence=0.72, raw_direction=2)
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            p.direction = -1  # type: ignore[misc]

    def test_metadata_default_independent(self) -> None:
        p1 = MlPrediction(direction=0, confidence=0.5, raw_direction=0)
        p2 = MlPrediction(direction=0, confidence=0.5, raw_direction=0)
        # Default factory produces separate dict instances
        assert p1.metadata is not p2.metadata


# ---------------------------------------------------------------------------
# MlFeatureSchema
# ---------------------------------------------------------------------------

NAMES_35 = tuple(f"f{i}" for i in range(35))


class TestMlFeatureSchema:
    def test_valid(self) -> None:
        schema = MlFeatureSchema(
            feature_names=NAMES_35,
            feature_count=35,
            min_bars=70,
            vector_class="FeatureVector",
        )
        assert schema.feature_count == 35
        assert len(schema.feature_names) == 35

    def test_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="feature_count"):
            MlFeatureSchema(
                feature_names=NAMES_35,
                feature_count=10,  # wrong
                min_bars=70,
                vector_class="FeatureVector",
            )

    def test_frozen(self) -> None:
        schema = MlFeatureSchema(
            feature_names=NAMES_35,
            feature_count=35,
            min_bars=70,
            vector_class="FeatureVector",
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            schema.feature_count = 99  # type: ignore[misc]

    def test_hashable(self) -> None:
        schema = MlFeatureSchema(
            feature_names=NAMES_35,
            feature_count=35,
            min_bars=70,
            vector_class="FeatureVector",
        )
        # MlFeatureSchema only has hashable fields — must be usable as dict key
        d = {schema: "ok"}
        assert d[schema] == "ok"


# ---------------------------------------------------------------------------
# MlModelArtifacts
# ---------------------------------------------------------------------------

def _make_schema() -> MlFeatureSchema:
    return MlFeatureSchema(
        feature_names=NAMES_35,
        feature_count=35,
        min_bars=70,
        vector_class="FeatureVector",
    )


class TestMlModelArtifacts:
    def test_fields(self) -> None:
        schema = _make_schema()
        art = MlModelArtifacts(
            version="v20240601_1200",
            symbol="EURUSD",
            created_at=1717228800.0,
            feature_schema=schema,
            model_bytes=b"fake_model_bytes",
            metrics={"accuracy": 0.65, "n_samples": 500},
        )
        assert art.version == "v20240601_1200"
        assert art.model_bytes == b"fake_model_bytes"
        assert art.tags == {}

    def test_frozen(self) -> None:
        art = MlModelArtifacts(
            version="v1",
            symbol="EURUSD",
            created_at=0.0,
            feature_schema=_make_schema(),
            model_bytes=b"",
            metrics={},
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            art.version = "v2"  # type: ignore[misc]

    def test_metrics_json_serializable(self) -> None:
        """Metrics dict must survive a JSON round-trip (used in ArtifactStore)."""
        import json

        schema = _make_schema()
        metrics: dict[str, object] = {
            "accuracy": 0.65,
            "n_samples": 500,
            "feature_importances": {"f0": 0.1, "f1": 0.05},
            "class_distribution": {"1": 120, "-1": 80, "0": 300},
        }
        art = MlModelArtifacts(
            version="v1",
            symbol="EURUSD",
            created_at=0.0,
            feature_schema=schema,
            model_bytes=b"x",
            metrics=metrics,
        )
        # Should not raise
        serialized = json.dumps(art.metrics)
        recovered = json.loads(serialized)
        assert recovered["accuracy"] == 0.65


# ---------------------------------------------------------------------------
# MlEvaluationReport
# ---------------------------------------------------------------------------

class TestMlEvaluationReport:
    def test_valid(self) -> None:
        r = MlEvaluationReport(
            fold_id=0,
            n_samples=300,
            accuracy=0.60,
            buy_precision=0.58,
            sell_precision=0.62,
            hold_precision=0.61,
            buy_recall=0.55,
            sell_recall=0.60,
            class_distribution={1: 100, -1: 80, 0: 120},
        )
        assert r.accuracy == 0.60
        assert r.fold_id == 0

    def test_invalid_accuracy(self) -> None:
        with pytest.raises(ValueError, match="accuracy"):
            MlEvaluationReport(
                fold_id=0,
                n_samples=10,
                accuracy=1.5,  # out of range
                buy_precision=0.5,
                sell_precision=0.5,
                hold_precision=0.5,
                buy_recall=0.5,
                sell_recall=0.5,
                class_distribution={},
            )

    def test_invalid_precision(self) -> None:
        with pytest.raises(ValueError):
            MlEvaluationReport(
                fold_id=0,
                n_samples=10,
                accuracy=0.5,
                buy_precision=-0.1,  # negative
                sell_precision=0.5,
                hold_precision=0.5,
                buy_recall=0.5,
                sell_recall=0.5,
                class_distribution={},
            )

    def test_frozen(self) -> None:
        r = MlEvaluationReport(
            fold_id=0,
            n_samples=10,
            accuracy=0.5,
            buy_precision=0.5,
            sell_precision=0.5,
            hold_precision=0.5,
            buy_recall=0.5,
            sell_recall=0.5,
            class_distribution={},
        )
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            r.fold_id = 99  # type: ignore[misc]
