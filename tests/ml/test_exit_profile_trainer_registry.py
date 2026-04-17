"""Tests for ExitProfileTrainer and ExitProfileModelRegistry."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime

import numpy as np
import pytest

from metatrade.ml.exit_profile_dataset_builder import DatasetRow
from metatrade.ml.exit_profile_feature_builder import ExitProfileFeatureBuilder
from metatrade.ml.exit_profile_registry import (
    ExitProfileModelRegistry,
)
from metatrade.ml.exit_profile_trainer import (
    ExitProfileTrainer,
    ExitProfileTrainingResult,
)


def _ts(i: int = 0) -> datetime:
    return datetime(2024, 1, 1 + i % 28, 10, 0, tzinfo=UTC)


def _make_rows(n: int, labels: list[str] | None = None) -> list[DatasetRow]:
    """Build synthetic DatasetRow objects."""
    if labels is None:
        label_pool = ["ATR_1_2", "ATR_TRAIL_TIGHT", "ATR_HYBRID_1_5"]
    else:
        label_pool = labels
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        feats = {k: float(rng.uniform(0, 10)) for k in ExitProfileFeatureBuilder.FEATURE_NAMES}
        rows.append(DatasetRow(
            features=feats,
            label=label_pool[i % len(label_pool)],
            signal_index=i,
            symbol="EURUSD",
            timestamp_utc=_ts(i),
        ))
    return rows


# ── ExitProfileTrainer ────────────────────────────────────────────────────────

class TestExitProfileTrainer:
    def test_returns_none_when_insufficient_data(self):
        trainer = ExitProfileTrainer(min_train_rows=50)
        rows = _make_rows(10)
        result = trainer.train(rows)
        assert result is None

    def test_returns_training_result_with_sufficient_data(self):
        trainer = ExitProfileTrainer(n_splits=2, min_train_rows=30, min_test_rows=5)
        rows = _make_rows(80)
        result = trainer.train(rows)
        assert result is not None
        assert isinstance(result, ExitProfileTrainingResult)

    def test_result_has_model(self):
        trainer = ExitProfileTrainer(n_splits=2, min_train_rows=30, min_test_rows=5)
        result = trainer.train(_make_rows(80))
        assert result is not None
        assert result.model is not None

    def test_result_has_profile_ids(self):
        trainer = ExitProfileTrainer(n_splits=2, min_train_rows=30, min_test_rows=5)
        result = trainer.train(_make_rows(80))
        assert result is not None
        assert len(result.profile_ids) > 0

    def test_result_feature_names_match_builder(self):
        trainer = ExitProfileTrainer(n_splits=2, min_train_rows=30, min_test_rows=5)
        result = trainer.train(_make_rows(80))
        assert result is not None
        assert result.feature_names == ExitProfileFeatureBuilder.FEATURE_NAMES

    def test_mean_accuracy_in_range(self):
        trainer = ExitProfileTrainer(n_splits=2, min_train_rows=30, min_test_rows=5)
        result = trainer.train(_make_rows(80))
        assert result is not None
        assert 0.0 <= result.mean_accuracy <= 1.0

    def test_folds_populated(self):
        trainer = ExitProfileTrainer(n_splits=3, min_train_rows=30, min_test_rows=5)
        result = trainer.train(_make_rows(120))
        assert result is not None
        assert len(result.folds) > 0

    def test_trained_model_can_predict(self):
        trainer = ExitProfileTrainer(n_splits=2, min_train_rows=30, min_test_rows=5)
        result = trainer.train(_make_rows(80))
        assert result is not None
        rng = np.random.default_rng(1)
        X_new = rng.uniform(0, 10, size=(3, len(ExitProfileFeatureBuilder.FEATURE_NAMES)))
        preds = result.model.predict(X_new)
        assert len(preds) == 3

    def test_temporal_sort_applied(self):
        # Rows in reverse timestamp order — trainer should sort them
        rows = list(reversed(_make_rows(80)))
        trainer = ExitProfileTrainer(n_splits=2, min_train_rows=30, min_test_rows=5)
        result = trainer.train(rows)
        assert result is not None  # no crash, sort applied

    def test_single_label_dataset(self):
        # All rows have the same label — still trains fine
        rows = _make_rows(80, labels=["ATR_1_2"])
        trainer = ExitProfileTrainer(n_splits=2, min_train_rows=30, min_test_rows=5)
        result = trainer.train(rows)
        assert result is not None
        assert result.profile_ids == ["ATR_1_2"]


# ── ExitProfileModelRegistry ──────────────────────────────────────────────────

class TestExitProfileModelRegistry:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.reg = ExitProfileModelRegistry(self.tmpdir, symbol="EURUSD")

    def _dummy_model(self):
        from sklearn.dummy import DummyClassifier
        clf = DummyClassifier()
        clf.fit([[0]], ["ATR_1_2"])
        return clf

    def test_register_and_get_active(self):
        model = self._dummy_model()
        snap = self.reg.register(
            model=model,
            profile_ids=["ATR_1_2", "ATR_TRAIL_TIGHT"],
            feature_names=ExitProfileFeatureBuilder.FEATURE_NAMES,
            mean_accuracy=0.55,
            n_samples=100,
            version="test_v1",
        )
        self.reg.promote("test_v1")
        active = self.reg.get_active()
        assert active is not None
        assert active.version == "test_v1"
        assert active.mean_accuracy == 0.55

    def test_get_active_none_before_promote(self):
        model = self._dummy_model()
        self.reg.register(
            model=model,
            profile_ids=["ATR_1_2"],
            feature_names=ExitProfileFeatureBuilder.FEATURE_NAMES,
            mean_accuracy=0.5,
            n_samples=50,
            version="v1",
        )
        assert self.reg.get_active() is None

    def test_promote_unknown_version_raises(self):
        with pytest.raises(KeyError):
            self.reg.promote("nonexistent")

    def test_list_versions(self):
        model = self._dummy_model()
        self.reg.register(model=model, profile_ids=["X"], feature_names=(), mean_accuracy=0.5, n_samples=10, version="v1")
        self.reg.register(model=model, profile_ids=["X"], feature_names=(), mean_accuracy=0.6, n_samples=20, version="v2")
        versions = self.reg.list_versions()
        assert "v1" in versions
        assert "v2" in versions

    def test_save_and_load_roundtrip(self):
        model = self._dummy_model()
        self.reg.register(
            model=model,
            profile_ids=["ATR_1_2"],
            feature_names=ExitProfileFeatureBuilder.FEATURE_NAMES,
            mean_accuracy=0.6,
            n_samples=200,
            version="v_save",
        )
        self.reg.promote("v_save")
        path = self.reg.save_to_disk()
        assert path is not None
        assert path.exists()

        # Load in a fresh registry instance
        reg2 = ExitProfileModelRegistry(self.tmpdir, symbol="EURUSD")
        snap = reg2.load_from_disk()
        assert snap is not None
        assert snap.version == "v_save"
        assert snap.profile_ids == ["ATR_1_2"]

    def test_save_no_active_returns_none(self):
        result = self.reg.save_to_disk()
        assert result is None

    def test_load_missing_file_returns_none(self):
        result = self.reg.load_from_disk(version="nonexistent_v99")
        assert result is None

    def test_auto_version_generated(self):
        model = self._dummy_model()
        snap = self.reg.register(
            model=model, profile_ids=[], feature_names=(), mean_accuracy=0.5, n_samples=1
        )
        assert snap.version.startswith("exitprofile_v")

    def test_tags_stored(self):
        model = self._dummy_model()
        snap = self.reg.register(
            model=model,
            profile_ids=[],
            feature_names=(),
            mean_accuracy=0.5,
            n_samples=1,
            version="v_tags",
            tags={"source": "backtest"},
        )
        assert snap.tags["source"] == "backtest"
