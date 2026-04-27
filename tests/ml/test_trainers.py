"""Tests for ml/trainers.py — ExpectedValueTrainer."""

from __future__ import annotations

import pickle
from unittest.mock import MagicMock, patch

import pytest

from metatrade.ml.targets import ContinuousTargetBuilder, MlContinuousTarget
from metatrade.ml.trainers import EvMetrics, ExpectedValueTrainer

# Reuse helpers from the main ML test suite
from tests.ml.test_ml import _test_cfg, make_labeled_dataset, make_bars
from metatrade.ml.features import MIN_FEATURE_BARS, extract_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dataset(
    n_bars: int = 300,
    forward_bars: int = 5,
    atr_period: int = 14,
) -> tuple[list, list[MlContinuousTarget]]:
    """Build aligned (feature_vectors, targets) using the same bar series."""
    bars = make_bars(n_bars, start=1.1, step=0.0002)

    target_builder = ContinuousTargetBuilder(
        forward_bars=forward_bars, atr_period=atr_period
    )
    targets = target_builder.build(bars)

    from datetime import datetime, timezone
    feature_vectors = []
    valid_targets = []
    for t in targets:
        start = t.bar_index - MIN_FEATURE_BARS + 1
        if start < 0:
            continue
        window = bars[start: t.bar_index + 1]
        if len(window) < MIN_FEATURE_BARS:
            continue
        ts = window[-1].timestamp_utc
        fv = extract_features(window, timestamp_utc=ts)
        if fv is None:
            continue
        feature_vectors.append(fv)
        valid_targets.append(t)

    return feature_vectors, valid_targets


def _fake_histgbm() -> object:
    """Return a real HistGBM regressor to stand in for mocked backends."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=10, random_state=42)


# ---------------------------------------------------------------------------
# EvMetrics
# ---------------------------------------------------------------------------

class TestEvMetrics:
    def test_to_dict_keys(self) -> None:
        m = EvMetrics(r2=0.10, mae=0.5, n_samples=200)
        d = m.to_dict()
        assert "r2" in d
        assert "mae" in d
        assert "n_samples" in d
        assert "feature_importances" in d

    def test_to_dict_json_serializable(self) -> None:
        import json
        m = EvMetrics(r2=0.10, mae=0.5, n_samples=200, feature_importances={"f0": 0.1})
        json.dumps(m.to_dict())  # should not raise


# ---------------------------------------------------------------------------
# ExpectedValueTrainer — construction
# ---------------------------------------------------------------------------

class TestExpectedValueTrainerInit:
    def test_not_trained_by_default(self) -> None:
        trainer = ExpectedValueTrainer(_test_cfg())
        assert not trainer.is_trained

    def test_metrics_none_before_fit(self) -> None:
        assert ExpectedValueTrainer().metrics is None


# ---------------------------------------------------------------------------
# ExpectedValueTrainer — fit()
# ---------------------------------------------------------------------------

class TestExpectedValueTrainerFit:
    def test_fit_returns_metrics(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data to build dataset")
        trainer = ExpectedValueTrainer(_test_cfg())
        metrics = trainer.fit(fvs, targets)
        assert isinstance(metrics, EvMetrics)

    def test_is_trained_after_fit(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data to build dataset")
        trainer = ExpectedValueTrainer(_test_cfg())
        trainer.fit(fvs, targets)
        assert trainer.is_trained

    def test_metrics_n_samples(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data to build dataset")
        trainer = ExpectedValueTrainer(_test_cfg())
        metrics = trainer.fit(fvs, targets)
        assert metrics.n_samples == len(fvs)

    def test_length_mismatch_raises(self) -> None:
        fvs, targets = _make_dataset()
        if len(fvs) < 2:
            pytest.skip("not enough data")
        trainer = ExpectedValueTrainer(_test_cfg())
        with pytest.raises(ValueError, match="equal length"):
            trainer.fit(fvs[:-1], targets)

    def test_too_few_samples_raises(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data to build dataset")
        trainer = ExpectedValueTrainer(_test_cfg(min_train_samples=10))
        with pytest.raises(ValueError, match="at least"):
            trainer.fit(fvs[:1], targets[:1])


# ---------------------------------------------------------------------------
# ExpectedValueTrainer — predict()
# ---------------------------------------------------------------------------

class TestExpectedValueTrainerPredict:
    def _trained(self) -> tuple[ExpectedValueTrainer, list]:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data to build dataset")
        trainer = ExpectedValueTrainer(_test_cfg())
        trainer.fit(fvs, targets)
        return trainer, fvs

    def test_predict_returns_float(self) -> None:
        trainer, fvs = self._trained()
        result = trainer.predict(fvs[0])
        assert isinstance(result, float)

    def test_predict_untrained_raises(self) -> None:
        from metatrade.ml.features import extract_features
        from datetime import datetime, timezone
        fvs, _ = make_labeled_dataset(200)
        trainer = ExpectedValueTrainer(_test_cfg())
        with pytest.raises(RuntimeError, match="not been trained"):
            trainer.predict(fvs[0])


# ---------------------------------------------------------------------------
# ExpectedValueTrainer — serialize / deserialize
# ---------------------------------------------------------------------------

class TestExpectedValueTrainerSerialization:
    def _trained(self) -> ExpectedValueTrainer:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data to build dataset")
        trainer = ExpectedValueTrainer(_test_cfg())
        trainer.fit(fvs, targets)
        return trainer

    def test_round_trip_is_trained(self) -> None:
        trainer = self._trained()
        restored = ExpectedValueTrainer.deserialize(trainer.serialize())
        assert restored.is_trained

    def test_round_trip_feature_names(self) -> None:
        trainer = self._trained()
        restored = ExpectedValueTrainer.deserialize(trainer.serialize())
        assert restored._feature_names == trainer._feature_names

    def test_round_trip_metrics(self) -> None:
        trainer = self._trained()
        restored = ExpectedValueTrainer.deserialize(trainer.serialize())
        assert restored.metrics is not None
        assert abs(restored.metrics.r2 - trainer.metrics.r2) < 1e-9  # type: ignore[union-attr]

    def test_round_trip_predict_consistent(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data to build dataset")
        trainer = ExpectedValueTrainer(_test_cfg())
        trainer.fit(fvs, targets)
        fv = fvs[0]
        original = trainer.predict(fv)
        restored = ExpectedValueTrainer.deserialize(trainer.serialize())
        assert abs(restored.predict(fv) - original) < 1e-9

    def test_serialize_untrained_raises(self) -> None:
        with pytest.raises(RuntimeError, match="No trained"):
            ExpectedValueTrainer().serialize()

    def test_round_trip_backend_preserved(self) -> None:
        trainer = self._trained()
        trainer._backend = "xgboost"
        restored = ExpectedValueTrainer.deserialize(trainer.serialize())
        assert restored._backend == "xgboost"

    def test_deserialize_legacy_payload_defaults_backend_to_histgbm(self) -> None:
        trainer = self._trained()
        old_payload = {
            "model": pickle.dumps(trainer._model),
            "feature_names": trainer._feature_names,
            "metrics": trainer.metrics.to_dict() if trainer.metrics else {},
        }
        data = pickle.dumps(old_payload)
        restored = ExpectedValueTrainer.deserialize(data)
        assert restored._backend == "histgbm"
        assert restored.is_trained


# ---------------------------------------------------------------------------
# ExpectedValueTrainer — backend selection
# ---------------------------------------------------------------------------

class TestExpectedValueTrainerBackend:
    def test_default_backend_is_histgbm(self) -> None:
        trainer = ExpectedValueTrainer(_test_cfg())
        assert trainer.backend == "histgbm"

    def test_backend_kwarg_overrides_config(self) -> None:
        trainer = ExpectedValueTrainer(_test_cfg(), backend="histgbm")
        assert trainer.backend == "histgbm"

    def test_histgbm_fit_produces_metrics(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        trainer = ExpectedValueTrainer(_test_cfg(), backend="histgbm")
        assert isinstance(trainer.fit(fvs, targets), EvMetrics)

    def test_unknown_backend_falls_back_to_histgbm(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        trainer = ExpectedValueTrainer(_test_cfg(), backend="bogus_backend")
        assert isinstance(trainer.fit(fvs, targets), EvMetrics)

    # ------------------------------------------------------------------
    # lightgbm
    # ------------------------------------------------------------------

    def test_lightgbm_backend_used_when_installed(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        fake_lgb = MagicMock()
        fake_lgb.LGBMRegressor.return_value = _fake_histgbm()
        with patch.dict("sys.modules", {"lightgbm": fake_lgb}):
            trainer = ExpectedValueTrainer(_test_cfg(), backend="lightgbm")
            trainer.fit(fvs, targets)
        assert trainer.backend == "lightgbm"
        fake_lgb.LGBMRegressor.assert_called_once()

    def test_lightgbm_gpu_flag_sets_device(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        fake_lgb = MagicMock()
        fake_lgb.LGBMRegressor.return_value = _fake_histgbm()
        with patch.dict("sys.modules", {"lightgbm": fake_lgb}):
            trainer = ExpectedValueTrainer(_test_cfg(use_gpu=True), backend="lightgbm")
            trainer.fit(fvs, targets)
        assert fake_lgb.LGBMRegressor.call_args[1].get("device") == "gpu"

    def test_lightgbm_importerror_falls_back_to_histgbm(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        with patch.dict("sys.modules", {"lightgbm": None}):
            trainer = ExpectedValueTrainer(_test_cfg(), backend="lightgbm")
            assert isinstance(trainer.fit(fvs, targets), EvMetrics)

    # ------------------------------------------------------------------
    # xgboost
    # ------------------------------------------------------------------

    def test_xgboost_backend_used_when_installed(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        fake_xgb = MagicMock()
        fake_xgb.XGBRegressor.return_value = _fake_histgbm()
        with patch.dict("sys.modules", {"xgboost": fake_xgb}):
            trainer = ExpectedValueTrainer(_test_cfg(), backend="xgboost")
            trainer.fit(fvs, targets)
        assert trainer.backend == "xgboost"
        fake_xgb.XGBRegressor.assert_called_once()

    def test_xgboost_gpu_flag_sets_device_and_tree_method(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        fake_xgb = MagicMock()
        fake_xgb.XGBRegressor.return_value = _fake_histgbm()
        with patch.dict("sys.modules", {"xgboost": fake_xgb}):
            trainer = ExpectedValueTrainer(_test_cfg(use_gpu=True), backend="xgboost")
            trainer.fit(fvs, targets)
        kwargs = fake_xgb.XGBRegressor.call_args[1]
        assert kwargs.get("device") == "cuda"
        assert kwargs.get("tree_method") == "hist"

    def test_xgboost_uses_squarederror_objective(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        fake_xgb = MagicMock()
        fake_xgb.XGBRegressor.return_value = _fake_histgbm()
        with patch.dict("sys.modules", {"xgboost": fake_xgb}):
            trainer = ExpectedValueTrainer(_test_cfg(), backend="xgboost")
            trainer.fit(fvs, targets)
        assert fake_xgb.XGBRegressor.call_args[1].get("objective") == "reg:squarederror"

    def test_xgboost_importerror_falls_back_to_histgbm(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        with patch.dict("sys.modules", {"xgboost": None}):
            trainer = ExpectedValueTrainer(_test_cfg(), backend="xgboost")
            assert isinstance(trainer.fit(fvs, targets), EvMetrics)

    # ------------------------------------------------------------------
    # catboost
    # ------------------------------------------------------------------

    def test_catboost_backend_used_when_installed(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        fake_cb_reg = MagicMock(return_value=_fake_histgbm())
        fake_catboost = MagicMock()
        fake_catboost.CatBoostRegressor = fake_cb_reg
        with patch.dict("sys.modules", {"catboost": fake_catboost}):
            trainer = ExpectedValueTrainer(_test_cfg(), backend="catboost")
            trainer.fit(fvs, targets)
        assert trainer.backend == "catboost"
        fake_cb_reg.assert_called_once()

    def test_catboost_gpu_sets_task_type_when_enough_memory(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        fake_cb_reg = MagicMock(return_value=_fake_histgbm())
        fake_catboost = MagicMock()
        fake_catboost.CatBoostRegressor = fake_cb_reg
        with patch.dict("sys.modules", {"catboost": fake_catboost}):
            with patch("metatrade.ml.trainers._free_gpu_mb", return_value=4096):
                trainer = ExpectedValueTrainer(_test_cfg(use_gpu=True), backend="catboost")
                trainer.fit(fvs, targets)
        assert fake_cb_reg.call_args[1].get("task_type") == "GPU"

    def test_catboost_gpu_skipped_when_low_memory(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        fake_cb_reg = MagicMock(return_value=_fake_histgbm())
        fake_catboost = MagicMock()
        fake_catboost.CatBoostRegressor = fake_cb_reg
        with patch.dict("sys.modules", {"catboost": fake_catboost}):
            with patch("metatrade.ml.trainers._free_gpu_mb", return_value=512):
                trainer = ExpectedValueTrainer(_test_cfg(use_gpu=True), backend="catboost")
                trainer.fit(fvs, targets)
        assert "task_type" not in fake_cb_reg.call_args[1]

    def test_catboost_importerror_falls_back_to_histgbm(self) -> None:
        fvs, targets = _make_dataset()
        if not fvs:
            pytest.skip("not enough data")
        with patch.dict("sys.modules", {"catboost": None}):
            trainer = ExpectedValueTrainer(_test_cfg(), backend="catboost")
            assert isinstance(trainer.fit(fvs, targets), EvMetrics)
