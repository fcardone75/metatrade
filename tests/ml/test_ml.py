"""Tests for the ML module: features, labels, classifier, registry, module."""

from __future__ import annotations

import math
import os
import tempfile
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import SignalDirection, Timeframe
from metatrade.core.errors import ModuleNotReadyError
from metatrade.ml.classifier import ClassifierMetrics, MLClassifier
from metatrade.ml.config import MLConfig
from metatrade.ml.features import MIN_FEATURE_BARS, FeatureVector, extract_features
from metatrade.ml.labels import label_bars, label_single
from metatrade.ml.module import MLModule
from metatrade.ml.registry import ModelRegistry, ModelSnapshot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def D(v: str | float | int) -> Decimal:
    return Decimal(str(v))


def make_bar(
    close: float,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1000.0,
    i: int = 0,
) -> Bar:
    o = open_ if open_ is not None else close
    h = high if high is not None else close + 0.0010
    lo = low if low is not None else close - 0.0010
    ts = datetime(2024, 1, 1 + (i // 24), i % 24, 0, 0, tzinfo=UTC)
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_utc=ts,
        open=D(str(round(o, 5))),
        high=D(str(round(h, 5))),
        low=D(str(round(lo, 5))),
        close=D(str(round(close, 5))),
        volume=D(str(volume)),
    )


def make_bars(n: int, start: float = 1.1, step: float = 0.0001) -> list[Bar]:
    return [make_bar(round(start + i * step, 5), i=i) for i in range(n)]


def make_uniform_bars(n: int, price: float = 1.1, volume: float = 1000.0) -> list[Bar]:
    return [make_bar(price, volume=volume, i=i) for i in range(n)]


def _test_cfg(**kwargs) -> MLConfig:
    """Create an MLConfig with test-friendly defaults that pass all validators."""
    defaults = dict(
        min_train_samples=10,
        max_iter=10,
        min_accuracy=0.5,   # floor value — almost any real fit exceeds this
    )
    defaults.update(kwargs)
    return MLConfig(**defaults)


def make_labeled_dataset(
    n: int = 200,
) -> tuple[list[FeatureVector], list[int]]:
    """Build a synthetic labeled dataset using zigzag prices."""
    bars = []
    for i in range(n + MIN_FEATURE_BARS):
        close = 1.1 + (0.01 if i % 10 < 5 else -0.01) * ((i // 10) % 3 + 1)
        bars.append(make_bar(round(close, 5), i=i))

    labels = label_bars(bars, forward_bars=3, atr_threshold_mult=0.5)
    X: list[FeatureVector] = []
    y: list[int] = []
    for i, lbl in enumerate(labels):
        if lbl is None:
            continue
        window = bars[max(0, i - MIN_FEATURE_BARS + 1) : i + 1]
        fv = extract_features(window)
        if fv is not None:
            X.append(fv)
            y.append(lbl)
    return X, y


def make_trained_classifier(n: int = 200) -> MLClassifier:
    cfg = _test_cfg()
    clf = MLClassifier(cfg)
    X, y = make_labeled_dataset(n)
    clf.fit(X, y)
    return clf


# ---------------------------------------------------------------------------
# FeatureVector
# ---------------------------------------------------------------------------

def _make_fv(**kwargs) -> FeatureVector:
    """Create a FeatureVector with default zero values, overriding given fields."""
    defaults = dict(
        returns_1=0.001, returns_2=0.002, returns_3=0.003,
        returns_5=0.003, returns_10=0.005, returns_20=0.008, returns_30=0.010,
        ema9_dist=0.0002, ema21_dist=-0.0001, ema9_21_cross=0.0003,
        rsi5=0.60, rsi14=0.55, rsi21=0.52, rsi14_change=0.01,
        macd_signal_norm=0.0001, macd_hist_norm=0.00005,
        stoch_k=0.5, stoch_d=0.5,
        atr14_rel=0.002, rolling_std5=0.001, rolling_std20=0.0008,
        body_ratio=0.5, upper_wick=0.3, lower_wick=0.2,
        vol_rel=1.0, hour_sin=0.0, hour_cos=1.0,
        weekday_sin=0.0, weekday_cos=1.0,
        adx_slope=0.01, atr_zscore=0.0, donchian_pos=0.5,
        bb_pct_b=0.5, rsi_divergence=0.0, hurst_short=0.5,
    )
    defaults.update(kwargs)
    return FeatureVector(**defaults)


class TestFeatureVector:
    def test_to_list_length(self) -> None:
        fv = _make_fv()
        assert len(fv.to_list()) == 35  # 27 original + 8 new

    def test_feature_names_length(self) -> None:
        assert len(FeatureVector.feature_names()) == 35  # 27 original + 8 new

    def test_feature_names_are_strings(self) -> None:
        for name in FeatureVector.feature_names():
            assert isinstance(name, str)
            assert len(name) > 0

    def test_to_list_order_matches_names(self) -> None:
        # returns_1 is first, hurst_short is last
        fv = _make_fv(returns_1=1.0, hurst_short=27.0)
        values = fv.to_list()
        assert values[0] == 1.0    # returns_1 is first
        assert values[-1] == 27.0  # hurst_short is last


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_none_when_too_few_bars(self) -> None:
        bars = make_bars(MIN_FEATURE_BARS - 1)
        result = extract_features(bars)
        assert result is None

    def test_returns_feature_vector_with_enough_bars(self) -> None:
        bars = make_bars(MIN_FEATURE_BARS + 5)
        result = extract_features(bars)
        assert isinstance(result, FeatureVector)

    def test_returns_none_for_empty_list(self) -> None:
        assert extract_features([]) is None

    def test_all_features_are_finite(self) -> None:
        bars = make_bars(MIN_FEATURE_BARS + 10)
        fv = extract_features(bars)
        assert fv is not None
        for v in fv.to_list():
            assert math.isfinite(v), f"Feature value {v} is not finite"

    def test_rsi14_scaled_0_to_1(self) -> None:
        # Strongly rising prices → RSI close to 1.0 when scaled
        bars = make_bars(MIN_FEATURE_BARS + 5, step=0.001)
        fv = extract_features(bars)
        assert fv is not None
        assert 0.0 <= fv.rsi14 <= 1.0

    def test_body_ratio_0_to_1(self) -> None:
        bars = make_bars(MIN_FEATURE_BARS + 5)
        fv = extract_features(bars)
        assert fv is not None
        assert 0.0 <= fv.body_ratio <= 1.0 + 1e-9  # small float tolerance

    def test_atr14_rel_positive(self) -> None:
        bars = make_bars(MIN_FEATURE_BARS + 5)
        fv = extract_features(bars)
        assert fv is not None
        assert fv.atr14_rel > 0.0

    def test_vol_rel_near_1_for_uniform_volume(self) -> None:
        bars = make_uniform_bars(MIN_FEATURE_BARS + 5, volume=500.0)
        fv = extract_features(bars)
        assert fv is not None
        assert abs(fv.vol_rel - 1.0) < 0.1

    def test_exact_min_bars_works(self) -> None:
        bars = make_bars(MIN_FEATURE_BARS)
        fv = extract_features(bars)
        assert fv is not None

    def test_returns_negative_for_falling_prices(self) -> None:
        # Falling prices → returns_1 should be negative
        bars = make_bars(MIN_FEATURE_BARS + 5, start=1.2, step=-0.0002)
        fv = extract_features(bars)
        assert fv is not None
        assert fv.returns_1 < 0.0


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

class TestLabelSingle:
    def test_buy_label_on_positive_return(self) -> None:
        result = label_single(1.10, 1.12, 0.005)
        assert result == 1

    def test_sell_label_on_negative_return(self) -> None:
        result = label_single(1.10, 1.08, 0.005)
        assert result == -1

    def test_hold_label_within_threshold(self) -> None:
        result = label_single(1.10, 1.10005, 0.005)
        assert result == 0

    def test_exactly_at_threshold_is_hold(self) -> None:
        # Return of exactly +threshold → not strictly greater → HOLD
        result = label_single(1.00, 1.005, 0.005)
        assert result == 0  # 0.005 is not > 0.005

    def test_zero_threshold_always_directional(self) -> None:
        # Any positive return → BUY with 0 threshold
        result = label_single(1.0, 1.0001, 0.0)
        assert result == 1

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_pct must be >= 0"):
            label_single(1.0, 1.01, -0.001)


class TestLabelBars:
    def test_returns_list_same_length(self) -> None:
        bars = make_bars(50)
        labels = label_bars(bars, forward_bars=5)
        assert len(labels) == 50

    def test_last_n_labels_are_none(self) -> None:
        bars = make_bars(50)
        labels = label_bars(bars, forward_bars=5)
        # Last 5 labels should be None (no future data)
        for lbl in labels[-5:]:
            assert lbl is None

    def test_early_labels_are_not_none(self) -> None:
        bars = make_bars(50)
        labels = label_bars(bars, forward_bars=5)
        non_none = [l for l in labels[:-5] if l is not None]
        assert len(non_none) > 0

    def test_all_labels_are_valid_values(self) -> None:
        bars = make_bars(50)
        labels = label_bars(bars, forward_bars=5)
        for lbl in labels:
            assert lbl in (1, -1, 0, None)

    def test_raises_invalid_forward_bars(self) -> None:
        bars = make_bars(50)
        with pytest.raises(ValueError, match="forward_bars must be >= 1"):
            label_bars(bars, forward_bars=0)

    def test_raises_invalid_atr_mult(self) -> None:
        bars = make_bars(50)
        with pytest.raises(ValueError, match="atr_threshold_mult must be > 0"):
            label_bars(bars, atr_threshold_mult=-1.0)

    def test_rising_bars_produce_buy_labels(self) -> None:
        # Steeply rising bars should produce BUY labels
        bars = make_bars(60, step=0.01)  # 1% step per bar
        labels = label_bars(bars, forward_bars=3, atr_threshold_mult=0.5)
        buy_count = sum(1 for l in labels if l == 1)
        assert buy_count > 0

    def test_falling_bars_produce_sell_labels(self) -> None:
        bars = make_bars(60, start=2.0, step=-0.01)
        labels = label_bars(bars, forward_bars=3, atr_threshold_mult=0.5)
        sell_count = sum(1 for l in labels if l == -1)
        assert sell_count > 0


# ---------------------------------------------------------------------------
# MLClassifier
# ---------------------------------------------------------------------------

class TestMLClassifier:
    def test_not_trained_initially(self) -> None:
        clf = MLClassifier()
        assert not clf.is_trained
        assert clf.metrics is None

    def test_fit_returns_metrics(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(150)
        metrics = clf.fit(X, y)
        assert isinstance(metrics, ClassifierMetrics)
        assert metrics.n_samples == len(X)
        assert 0.0 <= metrics.accuracy <= 1.0

    def test_fit_marks_trained_when_accuracy_ok(self) -> None:
        # With min_accuracy=0.0, any fit should mark as trained
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(150)
        clf.fit(X, y)
        assert clf.is_trained

    def test_fit_not_trained_when_below_min_accuracy(self) -> None:
        # Random labels + depth-1 stumps → in-sample accuracy ≈ 0.33 (3-class)
        # which is below min_accuracy=0.99 → is_trained stays False
        import random as _random
        rng = _random.Random(42)
        cfg = _test_cfg(min_accuracy=0.99, max_depth=1)
        clf = MLClassifier(cfg)
        X, _ = make_labeled_dataset(200)
        random_y = [rng.choice([1, -1, 0]) for _ in X]
        clf.fit(X, random_y)
        assert not clf.is_trained

    def test_raises_when_samples_below_minimum(self) -> None:
        cfg = _test_cfg(min_train_samples=100)
        clf = MLClassifier(cfg)
        X, y = make_labeled_dataset(20)
        with pytest.raises(ValueError, match="Need at least"):
            clf.fit(X[:5], y[:5])

    def test_raises_when_x_y_mismatch(self) -> None:
        clf = MLClassifier(_test_cfg())
        X, y = make_labeled_dataset(150)
        with pytest.raises(ValueError, match="equal length"):
            clf.fit(X[:10], y[:5])

    def test_predict_returns_direction_and_confidence(self) -> None:
        clf = make_trained_classifier()
        bars = make_bars(MIN_FEATURE_BARS + 5)
        fv = extract_features(bars)
        assert fv is not None
        direction, confidence = clf.predict(fv)
        assert direction in (1, -1, 0)
        assert 0.0 <= confidence <= 1.0

    def test_predict_raises_if_not_trained(self) -> None:
        clf = MLClassifier()
        bars = make_bars(MIN_FEATURE_BARS + 5)
        fv = extract_features(bars)
        assert fv is not None
        with pytest.raises(RuntimeError, match="not been trained"):
            clf.predict(fv)

    def test_metrics_has_feature_importances(self) -> None:
        clf = make_trained_classifier()
        assert clf.metrics is not None
        fi = clf.metrics.feature_importances
        assert len(fi) == 35  # one per feature (27 original + 8 new)

    def test_metrics_has_class_distribution(self) -> None:
        clf = make_trained_classifier()
        assert clf.metrics is not None
        # At least one class should be present
        assert len(clf.metrics.class_distribution) > 0

    def test_serialize_and_deserialize(self) -> None:
        clf = make_trained_classifier()
        data = clf.serialize()
        assert isinstance(data, bytes)
        clf2 = MLClassifier.deserialize(data)
        assert clf2.is_trained
        # Both should make the same predictions
        bars = make_bars(MIN_FEATURE_BARS + 5)
        fv = extract_features(bars)
        assert fv is not None
        d1, c1 = clf.predict(fv)
        d2, c2 = clf2.predict(fv)
        assert d1 == d2
        assert abs(c1 - c2) < 1e-9

    def test_serialize_raises_if_not_trained(self) -> None:
        clf = MLClassifier()
        with pytest.raises(RuntimeError, match="No trained model"):
            clf.serialize()


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_get_active_returns_none_when_empty(self) -> None:
        reg = ModelRegistry()
        assert reg.get_active() is None

    def test_register_trained_model(self) -> None:
        reg = ModelRegistry()
        clf = make_trained_classifier()
        snapshot = reg.register(clf, symbol="EURUSD", version="v1")
        assert isinstance(snapshot, ModelSnapshot)
        assert snapshot.version == "v1"
        assert snapshot.symbol == "EURUSD"

    def test_register_makes_it_active(self) -> None:
        reg = ModelRegistry()
        clf = make_trained_classifier()
        reg.register(clf, symbol="EURUSD", version="v1")
        active = reg.get_active()
        assert active is not None
        assert active.version == "v1"

    def test_register_untrained_raises(self) -> None:
        reg = ModelRegistry()
        clf = MLClassifier()
        with pytest.raises(ValueError, match="untrained"):
            reg.register(clf, symbol="EURUSD")

    def test_auto_version_is_generated(self) -> None:
        reg = ModelRegistry()
        clf = make_trained_classifier()
        snapshot = reg.register(clf, symbol="EURUSD")
        assert snapshot.version.startswith("v")

    def test_get_specific_version(self) -> None:
        reg = ModelRegistry()
        clf = make_trained_classifier()
        reg.register(clf, symbol="EURUSD", version="v_abc")
        result = reg.get("v_abc")
        assert result is not None
        assert result.version == "v_abc"

    def test_get_unknown_version_returns_none(self) -> None:
        reg = ModelRegistry()
        assert reg.get("nonexistent") is None

    def test_list_versions_sorted_newest_first(self) -> None:
        reg = ModelRegistry()
        clf = make_trained_classifier()
        reg.register(clf, symbol="EURUSD", version="v1")
        reg.register(clf, symbol="EURUSD", version="v3")
        reg.register(clf, symbol="EURUSD", version="v2")
        versions = reg.list_versions()
        assert versions == sorted(versions, reverse=True)

    def test_promote_changes_active(self) -> None:
        reg = ModelRegistry()
        clf = make_trained_classifier()
        reg.register(clf, symbol="EURUSD", version="v1")
        reg.register(clf, symbol="EURUSD", version="v2")
        assert reg.get_active().version == "v2"
        reg.promote("v1")
        assert reg.get_active().version == "v1"

    def test_promote_raises_for_unknown_version(self) -> None:
        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.promote("nonexistent")

    def test_clear_empties_registry(self) -> None:
        reg = ModelRegistry()
        clf = make_trained_classifier()
        reg.register(clf, symbol="EURUSD", version="v1")
        reg.clear()
        assert reg.size == 0
        assert reg.get_active() is None

    def test_size_tracks_registrations(self) -> None:
        reg = ModelRegistry()
        clf = make_trained_classifier()
        assert reg.size == 0
        reg.register(clf, symbol="EURUSD", version="v1")
        assert reg.size == 1
        reg.register(clf, symbol="EURUSD", version="v2")
        assert reg.size == 2

    def test_tags_stored_in_snapshot(self) -> None:
        reg = ModelRegistry()
        clf = make_trained_classifier()
        tags = {"fold": 3, "symbol": "EURUSD"}
        snapshot = reg.register(clf, symbol="EURUSD", version="v1", tags=tags)
        assert snapshot.tags["fold"] == 3

    def test_persist_to_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _test_cfg(model_registry_dir=tmpdir)
            reg = ModelRegistry(config=cfg, persist=True)
            clf = make_trained_classifier()
            reg.register(clf, symbol="EURUSD", version="v_disk_test")
            # File should exist
            expected = os.path.join(tmpdir, "EURUSD_v_disk_test.pkl")
            assert os.path.exists(expected)

    def test_load_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _test_cfg(model_registry_dir=tmpdir)
            reg = ModelRegistry(config=cfg, persist=True)
            clf = make_trained_classifier()
            reg.register(clf, symbol="EURUSD", version="v_load_test")

            # Load from disk into a fresh registry
            reg2 = ModelRegistry(config=cfg, persist=False)
            snapshot = reg2.load_from_disk("EURUSD", "v_load_test")
            assert snapshot is not None
            assert snapshot.classifier.is_trained

    def test_load_from_disk_nonexistent_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = MLConfig(model_registry_dir=tmpdir)
            reg = ModelRegistry(config=cfg)
            result = reg.load_from_disk("EURUSD", "nonexistent_version")
            assert result is None


# ---------------------------------------------------------------------------
# MLModule (ITechnicalModule integration)
# ---------------------------------------------------------------------------

class TestMLModule:
    def _make_module(
        self, trained: bool = True, min_confidence: float = 0.0
    ) -> tuple[MLModule, ModelRegistry]:
        reg = ModelRegistry()
        if trained:
            clf = make_trained_classifier()
            reg.register(clf, symbol="EURUSD", version="v1")
        module = MLModule(registry=reg, timeframe_id="h1", min_confidence=min_confidence)
        return module, reg

    def test_module_id(self) -> None:
        module, _ = self._make_module()
        assert module.module_id == "ml_h1"

    def test_min_bars_equals_feature_minimum(self) -> None:
        module, _ = self._make_module()
        assert module.min_bars == MIN_FEATURE_BARS

    def test_raises_module_not_ready(self) -> None:
        module, _ = self._make_module()
        bars = make_bars(5)
        with pytest.raises(ModuleNotReadyError):
            module.analyse(bars, NOW)

    def test_returns_hold_when_no_model(self) -> None:
        reg = ModelRegistry()
        module = MLModule(registry=reg, min_confidence=0.0)
        bars = make_bars(MIN_FEATURE_BARS + 5)
        signal = module.analyse(bars, NOW)
        assert signal.direction == SignalDirection.HOLD
        assert signal.metadata["model_available"] is False

    def test_returns_signal_with_trained_model(self) -> None:
        module, _ = self._make_module(min_confidence=0.0)
        bars = make_bars(MIN_FEATURE_BARS + 5)
        signal = module.analyse(bars, NOW)
        assert signal.direction in (SignalDirection.BUY, SignalDirection.SELL, SignalDirection.HOLD)

    def test_confidence_in_range(self) -> None:
        module, _ = self._make_module(min_confidence=0.0)
        bars = make_bars(MIN_FEATURE_BARS + 5)
        signal = module.analyse(bars, NOW)
        assert 0.0 <= signal.confidence <= 1.0

    def test_low_confidence_produces_hold(self) -> None:
        module, _ = self._make_module(min_confidence=1.0)  # impossible threshold
        bars = make_bars(MIN_FEATURE_BARS + 5)
        signal = module.analyse(bars, NOW)
        # Confidence can never reach 1.0 → should be HOLD
        assert signal.direction == SignalDirection.HOLD

    def test_metadata_contains_model_version(self) -> None:
        module, _ = self._make_module(min_confidence=0.0)
        bars = make_bars(MIN_FEATURE_BARS + 5)
        signal = module.analyse(bars, NOW)
        if signal.metadata.get("model_available"):
            assert "model_version" in signal.metadata

    def test_timestamp_preserved(self) -> None:
        module, _ = self._make_module()
        bars = make_bars(MIN_FEATURE_BARS + 5)
        signal = module.analyse(bars, NOW)
        assert signal.timestamp_utc == NOW

    def test_module_id_custom_timeframe(self) -> None:
        reg = ModelRegistry()
        module = MLModule(registry=reg, timeframe_id="m15")
        assert module.module_id == "ml_m15"


# ---------------------------------------------------------------------------
# WalkForwardTrainer
# ---------------------------------------------------------------------------

class TestWalkForwardTrainer:
    """Tests for WalkForwardTrainer and related dataclasses."""

    def _small_cfg(self, **kwargs) -> MLConfig:
        """Config with tiny windows for fast tests."""
        defaults = dict(
            train_window_bars=200,   # minimum allowed by MLConfig (ge=200)
            test_window_bars=50,
            step_bars=50,
            min_train_samples=10,
            max_iter=10,
            min_accuracy=0.5,        # minimum allowed by MLConfig (ge=0.5)
        )
        defaults.update(kwargs)
        return MLConfig(**defaults)

    def test_run_returns_result_and_model(self) -> None:
        from metatrade.ml.walk_forward import WalkForwardTrainer
        cfg = self._small_cfg()
        bars = make_bars(500)
        trainer = WalkForwardTrainer(cfg)
        result, model = trainer.run(bars)
        # With 500 bars, train=200, test=50, step=50 → at least 1 fold
        assert result.n_folds >= 1
        assert model is not None

    def test_result_has_folds(self) -> None:
        from metatrade.ml.walk_forward import WalkForwardTrainer
        cfg = self._small_cfg()
        bars = make_bars(500)
        trainer = WalkForwardTrainer(cfg)
        result, _ = trainer.run(bars)
        assert len(result.folds) >= 1

    def test_mean_test_accuracy_in_range(self) -> None:
        from metatrade.ml.walk_forward import WalkForwardTrainer
        cfg = self._small_cfg()
        bars = make_bars(500)
        result, _ = WalkForwardTrainer(cfg).run(bars)
        assert 0.0 <= result.mean_test_accuracy <= 1.0

    def test_best_test_accuracy_ge_mean(self) -> None:
        from metatrade.ml.walk_forward import WalkForwardTrainer
        cfg = self._small_cfg()
        bars = make_bars(500)
        result, _ = WalkForwardTrainer(cfg).run(bars)
        assert result.best_test_accuracy >= result.mean_test_accuracy

    def test_best_fold_index_valid(self) -> None:
        from metatrade.ml.walk_forward import WalkForwardTrainer
        cfg = self._small_cfg()
        bars = make_bars(500)
        result, _ = WalkForwardTrainer(cfg).run(bars)
        if result.n_folds > 0:
            assert 0 <= result.best_fold_index < result.n_folds

    def test_fold_indices_sequential(self) -> None:
        from metatrade.ml.walk_forward import WalkForwardTrainer
        cfg = self._small_cfg()
        bars = make_bars(500)
        result, _ = WalkForwardTrainer(cfg).run(bars)
        for i, fold in enumerate(result.folds):
            assert fold.fold_index == i

    def test_fold_train_end_less_than_test_start(self) -> None:
        from metatrade.ml.walk_forward import WalkForwardTrainer
        cfg = self._small_cfg()
        bars = make_bars(500)
        result, _ = WalkForwardTrainer(cfg).run(bars)
        for fold in result.folds:
            assert fold.train_end <= fold.test_start

    def test_empty_bars_returns_no_folds(self) -> None:
        from metatrade.ml.walk_forward import WalkForwardTrainer
        cfg = self._small_cfg()
        result, model = WalkForwardTrainer(cfg).run([])
        assert result.n_folds == 0
        assert model is None
        assert result.mean_test_accuracy == 0.0
        assert result.best_test_accuracy == 0.0

    def test_insufficient_bars_returns_no_folds(self) -> None:
        from metatrade.ml.walk_forward import WalkForwardTrainer
        # Only 50 bars but train_window=200 → can't run any fold (need 250+)
        cfg = self._small_cfg()
        result, model = WalkForwardTrainer(cfg).run(make_bars(50))
        assert result.n_folds == 0
        assert model is None

    def test_multiple_folds_with_many_bars(self) -> None:
        from metatrade.ml.walk_forward import WalkForwardTrainer
        cfg = self._small_cfg(train_window_bars=200, test_window_bars=50, step_bars=50)
        bars = make_bars(600)
        result, _ = WalkForwardTrainer(cfg).run(bars)
        assert result.n_folds >= 2

    def test_default_config_used_when_none(self) -> None:
        from metatrade.ml.walk_forward import WalkForwardTrainer
        # Just ensure it doesn't crash with default config (very large windows)
        trainer = WalkForwardTrainer(None)
        assert trainer is not None

    def test_walk_forward_result_properties_empty(self) -> None:
        from metatrade.ml.walk_forward import WalkForwardResult
        r = WalkForwardResult()
        assert r.n_folds == 0
        assert r.mean_test_accuracy == 0.0
        assert r.best_test_accuracy == 0.0
