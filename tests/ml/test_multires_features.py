"""Tests for multi-resolution feature extraction and walk-forward integration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.ml.config import MLConfig
from metatrade.ml.features import (
    MIN_FEATURE_BARS,
    MIN_HIGHER_TF_BARS,
    MultiResFeatureVector,
    extract_features_multires,
)
from metatrade.ml.walk_forward import (
    WalkForwardTrainer,
    _in_session,
)
from metatrade.ml.walk_forward import (
    _align_htf_window as wf_align,
)

# ── helpers ──────────────────────────────────────────────────────────────────

BASE_TS = datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)


def D(v: float) -> Decimal:
    return Decimal(str(round(v, 5)))


def make_bar(
    close: float,
    ts: datetime,
    tf: Timeframe = Timeframe.M1,
) -> Bar:
    return Bar(
        symbol="EURUSD",
        timeframe=tf,
        timestamp_utc=ts,
        open=D(close),
        high=D(close + 0.001),
        low=D(close - 0.001),
        close=D(close),
        volume=Decimal("1000"),
    )


def make_bars(
    n: int,
    tf: Timeframe = Timeframe.M1,
    tf_minutes: int = 1,
    start_close: float = 1.10,
    step: float = 0.0001,
) -> list[Bar]:
    return [
        make_bar(
            round(start_close + i * step, 5),
            BASE_TS + timedelta(minutes=i * tf_minutes),
            tf,
        )
        for i in range(n)
    ]


def _test_cfg(**kwargs) -> MLConfig:
    defaults = dict(
        min_train_samples=10,
        max_iter=10,
        min_accuracy=0.5,
        train_window_bars=200,
        test_window_bars=50,
        step_bars=50,
    )
    defaults.update(kwargs)
    return MLConfig(**defaults)


# ── MultiResFeatureVector contract ───────────────────────────────────────────

class TestMultiResFeatureVector:
    def test_feature_names_length(self):
        assert len(MultiResFeatureVector.feature_names()) == 43

    def test_feature_names_start_with_m1_names(self):
        from metatrade.ml.features import FeatureVector
        names = MultiResFeatureVector.feature_names()
        base_names = FeatureVector.feature_names()
        assert names[:27] == base_names

    def test_feature_names_m5_prefix(self):
        names = MultiResFeatureVector.feature_names()
        m5_names = names[27:35]
        assert all(n.startswith("m5_") for n in m5_names)

    def test_feature_names_m15_prefix(self):
        names = MultiResFeatureVector.feature_names()
        m15_names = names[35:43]
        assert all(n.startswith("m15_") for n in m15_names)

    def test_to_list_length(self):
        bars_m1 = make_bars(MIN_FEATURE_BARS + 10)
        bars_m5 = make_bars(MIN_HIGHER_TF_BARS + 5, tf=Timeframe.M5, tf_minutes=5)
        bars_m15 = make_bars(MIN_HIGHER_TF_BARS + 5, tf=Timeframe.M15, tf_minutes=15)
        fv = extract_features_multires(bars_m1, bars_m5, bars_m15)
        assert fv is not None
        assert len(fv.to_list()) == 43

    def test_to_list_all_finite(self):
        import math
        bars_m1 = make_bars(MIN_FEATURE_BARS + 10)
        bars_m5 = make_bars(MIN_HIGHER_TF_BARS + 5, tf=Timeframe.M5, tf_minutes=5)
        bars_m15 = make_bars(MIN_HIGHER_TF_BARS + 5, tf=Timeframe.M15, tf_minutes=15)
        fv = extract_features_multires(bars_m1, bars_m5, bars_m15)
        assert fv is not None
        for v in fv.to_list():
            assert math.isfinite(v), f"Non-finite value: {v}"


# ── extract_features_multires ────────────────────────────────────────────────

class TestExtractFeaturesMultires:
    def test_returns_none_when_m1_insufficient(self):
        bars_m1 = make_bars(MIN_FEATURE_BARS - 1)
        bars_m5 = make_bars(MIN_HIGHER_TF_BARS + 10, tf=Timeframe.M5, tf_minutes=5)
        result = extract_features_multires(bars_m1, bars_m5)
        assert result is None

    def test_returns_vector_with_no_htf(self):
        bars_m1 = make_bars(MIN_FEATURE_BARS + 10)
        fv = extract_features_multires(bars_m1)
        assert isinstance(fv, MultiResFeatureVector)

    def test_neutral_m5_when_too_short(self):
        bars_m1 = make_bars(MIN_FEATURE_BARS + 10)
        short_m5 = make_bars(10, tf=Timeframe.M5, tf_minutes=5)
        fv = extract_features_multires(bars_m1, short_m5)
        assert fv is not None
        # Neutral values: rsi14=0.5, donchian_pos=0.5; ema dists=0.0
        assert fv.m5_ema9_dist == 0.0
        assert fv.m5_rsi14 == 0.5
        assert fv.m5_donchian_pos == 0.5

    def test_neutral_m15_when_none(self):
        bars_m1 = make_bars(MIN_FEATURE_BARS + 10)
        fv = extract_features_multires(bars_m1, m15_bars=None)
        assert fv is not None
        assert fv.m15_ema9_dist == 0.0
        assert fv.m15_rsi14 == 0.5

    def test_m1_base_matches_extract_features(self):
        from metatrade.ml.features import extract_features
        bars_m1 = make_bars(MIN_FEATURE_BARS + 10)
        ts = bars_m1[-1].timestamp_utc
        fv_base = extract_features(bars_m1, ts)
        fv_multi = extract_features_multires(bars_m1, timestamp_utc=ts)
        assert fv_base is not None
        assert fv_multi is not None
        assert fv_multi.returns_1 == pytest.approx(fv_base.returns_1, abs=1e-9)
        assert fv_multi.rsi14 == pytest.approx(fv_base.rsi14, abs=1e-9)
        assert fv_multi.hurst_short == pytest.approx(fv_base.hurst_short, abs=1e-9)

    def test_full_htf_produces_non_neutral_m5(self):
        bars_m1 = make_bars(MIN_FEATURE_BARS + 10)
        # M5 bars with a clear trend (strong upward step) to get non-zero ema dist
        bars_m5 = make_bars(
            MIN_HIGHER_TF_BARS + 10, tf=Timeframe.M5, tf_minutes=5, step=0.001
        )
        fv = extract_features_multires(bars_m1, bars_m5)
        assert fv is not None
        # With a strong upward step, EMA9 < close → positive ema9_dist
        assert fv.m5_ema9_dist != 0.0


# ── _align_htf_window (walk_forward module) ──────────────────────────────────

class TestAlignHtfWindow:
    def test_returns_empty_for_empty_bars(self):
        result = wf_align([], [], BASE_TS)
        assert result == []

    def test_returns_all_bars_when_all_before_ref(self):
        bars = make_bars(10, tf=Timeframe.M5, tf_minutes=5)
        ts_index = [b.timestamp_utc for b in bars]
        ref = BASE_TS + timedelta(minutes=50 + 100)
        result = wf_align(bars, ts_index, ref, window_size=MIN_HIGHER_TF_BARS)
        assert result == bars  # all 10 bars, less than window_size

    def test_respects_window_size(self):
        bars = make_bars(100, tf=Timeframe.M5, tf_minutes=5)
        ts_index = [b.timestamp_utc for b in bars]
        ref = bars[-1].timestamp_utc
        result = wf_align(bars, ts_index, ref, window_size=20)
        assert len(result) == 20

    def test_excludes_bars_after_ref(self):
        bars = make_bars(60, tf=Timeframe.M5, tf_minutes=5)
        ts_index = [b.timestamp_utc for b in bars]
        # ref_ts = timestamp of bar 30
        ref = bars[30].timestamp_utc
        result = wf_align(bars, ts_index, ref, window_size=100)
        # All returned bars must have timestamp <= ref
        assert all(b.timestamp_utc <= ref for b in result)
        assert len(result) == 31  # bars[0..30] inclusive


# ── session filter ────────────────────────────────────────────────────────────

class TestInSession:
    def _bar_at_hour(self, hour: int) -> Bar:
        ts = datetime(2024, 3, 4, hour, 0, 0, tzinfo=UTC)
        return make_bar(1.1, ts)

    def test_inside_session(self):
        assert _in_session(self._bar_at_hour(10), 7, 21) is True
        assert _in_session(self._bar_at_hour(7), 7, 21) is True

    def test_outside_session(self):
        assert _in_session(self._bar_at_hour(21), 7, 21) is False
        assert _in_session(self._bar_at_hour(3), 7, 21) is False

    def test_overnight_wrap_inside(self):
        # Session 22–6 UTC: hour 23 and hour 3 are inside
        assert _in_session(self._bar_at_hour(23), 22, 6) is True
        assert _in_session(self._bar_at_hour(3), 22, 6) is True

    def test_overnight_wrap_outside(self):
        assert _in_session(self._bar_at_hour(10), 22, 6) is False


# ── WalkForwardTrainer integration ───────────────────────────────────────────

class TestWalkForwardTrainerMultires:
    def _make_dataset(self, n_m1: int = 600) -> tuple[list[Bar], dict[str, list[Bar]]]:
        m1 = make_bars(n_m1)
        # M5: cover same period, step=5min
        m5 = make_bars(n_m1 // 5 + 10, tf=Timeframe.M5, tf_minutes=5)
        m15 = make_bars(n_m1 // 15 + 10, tf=Timeframe.M15, tf_minutes=15)
        return m1, {"M5": m5, "M15": m15}

    def test_run_produces_folds(self):
        bars, htf = self._make_dataset(600)
        cfg = _test_cfg(train_window_bars=200, test_window_bars=50, step_bars=50)
        trainer = WalkForwardTrainer(cfg)
        result, model = trainer.run(bars, higher_tf_bars=htf)
        assert result.n_folds >= 1

    def test_run_without_htf_still_works(self):
        bars, _ = self._make_dataset(600)
        cfg = _test_cfg(train_window_bars=200, test_window_bars=50, step_bars=50)
        trainer = WalkForwardTrainer(cfg)
        result, model = trainer.run(bars)
        assert result.n_folds >= 1

    def test_session_filter_reduces_samples(self):
        """Session filter should produce fewer training samples than no filter."""
        bars, _ = self._make_dataset(800)
        cfg_all = _test_cfg(train_window_bars=300, test_window_bars=100, step_bars=100)
        # Filter to only 2 hours of the day (very restrictive)
        cfg_filtered = _test_cfg(
            train_window_bars=300,
            test_window_bars=100,
            step_bars=100,
            session_filter_utc_start=10,
            session_filter_utc_end=12,
        )
        trainer_all = WalkForwardTrainer(cfg_all)
        trainer_filtered = WalkForwardTrainer(cfg_filtered)
        result_all, _ = trainer_all.run(bars)
        result_filtered, _ = trainer_filtered.run(bars)

        total_all = sum(f.n_test_samples for f in result_all.folds)
        total_filtered = sum(f.n_test_samples for f in result_filtered.folds)
        assert total_filtered < total_all


class TestSessionFilterInBuildDataset:
    def test_build_dataset_without_filter_includes_all_hours(self):
        bars = make_bars(300)
        cfg = _test_cfg()
        trainer = WalkForwardTrainer(cfg)
        from metatrade.ml.labels import label_bars
        labels = label_bars(bars, forward_bars=5, atr_threshold_mult=0.8)
        X, y = trainer._build_dataset(bars, labels, 0, len(bars))
        assert len(X) > 0

    def test_build_dataset_with_exact_session_excludes_bars(self):
        # Create bars all within a single UTC hour (22:xx) using 30-second steps
        # so the entire batch stays in hour=22, which is outside session 7–21.
        from datetime import timedelta as TD

        ts_start = datetime(2024, 3, 1, 22, 0, 0, tzinfo=UTC)
        bars = [
            make_bar(round(1.1 + i * 0.0001, 5), ts_start + TD(seconds=30 * i))
            for i in range(200)
        ]
        # All bars are at hour=22, which is outside [7, 21)
        cfg = _test_cfg(session_filter_utc_start=7, session_filter_utc_end=21)
        trainer = WalkForwardTrainer(cfg)
        from metatrade.ml.labels import label_bars
        labels = label_bars(bars, forward_bars=5, atr_threshold_mult=0.8)
        X, y = trainer._build_dataset(bars, labels, 0, len(bars))
        assert len(X) == 0
