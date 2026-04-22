"""Walk-forward training harness for ML classifiers.

Performance characteristics
---------------------------
Features are precomputed in a single O(n) pass before any fold begins.
Labels are computed in O(n) (ATR computed once on the full series).
Each fold's _build_from_cache() is O(fold_size) via index lookup — no
repeated indicator computation.  Total complexity: O(n + folds × fold_size).

Multi-resolution mode
---------------------
Pass ``higher_tf_bars`` to ``run()`` or ``precompute_features()`` to activate
43-feature MultiResFeatureVector extraction.  Precomputation aligns M5/M15
windows by timestamp via bisect (O(log n) per bar).

Session filter
--------------
Set ``MLConfig.session_filter_utc_start`` and ``session_filter_utc_end``
to drop thin-market bars.  Applied during _build_from_cache(), not during
precomputation, so the same feature cache works with any session setting.

Holdout evaluation
------------------
Set ``MLConfig.holdout_fraction > 0`` (e.g. 0.2) to reserve the last 20%
of bars as a final out-of-sample holdout.  Walk-forward trains only on the
first 80%; after all folds, the best fold's model is evaluated on the
holdout and the accuracy is stored in ``WalkForwardResult.holdout_accuracy``.

Auto-tune
---------
To sweep hyperparameters efficiently, call ``precompute_features()`` once,
then pass the returned cache to multiple ``run()`` calls via the
``feature_cache`` parameter.  Features only depend on price data, not on
label parameters (forward_bars, atr_threshold_mult), so the cache is reusable.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from metatrade.core.contracts.market import Bar
from metatrade.core.log import get_logger
from metatrade.core.utils.session import is_in_session
from metatrade.ml.classifier import ClassifierMetrics, MLClassifier
from metatrade.ml.config import MLConfig
from metatrade.ml.features import (
    MIN_FEATURE_BARS,
    MIN_HIGHER_TF_BARS,
    FeatureVector,
    MultiResFeatureVector,
    extract_features,
    extract_features_multires,
)
from metatrade.ml.labels import label_bars

log = get_logger(__name__)

# Type alias for a feature cache slot (None = insufficient bars or session-filtered)
_FVType = FeatureVector | MultiResFeatureVector
_CacheSlot = _FVType | None


@dataclass
class WalkForwardFold:
    """Results from a single walk-forward fold."""

    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_metrics: ClassifierMetrics
    test_accuracy: float
    n_test_samples: int


@dataclass
class WalkForwardResult:
    """Aggregated results from all walk-forward folds."""

    folds: list[WalkForwardFold] = field(default_factory=list)
    best_fold_index: int = -1
    holdout_accuracy: float | None = None
    holdout_n_samples: int = 0
    holdout_precision_by_class: dict[int, float] = field(default_factory=dict)
    holdout_recall_by_class: dict[int, float] = field(default_factory=dict)

    @property
    def mean_test_accuracy(self) -> float:
        if not self.folds:
            return 0.0
        return sum(f.test_accuracy for f in self.folds) / len(self.folds)

    @property
    def best_test_accuracy(self) -> float:
        if not self.folds:
            return 0.0
        return max(f.test_accuracy for f in self.folds)

    @property
    def holdout_signal_precision(self) -> float | None:
        """Average precision on BUY (1) and SELL (-1) — the metric that matters for trading."""
        values = [
            self.holdout_precision_by_class[c]
            for c in (1, -1)
            if c in self.holdout_precision_by_class
        ]
        return sum(values) / len(values) if values else None

    @property
    def n_folds(self) -> int:
        return len(self.folds)


def _align_htf_window(
    bars: list[Bar],
    ts_index: list[datetime],
    ref_ts: datetime,
    window_size: int = MIN_HIGHER_TF_BARS,
) -> list[Bar]:
    """Return the last ``window_size`` higher-TF bars with timestamp <= ref_ts."""
    idx = bisect.bisect_right(ts_index, ref_ts)
    start = max(0, idx - window_size)
    return bars[start:idx]


def _in_session(bar: Bar, start_utc: int, end_utc: int) -> bool:
    """True if bar's UTC hour is in [start_utc, end_utc). Handles overnight wrap."""
    return is_in_session(bar.timestamp_utc, start_utc, end_utc)


class WalkForwardTrainer:
    """Walk-forward training coordinator with O(n) feature precomputation.

    Args:
        config: MLConfig controlling window sizes, model hyper-params, etc.
    """

    def __init__(self, config: MLConfig | None = None) -> None:
        self._config = config or MLConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def precompute_features(
        self,
        bars: list[Bar],
        higher_tf_bars: dict[str, list[Bar]] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[_CacheSlot]:
        """Build a feature cache for all bars in one O(n) pass.

        The cache is reusable across multiple ``run()`` calls with different
        label parameters (forward_bars, atr_threshold_mult) because features
        do not depend on those parameters.

        Args:
            bars:           Full bar series (oldest first).
            higher_tf_bars: Optional higher-TF bars for multi-resolution mode.
            on_progress:    Optional callback(current_idx, total) fired every 5%.

        Returns:
            List of feature vectors (or None) aligned to bars, length == len(bars).
        """
        n = len(bars)
        htf_ts_indices: dict[str, list[datetime]] = {}
        if higher_tf_bars:
            for tf_name, tf_bars in higher_tf_bars.items():
                htf_ts_indices[tf_name] = [b.timestamp_utc for b in tf_bars]

        m5_bars = (higher_tf_bars or {}).get("M5", [])
        m15_bars = (higher_tf_bars or {}).get("M15", [])
        m5_ts = htf_ts_indices.get("M5", [])
        m15_ts = htf_ts_indices.get("M15", [])
        use_multires = bool(higher_tf_bars)

        report_every = max(1, n // 20)  # every 5%
        cache: list[_CacheSlot] = [None] * n

        for i in range(n):
            if on_progress is not None and i % report_every == 0:
                on_progress(i, n)

            window_start = max(0, i - MIN_FEATURE_BARS + 1)
            window = bars[window_start : i + 1]

            if use_multires:
                ref_ts = bars[i].timestamp_utc
                m5_win = _align_htf_window(m5_bars, m5_ts, ref_ts) if m5_bars else []
                m15_win = _align_htf_window(m15_bars, m15_ts, ref_ts) if m15_bars else []
                cache[i] = extract_features_multires(window, m5_win, m15_win, ref_ts)
            else:
                cache[i] = extract_features(window, bars[i].timestamp_utc)

        if on_progress is not None:
            on_progress(n, n)

        return cache

    def run(
        self,
        bars: list[Bar],
        *,
        on_fold_complete: Callable[[WalkForwardFold], None] | None = None,
        higher_tf_bars: dict[str, list[Bar]] | None = None,
        feature_cache: list[_CacheSlot] | None = None,
        on_precompute_progress: Callable[[int, int], None] | None = None,
    ) -> tuple[WalkForwardResult, MLClassifier | None]:
        """Execute walk-forward training.

        Args:
            bars:                   Full historical bar series (oldest first).
            on_fold_complete:       Called after each fold is appended to result.
            higher_tf_bars:         Higher-TF bars for multi-resolution features.
                                    Ignored when feature_cache is provided.
            feature_cache:          Pre-built feature cache from precompute_features().
                                    When provided, no feature extraction is performed
                                    during training — pass this in auto-tune loops
                                    to avoid redundant computation.
            on_precompute_progress: Progress callback(current, total) for the
                                    precomputation phase (only called when
                                    feature_cache is None).

        Returns:
            (result, final_model) — WalkForwardResult + last trained classifier.
        """
        cfg = self._config
        n = len(bars)

        if n == 0:
            return WalkForwardResult(), None

        # ── Feature cache ──────────────────────────────────────────────────────
        if feature_cache is None:
            feature_cache = self.precompute_features(
                bars,
                higher_tf_bars=higher_tf_bars,
                on_progress=on_precompute_progress,
            )

        # ── Labels ─────────────────────────────────────────────────────────────
        all_labels = label_bars(
            bars,
            forward_bars=cfg.forward_bars,
            atr_threshold_mult=cfg.atr_threshold_mult,
        )

        # ── Holdout split ──────────────────────────────────────────────────────
        holdout_start = n
        if cfg.holdout_fraction > 0:
            holdout_start = max(
                cfg.train_window_bars + cfg.test_window_bars,
                int(n * (1.0 - cfg.holdout_fraction)),
            )

        result = WalkForwardResult()
        final_model: MLClassifier | None = None
        best_model: MLClassifier | None = None
        best_accuracy = 0.0

        fold_index = 0
        train_end = cfg.train_window_bars

        while train_end + cfg.embargo_bars + cfg.test_window_bars <= holdout_start:
            # embargo_bars: skip this many bars between train and test to prevent
            # leakage from autocorrelated consecutive bars (should be >= forward_bars).
            test_start = train_end + cfg.embargo_bars
            test_end = min(test_start + cfg.test_window_bars, holdout_start)

            train_X, train_y = self._build_from_cache(
                feature_cache, all_labels, bars, 0, train_end
            )
            if len(train_X) < cfg.min_train_samples:
                train_end += cfg.step_bars
                continue

            classifier = MLClassifier(cfg)
            try:
                train_metrics = classifier.fit(train_X, train_y)
            except (ValueError, ImportError):
                train_end += cfg.step_bars
                continue

            test_X, test_y = self._build_from_cache(
                feature_cache, all_labels, bars, test_start, test_end
            )
            n_test = len(test_X)
            test_accuracy = 0.0
            if n_test > 0:
                correct = sum(
                    1
                    for fv, lbl in zip(test_X, test_y, strict=False)
                    if _safe_predict(classifier, fv) == lbl
                )
                test_accuracy = correct / n_test

            fold = WalkForwardFold(
                fold_index=fold_index,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_metrics=train_metrics,
                test_accuracy=test_accuracy,
                n_test_samples=n_test,
            )
            result.folds.append(fold)
            final_model = classifier

            if on_fold_complete is not None:
                on_fold_complete(fold)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                result.best_fold_index = fold_index
                best_model = classifier

            fold_index += 1
            train_end += cfg.step_bars

        # ── Holdout evaluation ─────────────────────────────────────────────────
        if holdout_start < n and best_model is not None:
            holdout_X, holdout_y = self._build_from_cache(
                feature_cache, all_labels, bars, holdout_start, n
            )
            n_holdout = len(holdout_X)
            if n_holdout > 0:
                preds = [_safe_predict(best_model, fv) for fv in holdout_X]
                correct = sum(p == lbl for p, lbl in zip(preds, holdout_y))
                result.holdout_accuracy = correct / n_holdout
                result.holdout_n_samples = n_holdout

                # Per-class precision and recall
                classes = {1, -1, 0}
                tp: dict[int, int] = {c: 0 for c in classes}
                fp: dict[int, int] = {c: 0 for c in classes}
                fn: dict[int, int] = {c: 0 for c in classes}
                for pred, actual in zip(preds, holdout_y):
                    if pred == actual:
                        tp[pred] = tp.get(pred, 0) + 1
                    else:
                        fp[pred] = fp.get(pred, 0) + 1
                        fn[actual] = fn.get(actual, 0) + 1
                for c in classes:
                    denom_p = tp[c] + fp[c]
                    denom_r = tp[c] + fn[c]
                    result.holdout_precision_by_class[c] = tp[c] / denom_p if denom_p else 0.0
                    result.holdout_recall_by_class[c] = tp[c] / denom_r if denom_r else 0.0

        return result, final_model

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_from_cache(
        self,
        cache: list[_CacheSlot],
        labels: list[int | None],
        bars: list[Bar],
        start: int,
        end: int,
    ) -> tuple[list[Any], list[int]]:
        """Build (X, y) by looking up pre-computed features from the cache.

        Session filtering (from MLConfig) is applied here so the same cache
        can serve different session configurations.
        """
        cfg = self._config
        session_start = cfg.session_filter_utc_start
        session_end = cfg.session_filter_utc_end
        use_session = session_start is not None and session_end is not None

        X: list[Any] = []
        y: list[int] = []

        for i in range(start, end):
            lbl = labels[i]
            if lbl is None:
                continue
            if use_session:
                assert session_start is not None and session_end is not None
                if not _in_session(bars[i], session_start, session_end):
                    continue
            fv = cache[i]
            if fv is not None:
                X.append(fv)
                y.append(lbl)

        return X, y

    # ── Backward-compatible _build_dataset (kept for tests) ───────────────────

    def _build_dataset(
        self,
        bars: list[Bar],
        labels: list[int | None],
        start: int,
        end: int,
        *,
        higher_tf_bars: dict[str, list[Bar]] | None = None,
        htf_ts_indices: dict[str, list[datetime]] | None = None,
    ) -> tuple[list[_FVType], list[int]]:
        """Legacy per-bar feature extraction (used only in tests).

        Prefer precompute_features() + _build_from_cache() for production use.
        """
        cfg = self._config
        session_start = cfg.session_filter_utc_start
        session_end = cfg.session_filter_utc_end
        use_session = session_start is not None and session_end is not None
        use_multires = bool(higher_tf_bars and htf_ts_indices)

        m5_bars = (higher_tf_bars or {}).get("M5", [])
        m15_bars = (higher_tf_bars or {}).get("M15", [])
        m5_ts = (htf_ts_indices or {}).get("M5", [])
        m15_ts = (htf_ts_indices or {}).get("M15", [])

        X: list[_FVType] = []
        y: list[int] = []

        for i in range(start, end):
            lbl = labels[i]
            if lbl is None:
                continue
            bar = bars[i]
            if use_session:
                assert session_start is not None and session_end is not None
                if not _in_session(bar, session_start, session_end):
                    continue
            window_start = max(0, i - MIN_FEATURE_BARS + 1)
            window = bars[window_start : i + 1]
            if use_multires:
                ref_ts = bar.timestamp_utc
                m5_win = _align_htf_window(m5_bars, m5_ts, ref_ts) if m5_bars else []
                m15_win = _align_htf_window(m15_bars, m15_ts, ref_ts) if m15_bars else []
                fv: _FVType | None = extract_features_multires(
                    window, m5_win, m15_win, ref_ts
                )
            else:
                fv = extract_features(window, bar.timestamp_utc)
            if fv is not None:
                X.append(fv)
                y.append(lbl)

        return X, y


def _safe_predict(classifier: MLClassifier, fv: _FVType) -> int:
    """Return predicted direction, or 0 (HOLD) on any error."""
    try:
        pred, _ = classifier.predict(fv)
        return pred
    except Exception as exc:
        log.warning("ml_safe_predict_error", exc_type=type(exc).__name__, error=str(exc))
        return 0
