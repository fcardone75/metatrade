"""Walk-forward training harness for ML classifiers.

Walk-forward (a.k.a. "expanding window" or "rolling window") training is
the standard validation method for time-series ML models in finance.
It prevents look-ahead bias by training only on past data to predict
future labels, then sliding the window forward in time.

Window layout (expanding variant):
    ┌─────────────────┬───────┐
    │   train[0..N-1] │ test  │
    └─────────────────┴───────┘
    ┌─────────────────────┬───────┐
    │   train[0..N+step]  │ test  │
    └─────────────────────┴───────┘

Each fold trains on all history up to the fold boundary, then evaluates
on the next `test_window_bars` bars.

Multi-resolution mode
---------------------
Pass ``higher_tf_bars`` to ``run()`` to activate multi-resolution feature
extraction.  The dict maps timeframe names (e.g. ``"M5"``, ``"M15"``) to
sorted bar lists.  For each primary bar the trainer aligns the higher-TF
windows by timestamp (bisect), then calls ``extract_features_multires()``
to produce a 43-feature ``MultiResFeatureVector`` instead of the standard
27-feature ``FeatureVector``.

Session filter
--------------
Set ``MLConfig.session_filter_utc_start`` and ``session_filter_utc_end``
(both as UTC hours, e.g. 7 and 21) to drop thin-market bars from training
and test datasets.  The filter applies to primary bar timestamps only;
higher-TF alignment windows are unaffected.
"""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from metatrade.core.contracts.market import Bar
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


@dataclass
class WalkForwardFold:
    """Results from a single walk-forward fold."""

    fold_index: int
    train_start: int       # Bar index (absolute) where training window starts
    train_end: int         # Bar index (absolute) where training window ends
    test_start: int        # Bar index (absolute) where test window starts
    test_end: int          # Bar index (absolute) where test window ends
    train_metrics: ClassifierMetrics
    test_accuracy: float
    n_test_samples: int


@dataclass
class WalkForwardResult:
    """Aggregated results from all walk-forward folds."""

    folds: list[WalkForwardFold] = field(default_factory=list)
    best_fold_index: int = -1

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
    def n_folds(self) -> int:
        return len(self.folds)


def _align_htf_window(
    bars: list[Bar],
    ts_index: list[datetime],
    ref_ts: datetime,
    window_size: int = MIN_HIGHER_TF_BARS,
) -> list[Bar]:
    """Return the last ``window_size`` higher-TF bars with timestamp <= ref_ts.

    Uses bisect for O(log n) alignment on the pre-built sorted timestamp index.
    """
    idx = bisect.bisect_right(ts_index, ref_ts)
    start = max(0, idx - window_size)
    return bars[start:idx]


def _in_session(bar: Bar, start_utc: int, end_utc: int) -> bool:
    """Return True if bar's UTC hour falls within [start_utc, end_utc).

    Handles same-day ranges (e.g. 7–21) and overnight ranges (e.g. 22–6).
    """
    h = bar.timestamp_utc.hour
    if start_utc < end_utc:
        return start_utc <= h < end_utc
    return h >= start_utc or h < end_utc


class WalkForwardTrainer:
    """Walk-forward training coordinator.

    Produces a sequence of trained MLClassifier instances — one per fold.
    The caller is responsible for selecting the model to deploy (e.g. the
    most recent fold, or the fold with the highest test accuracy).

    Args:
        config: MLConfig controlling window sizes, model hyper-params, etc.
    """

    def __init__(self, config: MLConfig | None = None) -> None:
        self._config = config or MLConfig()

    def run(
        self,
        bars: list[Bar],
        *,
        on_fold_complete: Callable[[WalkForwardFold], None] | None = None,
        higher_tf_bars: dict[str, list[Bar]] | None = None,
    ) -> tuple[WalkForwardResult, MLClassifier | None]:
        """Execute walk-forward training over the bar series.

        Args:
            bars: Full historical bar series (oldest first).
            on_fold_complete: Optional hook invoked after each successful fold
                (after append to ``result.folds``).
            higher_tf_bars: Optional mapping of timeframe name → sorted bar
                list for multi-resolution feature extraction.  When provided,
                ``extract_features_multires()`` is used, producing 43-feature
                vectors.  Typical keys: ``"M5"``, ``"M15"``.

        Returns:
            (result, final_model) where:
                result      — WalkForwardResult with fold metrics
                final_model — The last trained MLClassifier, or None if
                              no fold produced sufficient data.
        """
        cfg = self._config
        result = WalkForwardResult()
        final_model: MLClassifier | None = None
        best_accuracy = 0.0

        all_labels = label_bars(
            bars,
            forward_bars=cfg.forward_bars,
            atr_threshold_mult=cfg.atr_threshold_mult,
        )

        # Pre-build timestamp indices for O(log n) HTF alignment per bar.
        htf_ts_indices: dict[str, list[datetime]] = {}
        if higher_tf_bars:
            for tf_name, tf_bars in higher_tf_bars.items():
                htf_ts_indices[tf_name] = [b.timestamp_utc for b in tf_bars]

        n = len(bars)
        fold_index = 0
        train_end = cfg.train_window_bars  # exclusive upper bound

        while train_end + cfg.test_window_bars <= n:
            test_start = train_end
            test_end = min(test_start + cfg.test_window_bars, n)

            train_X, train_y = self._build_dataset(
                bars, all_labels, 0, train_end,
                higher_tf_bars=higher_tf_bars,
                htf_ts_indices=htf_ts_indices,
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

            test_X, test_y = self._build_dataset(
                bars, all_labels, test_start, test_end,
                higher_tf_bars=higher_tf_bars,
                htf_ts_indices=htf_ts_indices,
            )
            test_accuracy = 0.0
            n_test = len(test_X)
            if n_test > 0:
                correct = 0
                for fv, lbl in zip(test_X, test_y):
                    try:
                        pred, _ = classifier.predict(fv)
                        if pred == lbl:
                            correct += 1
                    except RuntimeError:
                        pass
                test_accuracy = correct / n_test if n_test > 0 else 0.0

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

            fold_index += 1
            train_end += cfg.step_bars

        return result, final_model

    def _build_dataset(
        self,
        bars: list[Bar],
        labels: list[int | None],
        start: int,
        end: int,
        *,
        higher_tf_bars: dict[str, list[Bar]] | None = None,
        htf_ts_indices: dict[str, list[datetime]] | None = None,
    ) -> tuple[list[FeatureVector | MultiResFeatureVector], list[int]]:
        """Build (X, y) for bars[start:end], skipping None labels.

        When higher_tf_bars is provided, aligns M5/M15 context for each
        primary bar and calls extract_features_multires().

        Session filtering (from MLConfig) drops thin-market bars before
        feature extraction.
        """
        cfg = self._config
        session_start = cfg.session_filter_utc_start
        session_end = cfg.session_filter_utc_end
        use_session_filter = session_start is not None and session_end is not None
        use_multires = bool(higher_tf_bars and htf_ts_indices)

        m5_bars = (higher_tf_bars or {}).get("M5", [])
        m15_bars = (higher_tf_bars or {}).get("M15", [])
        m5_ts = (htf_ts_indices or {}).get("M5", [])
        m15_ts = (htf_ts_indices or {}).get("M15", [])

        X: list[FeatureVector | MultiResFeatureVector] = []
        y: list[int] = []

        for i in range(start, end):
            lbl = labels[i]
            if lbl is None:
                continue

            bar = bars[i]

            if use_session_filter:
                assert session_start is not None and session_end is not None
                if not _in_session(bar, session_start, session_end):
                    continue

            window_start = max(0, i - MIN_FEATURE_BARS + 1)
            window = bars[window_start : i + 1]

            if use_multires:
                ref_ts = bar.timestamp_utc
                m5_win = _align_htf_window(m5_bars, m5_ts, ref_ts) if m5_bars else []
                m15_win = _align_htf_window(m15_bars, m15_ts, ref_ts) if m15_bars else []
                fv: FeatureVector | MultiResFeatureVector | None = (
                    extract_features_multires(window, m5_win, m15_win, ref_ts)
                )
            else:
                fv = extract_features(window, bar.timestamp_utc)

            if fv is not None:
                X.append(fv)
                y.append(lbl)

        return X, y
