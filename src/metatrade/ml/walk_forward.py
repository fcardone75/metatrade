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
"""

from __future__ import annotations

from dataclasses import dataclass, field

from metatrade.core.contracts.market import Bar
from metatrade.ml.classifier import MLClassifier, ClassifierMetrics
from metatrade.ml.config import MLConfig
from metatrade.ml.features import extract_features, MIN_FEATURE_BARS, FeatureVector
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

    def run(self, bars: list[Bar]) -> tuple[WalkForwardResult, MLClassifier | None]:
        """Execute walk-forward training over the bar series.

        Args:
            bars: Full historical bar series (oldest first).

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

        # Labels for the entire series (uses ATR-adaptive thresholds)
        all_labels = label_bars(
            bars,
            forward_bars=cfg.forward_bars,
            atr_threshold_mult=cfg.atr_threshold_mult,
        )

        n = len(bars)
        fold_index = 0
        train_end = cfg.train_window_bars  # exclusive upper bound

        while train_end + cfg.test_window_bars <= n:
            test_start = train_end
            test_end = min(test_start + cfg.test_window_bars, n)

            # Build train dataset
            train_X, train_y = self._build_dataset(bars, all_labels, 0, train_end)
            if len(train_X) < cfg.min_train_samples:
                train_end += cfg.step_bars
                continue

            # Fit model
            classifier = MLClassifier(cfg)
            try:
                train_metrics = classifier.fit(train_X, train_y)
            except (ValueError, ImportError):
                train_end += cfg.step_bars
                continue

            # Evaluate on test window
            test_X, test_y = self._build_dataset(bars, all_labels, test_start, test_end)
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
    ) -> tuple[list[FeatureVector], list[int]]:
        """Build (X, y) for bars[start:end], skipping None labels."""
        X: list[FeatureVector] = []
        y: list[int] = []
        for i in range(start, end):
            lbl = labels[i]
            if lbl is None:
                continue
            # Feature extraction requires MIN_FEATURE_BARS history
            window_start = max(0, i - MIN_FEATURE_BARS + 1)
            window = bars[window_start : i + 1]
            fv = extract_features(window)
            if fv is not None:
                X.append(fv)
                y.append(lbl)
        return X, y
