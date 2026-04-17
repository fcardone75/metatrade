"""Walk-forward training pipeline for the exit profile selector classifier.

Trains a multiclass classifier (one class per profile_id) on the dataset
produced by ExitProfileDatasetBuilder using a time-ordered walk-forward scheme
to prevent look-ahead bias.

ML backend
----------
Uses scikit-learn's HistGradientBoostingClassifier by default.
If lightgbm is installed, LGBMClassifier is used instead (better for
categorical features and faster on larger datasets).
Both expose the same sklearn-compatible API.

Walk-forward scheme
-------------------
The dataset rows are sorted by timestamp.  Each fold trains on the first N
rows and evaluates on the next K rows.  The final model is trained on all
data after folds are evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from metatrade.ml.exit_profile_dataset_builder import DatasetRow
from metatrade.ml.exit_profile_feature_builder import ExitProfileFeatureBuilder

log = structlog.get_logger(__name__)


@dataclass
class ExitProfileFoldMetrics:
    """Evaluation metrics from one walk-forward fold."""

    fold: int
    n_train: int
    n_test: int
    accuracy: float
    per_profile_accuracy: dict[str, float] = field(default_factory=dict)


@dataclass
class ExitProfileTrainingResult:
    """Full output of a training run."""

    model: Any                          # trained sklearn-compatible classifier
    profile_ids: list[str]              # class labels (profile_id strings)
    feature_names: tuple[str, ...]      # ordered feature columns
    folds: list[ExitProfileFoldMetrics]
    mean_accuracy: float
    n_total_samples: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_classifier(random_seed: int = 42) -> Any:
    """Instantiate the best available gradient-boosted classifier."""
    try:
        import lightgbm as lgb  # type: ignore[import]  # pragma: no cover
        return lgb.LGBMClassifier(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            random_state=random_seed,
            verbose=-1,
        )
    except ImportError:
        pass
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        random_state=random_seed,
        early_stopping=False,
    )


def _rows_to_arrays(
    rows: list[DatasetRow],
) -> tuple[np.ndarray, list[str]]:
    """Convert DatasetRow list to (X, y) arrays.

    Feature order is taken from ExitProfileFeatureBuilder.FEATURE_NAMES so
    that every training run and inference call uses the same column order.
    """
    if not rows:
        return np.zeros((0, len(ExitProfileFeatureBuilder.FEATURE_NAMES))), []

    feature_keys = list(ExitProfileFeatureBuilder.FEATURE_NAMES)
    X = np.array(
        [[r.features.get(k, 0.0) for k in feature_keys] for r in rows],
        dtype=float,
    )
    y = [r.label for r in rows]
    return X, y


# ── Trainer ───────────────────────────────────────────────────────────────────

class ExitProfileTrainer:
    """Trains an exit profile classifier with walk-forward validation.

    Args:
        n_splits:        Number of walk-forward test folds (default 5).
        min_train_rows:  Minimum training rows per fold (default 50).
        min_test_rows:   Minimum test rows per fold (default 10).
        random_seed:     PRNG seed for reproducibility (default 42).
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_rows: int = 50,
        min_test_rows: int = 10,
        random_seed: int = 42,
    ) -> None:
        self._n_splits = n_splits
        self._min_train_rows = min_train_rows
        self._min_test_rows = min_test_rows
        self._seed = random_seed

    def train(self, rows: list[DatasetRow]) -> ExitProfileTrainingResult | None:
        """Train the classifier with walk-forward validation.

        Args:
            rows: Dataset rows from ExitProfileDatasetBuilder, sorted by time.

        Returns:
            ExitProfileTrainingResult with the final model, or None when there
            is insufficient data.
        """
        if len(rows) < self._min_train_rows:
            log.warning(
                "exit_profile_insufficient_training_data",
                n_rows=len(rows),
                min_required=self._min_train_rows,
            )
            return None

        # Temporal sort (dataset builder may produce rows out of order)
        rows_sorted = sorted(rows, key=lambda r: r.timestamp_utc)
        X_all, y_all = _rows_to_arrays(rows_sorted)
        n = len(rows_sorted)
        profile_ids = sorted(set(y_all))

        folds: list[ExitProfileFoldMetrics] = []
        fold_size = max(self._min_test_rows, n // (self._n_splits + 1))

        for fold_idx in range(self._n_splits - 1, -1, -1):
            test_end   = n - fold_idx * fold_size
            test_start = test_end - fold_size
            train_end  = test_start

            if train_end < self._min_train_rows:
                continue
            if test_end - test_start < self._min_test_rows:
                continue

            X_tr = X_all[:train_end]
            y_tr = y_all[:train_end]
            X_te = X_all[test_start:test_end]
            y_te = y_all[test_start:test_end]

            clf = _make_classifier(self._seed)
            try:
                clf.fit(X_tr, y_tr)
                preds = list(clf.predict(X_te))
                acc = float(np.mean(np.array(preds) == np.array(y_te)))

                per_profile: dict[str, float] = {}
                for pid in set(y_te):
                    mask = [yt == pid for yt in y_te]
                    if any(mask):
                        per_profile[pid] = float(
                            np.mean(
                                np.array(preds)[mask] == np.array(y_te)[mask]
                            )
                        )

                folds.append(ExitProfileFoldMetrics(
                    fold=fold_idx,
                    n_train=train_end,
                    n_test=test_end - test_start,
                    accuracy=acc,
                    per_profile_accuracy=per_profile,
                ))
            except Exception as exc:
                log.warning(
                    "exit_profile_fold_failed", fold=fold_idx, error=str(exc)
                )

        if not folds:
            log.warning("exit_profile_no_valid_folds", n_rows=n)
            return None

        # Final model on all data
        final_clf = _make_classifier(self._seed)
        try:
            final_clf.fit(X_all, y_all)
        except Exception as exc:
            log.error("exit_profile_final_train_failed", error=str(exc))
            return None

        mean_acc = float(np.mean([f.accuracy for f in folds]))
        log.info(
            "exit_profile_training_complete",
            n_samples=n,
            n_folds=len(folds),
            mean_accuracy=round(mean_acc, 4),
            profile_ids=profile_ids,
        )

        return ExitProfileTrainingResult(
            model=final_clf,
            profile_ids=profile_ids,
            feature_names=ExitProfileFeatureBuilder.FEATURE_NAMES,
            folds=folds,
            mean_accuracy=mean_acc,
            n_total_samples=n,
        )
