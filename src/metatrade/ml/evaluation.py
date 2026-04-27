"""Walk-forward evaluation utilities.

Produces structured ``MlEvaluationReport`` objects from raw predictions and
``WalkForwardResult`` data.  Also provides class distribution helpers that
satisfy the PR 4 merge criterion ("distribuzione classi documentata").

Usage pattern:

    result, model = WalkForwardTrainer(cfg).run(bars)
    labels = label_bars(bars, ...)

    # Document class distribution
    dist = WalkForwardEvaluator.class_distribution(labels)
    print(WalkForwardEvaluator.distribution_summary(labels))

    # Build per-fold reports (requires predictions — caller must generate them)
    report = WalkForwardEvaluator.fold_report(fold_id=0, predictions=preds, actuals=actuals)

    # Build holdout report directly from WalkForwardResult
    holdout = WalkForwardEvaluator.holdout_report(result)
"""

from __future__ import annotations

from metatrade.ml.contracts import MlEvaluationReport
from metatrade.ml.walk_forward import WalkForwardResult


class WalkForwardEvaluator:
    """Static helpers for structured evaluation of walk-forward results.

    All methods are static — no instance state is required.
    """

    @staticmethod
    def class_distribution(labels: list[int | None]) -> dict[int, int]:
        """Count samples per class, excluding ``None`` entries.

        Args:
            labels: Label series (as returned by ``label_bars`` or
                    ``label_bars_v2``).

        Returns:
            Dict mapping each present class label to its count.
            Only classes that appear at least once are included.
        """
        dist: dict[int, int] = {}
        for lbl in labels:
            if lbl is not None:
                dist[lbl] = dist.get(lbl, 0) + 1
        return dist

    @staticmethod
    def distribution_summary(labels: list[int | None]) -> str:
        """Human-readable class distribution string.

        Satisfies the PR 4 merge criterion for class distribution documentation.

        Args:
            labels: Label series.

        Returns:
            Multi-line string, e.g.::

                BUY  ( 1):  1200  (40.0%)
                HOLD ( 0):  1000  (33.3%)
                SELL (-1):   800  (26.7%)
                None       :   120  (skipped)
        """
        dist = WalkForwardEvaluator.class_distribution(labels)
        n_none = sum(1 for lbl in labels if lbl is None)
        total_valid = sum(dist.values())
        total = total_valid + n_none

        name = {1: "BUY ", 0: "HOLD", -1: "SELL"}
        lines: list[str] = []
        for cls in (1, 0, -1):
            count = dist.get(cls, 0)
            pct = 100.0 * count / total if total > 0 else 0.0
            label_name = name.get(cls, str(cls))
            lines.append(f"{label_name} ({cls:2d}): {count:6d}  ({pct:5.1f}%)")

        if n_none > 0:
            pct = 100.0 * n_none / total if total > 0 else 0.0
            lines.append(f"None      : {n_none:6d}  ({pct:5.1f}%) [skipped]")

        return "\n".join(lines)

    @staticmethod
    def fold_report(
        fold_id: int,
        predictions: list[int],
        actuals: list[int],
    ) -> MlEvaluationReport:
        """Build an ``MlEvaluationReport`` from raw predictions and labels.

        Args:
            fold_id:     0-based fold index (use -1 for the holdout set).
            predictions: Model predictions for each sample.
            actuals:     True labels for each sample.

        Returns:
            ``MlEvaluationReport`` with per-class precision, recall and
            overall accuracy.

        Raises:
            ValueError: If ``predictions`` and ``actuals`` have different
                        lengths or are empty.
        """
        if len(predictions) != len(actuals):
            raise ValueError(
                f"predictions ({len(predictions)}) and actuals ({len(actuals)}) "
                "must have equal length"
            )
        if not predictions:
            raise ValueError("predictions must not be empty")

        n = len(predictions)
        correct = sum(p == a for p, a in zip(predictions, actuals))
        accuracy = correct / n

        def _precision(cls: int) -> float:
            tp = sum(p == cls and a == cls for p, a in zip(predictions, actuals))
            denom = sum(p == cls for p in predictions)
            return tp / denom if denom else 0.0

        def _recall(cls: int) -> float:
            tp = sum(p == cls and a == cls for p, a in zip(predictions, actuals))
            denom = sum(a == cls for a in actuals)
            return tp / denom if denom else 0.0

        dist: dict[int, int] = {}
        for lbl in actuals:
            dist[lbl] = dist.get(lbl, 0) + 1

        return MlEvaluationReport(
            fold_id=fold_id,
            n_samples=n,
            accuracy=accuracy,
            buy_precision=_precision(1),
            sell_precision=_precision(-1),
            hold_precision=_precision(0),
            buy_recall=_recall(1),
            sell_recall=_recall(-1),
            class_distribution=dist,
        )

    @staticmethod
    def holdout_report(result: WalkForwardResult) -> MlEvaluationReport | None:
        """Build an ``MlEvaluationReport`` from a ``WalkForwardResult``'s holdout data.

        Uses the already-computed ``holdout_accuracy``, ``holdout_n_samples``,
        ``holdout_precision_by_class`` and ``holdout_recall_by_class`` fields
        stored by ``WalkForwardTrainer.run()``.  No raw predictions are needed.

        Args:
            result: Result from ``WalkForwardTrainer.run()``.

        Returns:
            ``MlEvaluationReport`` for the holdout set, or ``None`` if no
            holdout data is available (e.g. ``holdout_fraction == 0``).
        """
        if result.holdout_accuracy is None or result.holdout_n_samples == 0:
            return None

        return MlEvaluationReport(
            fold_id=-1,
            n_samples=result.holdout_n_samples,
            accuracy=result.holdout_accuracy,
            buy_precision=result.holdout_precision_by_class.get(1, 0.0),
            sell_precision=result.holdout_precision_by_class.get(-1, 0.0),
            hold_precision=result.holdout_precision_by_class.get(0, 0.0),
            buy_recall=result.holdout_recall_by_class.get(1, 0.0),
            sell_recall=result.holdout_recall_by_class.get(-1, 0.0),
            class_distribution={},
        )
