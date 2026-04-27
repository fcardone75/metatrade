"""Tests for ml/evaluation.py — WalkForwardEvaluator."""

from __future__ import annotations

import pytest

from metatrade.ml.contracts import MlEvaluationReport
from metatrade.ml.evaluation import WalkForwardEvaluator
from metatrade.ml.walk_forward import WalkForwardResult


# ---------------------------------------------------------------------------
# class_distribution()
# ---------------------------------------------------------------------------

class TestClassDistribution:
    def test_counts_correctly(self) -> None:
        labels = [1, 1, 0, -1, 1, None, 0]
        dist = WalkForwardEvaluator.class_distribution(labels)
        assert dist[1] == 3
        assert dist[0] == 2
        assert dist[-1] == 1
        assert None not in dist

    def test_excludes_none(self) -> None:
        labels = [None, None, 1]
        dist = WalkForwardEvaluator.class_distribution(labels)
        assert sum(dist.values()) == 1

    def test_empty_series(self) -> None:
        assert WalkForwardEvaluator.class_distribution([]) == {}

    def test_all_none(self) -> None:
        assert WalkForwardEvaluator.class_distribution([None, None]) == {}

    def test_single_class(self) -> None:
        labels = [0, 0, 0, None]
        dist = WalkForwardEvaluator.class_distribution(labels)
        assert dist == {0: 3}


# ---------------------------------------------------------------------------
# distribution_summary()
# ---------------------------------------------------------------------------

class TestDistributionSummary:
    def test_returns_string(self) -> None:
        labels = [1, 0, -1, None]
        s = WalkForwardEvaluator.distribution_summary(labels)
        assert isinstance(s, str)
        assert len(s) > 0

    def test_contains_class_names(self) -> None:
        labels = [1, 0, -1]
        s = WalkForwardEvaluator.distribution_summary(labels)
        assert "BUY" in s
        assert "HOLD" in s
        assert "SELL" in s

    def test_contains_none_line_when_present(self) -> None:
        labels = [1, None]
        s = WalkForwardEvaluator.distribution_summary(labels)
        assert "None" in s or "skip" in s.lower()

    def test_no_none_line_when_absent(self) -> None:
        labels = [1, 0, -1]
        s = WalkForwardEvaluator.distribution_summary(labels)
        assert "skip" not in s.lower()

    def test_empty_labels(self) -> None:
        s = WalkForwardEvaluator.distribution_summary([])
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# fold_report()
# ---------------------------------------------------------------------------

class TestFoldReport:
    def test_returns_ml_evaluation_report(self) -> None:
        preds = [1, 0, -1, 1]
        actuals = [1, 0, -1, 0]
        report = WalkForwardEvaluator.fold_report(0, preds, actuals)
        assert isinstance(report, MlEvaluationReport)

    def test_accuracy_correct(self) -> None:
        preds   = [1, 1, 0, -1]
        actuals = [1, 0, 0, -1]
        report = WalkForwardEvaluator.fold_report(0, preds, actuals)
        assert abs(report.accuracy - 0.75) < 1e-9

    def test_perfect_accuracy(self) -> None:
        labels = [1, 0, -1, 1, 0]
        report = WalkForwardEvaluator.fold_report(0, labels, labels)
        assert report.accuracy == 1.0

    def test_buy_precision_correct(self) -> None:
        # 2 predictions of BUY, 1 correct
        preds   = [1, 1, 0]
        actuals = [1, 0, 0]
        report = WalkForwardEvaluator.fold_report(0, preds, actuals)
        assert abs(report.buy_precision - 0.5) < 1e-9

    def test_sell_recall_correct(self) -> None:
        # 2 actual SELL, 1 predicted correctly
        preds   = [1, -1, 0]
        actuals = [-1, -1, 0]
        report = WalkForwardEvaluator.fold_report(0, preds, actuals)
        assert abs(report.sell_recall - 0.5) < 1e-9

    def test_class_distribution_in_report(self) -> None:
        actuals = [1, 1, 0, -1]
        report = WalkForwardEvaluator.fold_report(0, actuals, actuals)
        assert report.class_distribution[1] == 2
        assert report.class_distribution[0] == 1
        assert report.class_distribution[-1] == 1

    def test_fold_id_preserved(self) -> None:
        preds = [1, 0]
        report = WalkForwardEvaluator.fold_report(7, preds, preds)
        assert report.fold_id == 7

    def test_n_samples_correct(self) -> None:
        preds = [1, 0, -1]
        report = WalkForwardEvaluator.fold_report(0, preds, preds)
        assert report.n_samples == 3

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            WalkForwardEvaluator.fold_report(0, [1, 0], [1])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            WalkForwardEvaluator.fold_report(0, [], [])

    def test_precision_zero_when_class_not_predicted(self) -> None:
        preds   = [0, 0, 0]
        actuals = [1, -1, 0]
        report = WalkForwardEvaluator.fold_report(0, preds, actuals)
        assert report.buy_precision == 0.0
        assert report.sell_precision == 0.0


# ---------------------------------------------------------------------------
# holdout_report()
# ---------------------------------------------------------------------------

class TestHoldoutReport:
    def test_none_when_no_holdout(self) -> None:
        result = WalkForwardResult()
        assert WalkForwardEvaluator.holdout_report(result) is None

    def test_none_when_zero_samples(self) -> None:
        result = WalkForwardResult()
        result.holdout_accuracy = 0.60
        result.holdout_n_samples = 0
        assert WalkForwardEvaluator.holdout_report(result) is None

    def test_returns_report_when_holdout_present(self) -> None:
        result = WalkForwardResult()
        result.holdout_accuracy = 0.62
        result.holdout_n_samples = 300
        result.holdout_precision_by_class = {1: 0.58, -1: 0.60, 0: 0.65}
        result.holdout_recall_by_class = {1: 0.50, -1: 0.55, 0: 0.70}
        report = WalkForwardEvaluator.holdout_report(result)
        assert report is not None
        assert isinstance(report, MlEvaluationReport)

    def test_holdout_report_accuracy(self) -> None:
        result = WalkForwardResult()
        result.holdout_accuracy = 0.62
        result.holdout_n_samples = 300
        result.holdout_precision_by_class = {1: 0.58, -1: 0.60, 0: 0.65}
        result.holdout_recall_by_class = {1: 0.50, -1: 0.55, 0: 0.70}
        report = WalkForwardEvaluator.holdout_report(result)
        assert report is not None
        assert abs(report.accuracy - 0.62) < 1e-9

    def test_holdout_report_fold_id_is_minus_one(self) -> None:
        result = WalkForwardResult()
        result.holdout_accuracy = 0.60
        result.holdout_n_samples = 100
        result.holdout_precision_by_class = {}
        result.holdout_recall_by_class = {}
        report = WalkForwardEvaluator.holdout_report(result)
        assert report is not None
        assert report.fold_id == -1

    def test_holdout_report_precision_from_result(self) -> None:
        result = WalkForwardResult()
        result.holdout_accuracy = 0.60
        result.holdout_n_samples = 200
        result.holdout_precision_by_class = {1: 0.70, -1: 0.65, 0: 0.55}
        result.holdout_recall_by_class = {1: 0.50, -1: 0.48, 0: 0.72}
        report = WalkForwardEvaluator.holdout_report(result)
        assert report is not None
        assert abs(report.buy_precision - 0.70) < 1e-9
        assert abs(report.sell_precision - 0.65) < 1e-9
