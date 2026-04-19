"""Tests for PromotionPolicy."""

from __future__ import annotations

import pytest

from metatrade.ml.live_tracker import LiveAccuracyStats
from metatrade.ml.promotion_policy import PromotionPolicy


def _stats(n_evaluated: int, n_correct: int, min_samples: int = 10) -> LiveAccuracyStats:
    acc = n_correct / n_evaluated if n_evaluated > 0 else None
    return LiveAccuracyStats(
        model_version="v1",
        n_evaluated=n_evaluated,
        n_correct=n_correct,
        n_pending=0,
        accuracy=acc,
        is_reliable=n_evaluated >= min_samples,
        min_samples=min_samples,
    )


@pytest.fixture
def policy():
    return PromotionPolicy(min_holdout=0.52, degradation_threshold=0.50)


class TestPromotionPolicyMinHoldout:
    def test_candidate_below_min_holdout_rejected(self, policy):
        d = policy.evaluate(
            current_holdout=0.55,
            candidate_version="v2",
            candidate_holdout=0.51,
            live_stats=None,
        )
        assert d.should_switch is False
        assert "minimum" in d.reason

    def test_candidate_exactly_at_min_holdout_accepted(self, policy):
        d = policy.evaluate(
            current_holdout=None,
            candidate_version="v2",
            candidate_holdout=0.52,
            live_stats=None,
        )
        assert d.should_switch is True


class TestPromotionPolicyUpgrade:
    def test_upgrade_when_no_current_model(self, policy):
        d = policy.evaluate(
            current_holdout=None,
            candidate_version="v2",
            candidate_holdout=0.56,
            live_stats=None,
        )
        assert d.should_switch is True
        assert d.trigger == "upgrade"
        assert d.candidate_version == "v2"

    def test_upgrade_when_candidate_strictly_better(self, policy):
        d = policy.evaluate(
            current_holdout=0.54,
            candidate_version="v2",
            candidate_holdout=0.58,
            live_stats=None,
        )
        assert d.should_switch is True
        assert d.trigger == "upgrade"

    def test_no_switch_when_candidate_equal(self, policy):
        d = policy.evaluate(
            current_holdout=0.56,
            candidate_version="v2",
            candidate_holdout=0.56,
            live_stats=None,
        )
        assert d.should_switch is False

    def test_no_switch_when_candidate_worse(self, policy):
        d = policy.evaluate(
            current_holdout=0.58,
            candidate_version="v2",
            candidate_holdout=0.55,
            live_stats=None,
        )
        assert d.should_switch is False

    def test_upgrade_works_without_live_stats(self, policy):
        d = policy.evaluate(
            current_holdout=0.53,
            candidate_version="v2",
            candidate_holdout=0.60,
            live_stats=None,
        )
        assert d.should_switch is True


class TestPromotionPolicyEmergency:
    def test_emergency_when_live_accuracy_degraded_and_reliable(self, policy):
        stats = _stats(n_evaluated=15, n_correct=6)  # 40% — below 50% threshold
        d = policy.evaluate(
            current_holdout=0.56,
            candidate_version="v2",
            candidate_holdout=0.54,  # worse than current, but emergency overrides
            live_stats=stats,
        )
        assert d.should_switch is True
        assert d.trigger == "emergency"

    def test_no_emergency_when_not_reliable(self, policy):
        stats = _stats(n_evaluated=5, n_correct=2)  # 40% but only 5 samples < min_samples=10
        d = policy.evaluate(
            current_holdout=0.56,
            candidate_version="v2",
            candidate_holdout=0.54,
            live_stats=stats,
        )
        assert d.should_switch is False  # not reliable yet

    def test_no_emergency_when_accuracy_ok(self, policy):
        stats = _stats(n_evaluated=20, n_correct=12)  # 60% — above threshold
        d = policy.evaluate(
            current_holdout=0.58,
            candidate_version="v2",
            candidate_holdout=0.54,
            live_stats=stats,
        )
        assert d.should_switch is False

    def test_emergency_requires_candidate_above_min_holdout(self, policy):
        stats = _stats(n_evaluated=15, n_correct=6)  # 40% — emergency
        d = policy.evaluate(
            current_holdout=0.56,
            candidate_version="v2",
            candidate_holdout=0.49,  # below min_holdout=0.52
            live_stats=stats,
        )
        # Even emergency doesn't promote a subpar candidate
        assert d.should_switch is False

    def test_emergency_threshold_exactly_at_boundary(self, policy):
        stats = _stats(n_evaluated=15, n_correct=7)  # 46.7%
        d = policy.evaluate(
            current_holdout=0.55,
            candidate_version="v2",
            candidate_holdout=0.53,
            live_stats=stats,
        )
        assert d.should_switch is True
        assert d.trigger == "emergency"

    def test_live_accuracy_exactly_at_threshold_no_emergency(self, policy):
        # threshold=0.50, accuracy exactly 0.50 → not below → no emergency
        stats = _stats(n_evaluated=10, n_correct=5)  # exactly 50%
        d = policy.evaluate(
            current_holdout=0.56,
            candidate_version="v2",
            candidate_holdout=0.53,
            live_stats=stats,
        )
        assert d.should_switch is False
