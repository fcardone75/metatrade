"""Tests for all three consensus engine implementations."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from metatrade.consensus.engines.dynamic_vote import DynamicVoteEngine
from metatrade.consensus.engines.simple_vote import SimpleVoteEngine
from metatrade.consensus.engines.weighted_vote import WeightedVoteEngine
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import ConsensusMode, SignalDirection
from metatrade.core.versioning import ModuleVersion

UTC = UTC
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
V1 = ModuleVersion(1, 0, 0)


def make_signal(
    direction: SignalDirection,
    confidence: float = 0.70,
    module_id: str = "m1",
) -> AnalysisSignal:
    return AnalysisSignal(
        direction=direction,
        confidence=confidence,
        reason="test",
        module_id=module_id,
        module_version=V1,
        timestamp_utc=T0,
    )


# ── SimpleVoteEngine ──────────────────────────────────────────────────────────


class TestSimpleVoteEngine:
    def _engine(self, threshold: float = 0.60, min_signals: int = 1) -> SimpleVoteEngine:
        return SimpleVoteEngine(threshold=threshold, min_signals=min_signals)

    def test_empty_signals_returns_hold(self) -> None:
        result = self._engine().evaluate("EURUSD", [], T0)
        assert result.final_direction == SignalDirection.HOLD
        assert not result.is_actionable

    def test_all_buy_is_actionable(self) -> None:
        signals = [make_signal(SignalDirection.BUY, module_id=f"m{i}") for i in range(3)]
        result = self._engine().evaluate("EURUSD", signals, T0)
        assert result.final_direction == SignalDirection.BUY
        assert result.is_actionable
        assert result.buy_pct == pytest.approx(1.0)

    def test_all_sell_is_actionable(self) -> None:
        signals = [make_signal(SignalDirection.SELL, module_id=f"m{i}") for i in range(3)]
        result = self._engine().evaluate("EURUSD", signals, T0)
        assert result.final_direction == SignalDirection.SELL
        assert result.is_actionable

    def test_majority_buy_below_threshold_not_actionable(self) -> None:
        # 2 BUY, 2 SELL, 2 HOLD → BUY wins with 33% < 60% threshold
        signals = [
            make_signal(SignalDirection.BUY, module_id="b1"),
            make_signal(SignalDirection.BUY, module_id="b2"),
            make_signal(SignalDirection.SELL, module_id="s1"),
            make_signal(SignalDirection.SELL, module_id="s2"),
            make_signal(SignalDirection.HOLD, module_id="h1"),
            make_signal(SignalDirection.HOLD, module_id="h2"),
        ]
        result = self._engine(threshold=0.60).evaluate("EURUSD", signals, T0)
        assert not result.is_actionable

    def test_threshold_met_exactly_is_actionable(self) -> None:
        # 3 BUY, 2 SELL → BUY = 60% exactly
        signals = [
            make_signal(SignalDirection.BUY, module_id="b1"),
            make_signal(SignalDirection.BUY, module_id="b2"),
            make_signal(SignalDirection.BUY, module_id="b3"),
            make_signal(SignalDirection.SELL, module_id="s1"),
            make_signal(SignalDirection.SELL, module_id="s2"),
        ]
        result = self._engine(threshold=0.60).evaluate("EURUSD", signals, T0)
        assert result.is_actionable
        assert result.buy_pct == pytest.approx(0.60, abs=1e-9)

    def test_min_signals_not_met_returns_hold(self) -> None:
        signals = [make_signal(SignalDirection.BUY)]
        result = self._engine(min_signals=3).evaluate("EURUSD", signals, T0)
        assert not result.is_actionable
        assert result.final_direction == SignalDirection.HOLD

    def test_percentages_sum_to_one(self) -> None:
        signals = [
            make_signal(SignalDirection.BUY, module_id="b1"),
            make_signal(SignalDirection.SELL, module_id="s1"),
            make_signal(SignalDirection.HOLD, module_id="h1"),
        ]
        result = self._engine().evaluate("EURUSD", signals, T0)
        assert result.buy_pct + result.sell_pct + result.hold_pct == pytest.approx(1.0)

    def test_vote_records_all_weight_one(self) -> None:
        signals = [make_signal(SignalDirection.BUY, module_id=f"m{i}") for i in range(3)]
        result = self._engine().evaluate("EURUSD", signals, T0)
        assert all(vr.weight == 1.0 for vr in result.vote_records)

    def test_aggregate_confidence(self) -> None:
        signals = [
            make_signal(SignalDirection.BUY, confidence=0.80, module_id="b1"),
            make_signal(SignalDirection.BUY, confidence=0.60, module_id="b2"),
        ]
        result = self._engine().evaluate("EURUSD", signals, T0)
        assert result.aggregate_confidence == pytest.approx(0.70)

    def test_consensus_mode_is_simple(self) -> None:
        result = self._engine().evaluate("EURUSD", [make_signal(SignalDirection.BUY)], T0)
        assert result.consensus_mode == ConsensusMode.SIMPLE_VOTE

    def test_symbol_preserved(self) -> None:
        result = self._engine().evaluate("GBPUSD", [make_signal(SignalDirection.BUY)], T0)
        assert result.symbol == "GBPUSD"

    def test_single_hold_signal_not_actionable(self) -> None:
        result = self._engine().evaluate("EURUSD", [make_signal(SignalDirection.HOLD)], T0)
        assert not result.is_actionable

    def test_empty_result_has_hold_pct_one(self) -> None:
        result = self._engine().evaluate("EURUSD", [], T0)
        assert result.hold_pct == 1.0
        assert result.buy_pct == 0.0
        assert result.sell_pct == 0.0


# ── WeightedVoteEngine ────────────────────────────────────────────────────────


class TestWeightedVoteEngine:
    def test_high_weight_module_overrides_majority(self) -> None:
        # 3 SELL (weight 1.0) vs 1 BUY (weight 5.0) → BUY should win on score
        signals = [
            make_signal(SignalDirection.BUY, confidence=1.0, module_id="heavy"),
            make_signal(SignalDirection.SELL, confidence=1.0, module_id="s1"),
            make_signal(SignalDirection.SELL, confidence=1.0, module_id="s2"),
            make_signal(SignalDirection.SELL, confidence=1.0, module_id="s3"),
        ]
        engine = WeightedVoteEngine(
            module_weights={"heavy": 5.0},
            threshold=0.50,
        )
        result = engine.evaluate("EURUSD", signals, T0)
        assert result.final_direction == SignalDirection.BUY
        assert result.is_actionable

    def test_default_weight_used_for_unknown_module(self) -> None:
        engine = WeightedVoteEngine(default_weight=2.0)
        assert engine.get_weight("unknown_module") == 2.0

    def test_known_module_uses_config_weight(self) -> None:
        engine = WeightedVoteEngine(module_weights={"rsi_h1": 1.5})
        assert engine.get_weight("rsi_h1") == 1.5

    def test_empty_signals_returns_hold(self) -> None:
        engine = WeightedVoteEngine()
        result = engine.evaluate("EURUSD", [], T0)
        assert not result.is_actionable

    def test_consensus_mode_is_weighted(self) -> None:
        engine = WeightedVoteEngine()
        result = engine.evaluate("EURUSD", [make_signal(SignalDirection.BUY)], T0)
        assert result.consensus_mode == ConsensusMode.WEIGHTED_VOTE

    def test_weighted_scores_in_vote_records(self) -> None:
        engine = WeightedVoteEngine(module_weights={"m1": 2.0})
        signal = make_signal(SignalDirection.BUY, confidence=0.80, module_id="m1")
        result = engine.evaluate("EURUSD", [signal], T0)
        vr = result.vote_records[0]
        assert vr.weight == 2.0
        assert vr.weighted_score == pytest.approx(1.60)

    def test_min_signals_not_met_returns_hold(self) -> None:
        engine = WeightedVoteEngine(min_signals=2)
        result = engine.evaluate("EURUSD", [make_signal(SignalDirection.BUY)], T0)
        assert not result.is_actionable

    def test_below_threshold_not_actionable(self) -> None:
        # 1 BUY, 1 SELL → 50% each, threshold=0.60
        signals = [
            make_signal(SignalDirection.BUY, module_id="b"),
            make_signal(SignalDirection.SELL, module_id="s"),
        ]
        engine = WeightedVoteEngine(threshold=0.60)
        result = engine.evaluate("EURUSD", signals, T0)
        assert not result.is_actionable


# ── DynamicVoteEngine ─────────────────────────────────────────────────────────


class TestDynamicVoteEngine:
    def test_initial_weight_is_base_weight(self) -> None:
        engine = DynamicVoteEngine(default_weight=1.0)
        assert engine.get_weight("any_module") == pytest.approx(1.0)

    def test_weight_increases_after_correct_predictions(self) -> None:
        engine = DynamicVoteEngine(default_weight=1.0, alpha=0.20)
        initial_w = engine.get_weight("m1")
        for _ in range(20):
            engine.update_performance("m1", correct=True)
        assert engine.get_weight("m1") > initial_w

    def test_weight_decreases_after_wrong_predictions(self) -> None:
        engine = DynamicVoteEngine(default_weight=1.0, alpha=0.20, min_weight=0.0)
        for _ in range(20):
            engine.update_performance("m1", correct=False)
        # After many wrong predictions, accuracy → 0 → weight → 0 (floored by min_weight)
        assert engine.get_weight("m1") < 1.0

    def test_weight_floored_at_min_weight(self) -> None:
        engine = DynamicVoteEngine(default_weight=1.0, min_weight=0.10, alpha=1.0)
        engine.update_performance("m1", correct=False)  # alpha=1.0 → accuracy=0 instantly
        assert engine.get_weight("m1") == pytest.approx(0.10)

    def test_accuracy_initialised_at_neutral(self) -> None:
        engine = DynamicVoteEngine()
        assert engine.get_accuracy("new_module") == pytest.approx(0.5)

    def test_accuracy_ema_converges_to_one(self) -> None:
        engine = DynamicVoteEngine(alpha=0.20)
        for _ in range(100):
            engine.update_performance("m1", correct=True)
        assert engine.get_accuracy("m1") > 0.99

    def test_accuracy_ema_converges_to_zero(self) -> None:
        engine = DynamicVoteEngine(alpha=0.20)
        for _ in range(100):
            engine.update_performance("m1", correct=False)
        assert engine.get_accuracy("m1") < 0.01

    def test_evaluate_returns_consensus_result(self) -> None:
        engine = DynamicVoteEngine()
        signals = [make_signal(SignalDirection.BUY, module_id="m1")]
        result = engine.evaluate("EURUSD", signals, T0)
        assert result.final_direction == SignalDirection.BUY
        assert result.consensus_mode == ConsensusMode.DYNAMIC_VOTE

    def test_empty_signals_returns_hold(self) -> None:
        engine = DynamicVoteEngine()
        result = engine.evaluate("EURUSD", [], T0)
        assert not result.is_actionable

    def test_high_accuracy_module_gets_more_weight(self) -> None:
        engine = DynamicVoteEngine(default_weight=1.0, alpha=0.30)
        # Train m1 to be accurate
        for _ in range(30):
            engine.update_performance("m1", correct=True)
        w_accurate = engine.get_weight("m1")
        w_neutral = engine.get_weight("m_unknown")
        assert w_accurate > w_neutral

    def test_update_performance_independent_per_module(self) -> None:
        engine = DynamicVoteEngine(alpha=0.30)
        for _ in range(20):
            engine.update_performance("good", correct=True)
            engine.update_performance("bad", correct=False)
        assert engine.get_accuracy("good") > engine.get_accuracy("bad")

    def test_dynamic_weight_favours_accurate_module_in_evaluation(self) -> None:
        # Train "accurate" to have high weight, "noise" to have low weight
        engine = DynamicVoteEngine(default_weight=1.0, alpha=0.40, min_weight=0.01)
        for _ in range(30):
            engine.update_performance("accurate", correct=True)
            engine.update_performance("noise", correct=False)

        signals = [
            make_signal(SignalDirection.BUY, confidence=1.0, module_id="accurate"),
            make_signal(SignalDirection.SELL, confidence=1.0, module_id="noise"),
            make_signal(SignalDirection.SELL, confidence=1.0, module_id="noise2"),
        ]
        # noise2 is unknown → neutral weight. But "accurate" has ~2x base weight.
        # With threshold=0.40, BUY should win.
        result = engine.evaluate("EURUSD", signals, T0)
        assert result.final_direction == SignalDirection.BUY
