"""Tests for dynamic weight adaptation based on market accuracy.

Covers:
  - DynamicVoteEngine.update_performance_scored()
  - MarketAccuracyTracker: record, advance, evaluate, score computation
  - BaseRunner integration: weights update after forward_bars
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.consensus.engines.dynamic_vote import DynamicVoteEngine
from metatrade.consensus.market_accuracy_tracker import (
    EvalResult,
    MarketAccuracyTracker,
    PendingEvaluation,
    _compute_score,
)
from metatrade.core.enums import SignalDirection

# ── Helpers ────────────────────────────────────────────────────────────────────

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def _signal_stub(module_id: str, direction: SignalDirection):
    """Minimal stub that exposes .module_id and .direction."""
    from unittest.mock import MagicMock
    s = MagicMock()
    s.module_id = module_id
    s.direction = direction
    return s


def _close(price: float) -> Decimal:
    return Decimal(str(price))


# ══════════════════════════════════════════════════════════════════════════════
# _compute_score
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeScore:
    def test_buy_correct_large_move(self) -> None:
        """BUY + market up 1% → score near 1."""
        _, _, score = _compute_score(SignalDirection.BUY, _close(1.0), _close(1.01))
        assert score > 0.9

    def test_buy_wrong_large_move(self) -> None:
        """BUY + market down 1% → score near 0."""
        _, _, score = _compute_score(SignalDirection.BUY, _close(1.0), _close(0.99))
        assert score < 0.1

    def test_buy_flat_market(self) -> None:
        """BUY + market flat → score ≈ 0.5 (neutral)."""
        _, _, score = _compute_score(SignalDirection.BUY, _close(1.0), _close(1.0))
        assert abs(score - 0.5) < 1e-6

    def test_sell_correct_large_move(self) -> None:
        """SELL + market down 1% → score near 1."""
        _, _, score = _compute_score(SignalDirection.SELL, _close(1.0), _close(0.99))
        assert score > 0.9

    def test_sell_wrong_large_move(self) -> None:
        """SELL + market up 1% → score near 0."""
        _, _, score = _compute_score(SignalDirection.SELL, _close(1.0), _close(1.01))
        assert score < 0.1

    def test_score_in_0_1_range(self) -> None:
        for entry, exit_ in [(1.0, 2.0), (1.0, 0.5), (1.0, 1.0), (1.0, 1.0001)]:
            _, _, score = _compute_score(
                SignalDirection.BUY, _close(entry), _close(exit_)
            )
            assert 0.0 <= score <= 1.0

    def test_zero_entry_returns_neutral(self) -> None:
        _, _, score = _compute_score(SignalDirection.BUY, _close(0.0), _close(1.0))
        assert score == 0.5

    def test_proportionality_buy(self) -> None:
        """Larger correct move → higher score."""
        _, _, s_small = _compute_score(SignalDirection.BUY, _close(1.0), _close(1.001))
        _, _, s_large = _compute_score(SignalDirection.BUY, _close(1.0), _close(1.01))
        assert s_large > s_small

    def test_proportionality_wrong_direction(self) -> None:
        """Larger wrong move → lower score."""
        _, _, s_small = _compute_score(SignalDirection.BUY, _close(1.0), _close(0.999))
        _, _, s_large = _compute_score(SignalDirection.BUY, _close(1.0), _close(0.99))
        assert s_large < s_small

    def test_market_return_raw_value(self) -> None:
        market_ret, _, _ = _compute_score(SignalDirection.BUY, _close(1.0), _close(1.01))
        assert abs(market_ret - 0.01) < 1e-9

    def test_direction_return_flipped_for_sell(self) -> None:
        _, dir_ret, _ = _compute_score(SignalDirection.SELL, _close(1.0), _close(0.99))
        assert dir_ret > 0  # price fell = good for SELL = positive direction_return


# ══════════════════════════════════════════════════════════════════════════════
# DynamicVoteEngine.update_performance_scored
# ══════════════════════════════════════════════════════════════════════════════

class TestDynamicVoteEngineScored:
    def _engine(self, alpha: float = 0.5) -> DynamicVoteEngine:
        return DynamicVoteEngine(alpha=alpha, min_weight=0.01)

    def test_perfect_score_increases_accuracy(self) -> None:
        eng = self._engine()
        eng.update_performance_scored("mod_a", 1.0)
        assert eng.get_accuracy("mod_a") > 0.5

    def test_zero_score_decreases_accuracy(self) -> None:
        eng = self._engine()
        eng.update_performance_scored("mod_a", 0.0)
        assert eng.get_accuracy("mod_a") < 0.5

    def test_neutral_score_leaves_accuracy_unchanged(self) -> None:
        eng = self._engine()
        eng.update_performance_scored("mod_a", 0.5)
        assert abs(eng.get_accuracy("mod_a") - 0.5) < 1e-9

    def test_weight_increases_after_correct_predictions(self) -> None:
        eng = self._engine(alpha=0.9)
        for _ in range(10):
            eng.update_performance_scored("mod_a", 1.0)
        assert eng.get_weight("mod_a") > 1.0

    def test_weight_decreases_after_wrong_predictions(self) -> None:
        eng = self._engine(alpha=0.9)
        for _ in range(10):
            eng.update_performance_scored("mod_a", 0.0)
        assert eng.get_weight("mod_a") < 1.0

    def test_weight_never_below_min_weight(self) -> None:
        eng = DynamicVoteEngine(alpha=1.0, min_weight=0.1)
        for _ in range(100):
            eng.update_performance_scored("mod_a", 0.0)
        assert eng.get_weight("mod_a") >= 0.1

    def test_score_clamped_above_1(self) -> None:
        eng = self._engine()
        eng.update_performance_scored("mod_a", 99.0)  # should be clamped to 1.0
        assert eng.get_accuracy("mod_a") <= 1.0

    def test_score_clamped_below_0(self) -> None:
        eng = self._engine()
        eng.update_performance_scored("mod_a", -5.0)  # should be clamped to 0.0
        assert eng.get_accuracy("mod_a") >= 0.0

    def test_binary_update_performance_delegates_to_scored(self) -> None:
        eng = self._engine(alpha=1.0)
        eng.update_performance("mod_a", correct=True)
        assert eng.get_accuracy("mod_a") == 1.0
        eng.update_performance("mod_a", correct=False)
        assert eng.get_accuracy("mod_a") == 0.0

    def test_large_error_reduces_weight_more_than_small_error(self) -> None:
        """Proportionality: score=0.1 reduces weight more than score=0.4."""
        eng_small = self._engine(alpha=0.5)
        eng_large = self._engine(alpha=0.5)
        eng_small.update_performance_scored("m", 0.4)   # small error
        eng_large.update_performance_scored("m", 0.1)   # large error
        assert eng_large.get_weight("m") < eng_small.get_weight("m")


# ══════════════════════════════════════════════════════════════════════════════
# MarketAccuracyTracker
# ══════════════════════════════════════════════════════════════════════════════

class TestMarketAccuracyTracker:
    def _tracker(self, forward_bars: int = 3) -> tuple[MarketAccuracyTracker, list]:
        received: list[tuple[str, float]] = []

        def callback(module_id: str, score: float) -> None:
            received.append((module_id, score))

        return MarketAccuracyTracker(callback, forward_bars=forward_bars), received

    # ── record_signals ────────────────────────────────────────────────────────

    def test_record_buy_signal_adds_pending(self) -> None:
        tracker, _ = self._tracker()
        tracker.record_signals(
            [_signal_stub("mod_a", SignalDirection.BUY)], _close(1.0)
        )
        assert tracker.pending_count == 1

    def test_hold_signals_are_skipped_by_default(self) -> None:
        tracker, _ = self._tracker()
        tracker.record_signals(
            [_signal_stub("mod_a", SignalDirection.HOLD)], _close(1.0)
        )
        assert tracker.pending_count == 0

    def test_multiple_signals_all_recorded(self) -> None:
        tracker, _ = self._tracker()
        tracker.record_signals(
            [
                _signal_stub("mod_a", SignalDirection.BUY),
                _signal_stub("mod_b", SignalDirection.SELL),
            ],
            _close(1.0),
        )
        assert tracker.pending_count == 2

    def test_pending_for_module(self) -> None:
        tracker, _ = self._tracker()
        tracker.record_signals(
            [_signal_stub("mod_a", SignalDirection.BUY)], _close(1.0)
        )
        assert tracker.pending_for_module("mod_a") == 1
        assert tracker.pending_for_module("mod_b") == 0

    # ── on_bar — timing ───────────────────────────────────────────────────────

    def test_no_evaluation_before_forward_bars(self) -> None:
        tracker, received = self._tracker(forward_bars=3)
        tracker.record_signals(
            [_signal_stub("mod_a", SignalDirection.BUY)], _close(1.0)
        )
        # Advance 2 bars (< 3): should NOT evaluate yet
        tracker.on_bar(_close(1.005))
        tracker.on_bar(_close(1.010))
        assert received == []
        assert tracker.pending_count == 1

    def test_evaluation_fires_exactly_at_forward_bars(self) -> None:
        tracker, received = self._tracker(forward_bars=3)
        tracker.record_signals(
            [_signal_stub("mod_a", SignalDirection.BUY)], _close(1.0)
        )
        tracker.on_bar(_close(1.005))
        tracker.on_bar(_close(1.010))
        results = tracker.on_bar(_close(1.015))  # 3rd bar
        assert len(results) == 1
        assert len(received) == 1
        assert tracker.pending_count == 0

    def test_correct_buy_prediction_gives_high_score(self) -> None:
        tracker, received = self._tracker(forward_bars=1)
        tracker.record_signals(
            [_signal_stub("mod_a", SignalDirection.BUY)], _close(1.0)
        )
        tracker.on_bar(_close(1.02))  # market up 2%
        assert received[0][0] == "mod_a"
        assert received[0][1] > 0.9

    def test_wrong_buy_prediction_gives_low_score(self) -> None:
        tracker, received = self._tracker(forward_bars=1)
        tracker.record_signals(
            [_signal_stub("mod_a", SignalDirection.BUY)], _close(1.0)
        )
        tracker.on_bar(_close(0.98))  # market down 2%
        assert received[0][1] < 0.1

    def test_correct_sell_prediction_gives_high_score(self) -> None:
        tracker, received = self._tracker(forward_bars=1)
        tracker.record_signals(
            [_signal_stub("mod_a", SignalDirection.SELL)], _close(1.0)
        )
        tracker.on_bar(_close(0.98))  # market down 2% = correct for SELL
        assert received[0][1] > 0.9

    def test_flat_market_gives_neutral_score(self) -> None:
        tracker, received = self._tracker(forward_bars=1)
        tracker.record_signals(
            [_signal_stub("mod_a", SignalDirection.BUY)], _close(1.0)
        )
        tracker.on_bar(_close(1.0))  # flat
        assert abs(received[0][1] - 0.5) < 1e-6

    def test_eval_result_has_correct_fields(self) -> None:
        tracker, _ = self._tracker(forward_bars=1)
        tracker.record_signals(
            [_signal_stub("mod_a", SignalDirection.BUY)], _close(1.0)
        )
        results = tracker.on_bar(_close(1.01))
        assert len(results) == 1
        r = results[0]
        assert r.module_id == "mod_a"
        assert r.direction == SignalDirection.BUY
        assert r.entry_close == _close(1.0)
        assert r.exit_close == _close(1.01)
        assert r.market_return > 0
        assert 0.0 <= r.score <= 1.0

    # ── reset ─────────────────────────────────────────────────────────────────

    def test_reset_clears_pending(self) -> None:
        tracker, _ = self._tracker()
        tracker.record_signals(
            [_signal_stub("mod_a", SignalDirection.BUY)], _close(1.0)
        )
        tracker.reset()
        assert tracker.pending_count == 0

    def test_reset_does_not_trigger_callback(self) -> None:
        tracker, received = self._tracker()
        tracker.record_signals(
            [_signal_stub("mod_a", SignalDirection.BUY)], _close(1.0)
        )
        tracker.reset()
        assert received == []

    # ── multiple modules ──────────────────────────────────────────────────────

    def test_multiple_modules_evaluated_independently(self) -> None:
        tracker, received = self._tracker(forward_bars=1)
        tracker.record_signals(
            [
                _signal_stub("mod_a", SignalDirection.BUY),
                _signal_stub("mod_b", SignalDirection.SELL),
            ],
            _close(1.0),
        )
        tracker.on_bar(_close(1.01))  # market up: BUY correct, SELL wrong
        ids = {r[0] for r in received}
        scores = {r[0]: r[1] for r in received}
        assert ids == {"mod_a", "mod_b"}
        assert scores["mod_a"] > 0.9   # BUY correct
        assert scores["mod_b"] < 0.1   # SELL wrong


# ══════════════════════════════════════════════════════════════════════════════
# Integration: BaseRunner with auto_weight
# ══════════════════════════════════════════════════════════════════════════════

class TestBaseRunnerAutoWeight:
    """Verify that weights change after enough bars have passed."""

    def _make_runner(self, auto_weight: bool = True, forward_bars: int = 3):
        from decimal import Decimal
        from datetime import datetime, timezone
        from metatrade.core.contracts.market import Bar
        from metatrade.core.enums import SignalDirection, Timeframe
        from metatrade.core.contracts.signal import AnalysisSignal
        from metatrade.core.versioning import ModuleVersion
        from metatrade.technical_analysis.interface import ITechnicalModule
        from metatrade.runner.base import BaseRunner
        from metatrade.runner.config import RunnerConfig

        class AlwaysBuyModule(ITechnicalModule):
            @property
            def module_id(self) -> str:
                return "always_buy"
            @property
            def min_bars(self) -> int:
                return 1
            def analyse(self, bars, ts):
                return AnalysisSignal(
                    direction=SignalDirection.BUY,
                    confidence=0.8,
                    reason="always buy",
                    module_id=self.module_id,
                    module_version=ModuleVersion(1, 0, 0),
                    timestamp_utc=ts,
                )

        cfg = RunnerConfig(
            symbol="EURUSD",
            min_signals=1,
            consensus_threshold=0.5,
            daily_loss_limit_pct=0.99,
            auto_kill_drawdown_pct=0.99,
            signal_cooldown_bars=0,  # disable cooldown so weight tracker can accumulate data
        )
        runner = BaseRunner(
            config=cfg,
            modules=[AlwaysBuyModule()],
            auto_weight=auto_weight,
            weight_forward_bars=forward_bars,
        )
        return runner

    def _bar(self, price: float, i: int = 0) -> "Bar":
        from metatrade.core.contracts.market import Bar
        from metatrade.core.enums import Timeframe
        from decimal import Decimal
        return Bar(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            timestamp_utc=datetime(2024, 1, 1, i % 24, i // 24, tzinfo=timezone.utc),
            open=Decimal(str(price)),
            high=Decimal(str(price + 0.001)),
            low=Decimal(str(price - 0.001)),
            close=Decimal(str(price)),
            volume=Decimal("1000"),
        )

    def _process(self, runner, bars, price: float, i: int):
        from decimal import Decimal
        bar = self._bar(price, i)
        all_bars = bars + [bar]
        runner.process_bar(
            bars=all_bars,
            account_balance=Decimal("10000"),
            account_equity=Decimal("10000"),
            free_margin=Decimal("10000"),
            timestamp_utc=bar.timestamp_utc,
        )
        return bar

    def test_tracker_created_when_auto_weight_true(self) -> None:
        runner = self._make_runner(auto_weight=True)
        assert runner.accuracy_tracker is not None

    def test_tracker_none_when_auto_weight_false(self) -> None:
        runner = self._make_runner(auto_weight=False)
        assert runner.accuracy_tracker is None

    def test_initial_weight_is_neutral(self) -> None:
        runner = self._make_runner()
        # Before any bars, weight should be 1.0 (base × 2 × 0.5)
        assert runner.get_module_weight("always_buy") == pytest.approx(1.0)

    def test_weight_updates_after_forward_bars(self) -> None:
        """After forward_bars bars of rising prices, BUY weight should increase."""
        runner = self._make_runner(forward_bars=3)
        bars: list = []

        # Process 10 rising bars (BUY signals all correct)
        for i in range(10):
            bar = self._bar(1.0 + i * 0.001, i)
            bars.append(bar)
            runner.process_bar(
                bars=bars[-5:],
                account_balance=Decimal("10000"),
                account_equity=Decimal("10000"),
                free_margin=Decimal("10000"),
                timestamp_utc=bar.timestamp_utc,
            )

        # After enough correct predictions, weight > 1.0
        w = runner.get_module_weight("always_buy")
        assert w is not None
        assert w > 1.0  # weight grew because BUY signals were correct

    def test_weight_drops_after_wrong_predictions(self) -> None:
        """After forward_bars bars of falling prices, BUY weight should decrease."""
        runner = self._make_runner(forward_bars=3)
        bars: list = []

        # Process 10 falling bars (BUY signals all wrong)
        for i in range(10):
            bar = self._bar(1.2 - i * 0.005, i)
            bars.append(bar)
            runner.process_bar(
                bars=bars[-5:],
                account_balance=Decimal("10000"),
                account_equity=Decimal("10000"),
                free_margin=Decimal("10000"),
                timestamp_utc=bar.timestamp_utc,
            )

        w = runner.get_module_weight("always_buy")
        assert w is not None
        assert w < 1.0  # weight dropped because BUY signals were wrong

    def test_weight_updates_counted_in_stats(self) -> None:
        runner = self._make_runner(forward_bars=2)
        bars: list = []
        for i in range(6):
            bar = self._bar(1.0 + i * 0.001, i)
            bars.append(bar)
            runner.process_bar(
                bars=bars,
                account_balance=Decimal("10000"),
                account_equity=Decimal("10000"),
                free_margin=Decimal("10000"),
                timestamp_utc=bar.timestamp_utc,
            )
        # After 6 bars with forward_bars=2, should have some evaluations
        assert runner.stats.weight_updates > 0
