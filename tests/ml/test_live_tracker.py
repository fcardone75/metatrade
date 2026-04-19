"""Tests for LiveAccuracyTracker and TelemetryStore live prediction methods."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.ml.live_tracker import LiveAccuracyStats, LiveAccuracyTracker
from metatrade.observability.store import TelemetryStore


BASE_TS = datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)


def _make_bar(close: float, ts: datetime) -> Bar:
    d = Decimal(str(round(close, 5)))
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        timestamp_utc=ts,
        open=d,
        high=Decimal(str(round(close + 0.001, 5))),
        low=Decimal(str(round(close - 0.001, 5))),
        close=d,
        volume=Decimal("1000"),
    )


def _bar_at(minute_offset: int, close: float = 1.10000) -> Bar:
    return _make_bar(close, BASE_TS + timedelta(minutes=minute_offset))


@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "test_telemetry.db")
    s = TelemetryStore(db)
    yield s
    s.close()


@pytest.fixture
def tracker(store):
    return LiveAccuracyTracker(
        symbol="EURUSD",
        timeframe="M1",
        forward_bars=5,
        tf_minutes=1,
        telemetry=store,
        min_samples=10,
    )


# ── TelemetryStore low-level ──────────────────────────────────────────────────

class TestTelemetryStorePredictions:
    def test_record_and_retrieve_pending(self, store):
        now = BASE_TS
        evaluate_after = now + timedelta(minutes=5)
        store.record_ml_prediction(
            model_version="v1",
            symbol="EURUSD",
            timeframe="M1",
            bar_ts_utc=now,
            direction=1,
            confidence=0.7,
            entry_close=1.1000,
            evaluate_after_ts=evaluate_after,
        )
        # Not yet due
        pending = store.get_pending_predictions("EURUSD", "M1", before_ts=now)
        assert len(pending) == 0

        # Now due
        pending = store.get_pending_predictions("EURUSD", "M1", before_ts=evaluate_after)
        assert len(pending) == 1
        assert pending[0]["direction"] == 1
        assert abs(pending[0]["entry_close"] - 1.1000) < 1e-6

    def test_evaluate_marks_correct(self, store):
        now = BASE_TS
        evaluate_after = now + timedelta(minutes=5)
        pred_id = store.record_ml_prediction(
            model_version="v1",
            symbol="EURUSD",
            timeframe="M1",
            bar_ts_utc=now,
            direction=1,
            confidence=0.7,
            entry_close=1.1000,
            evaluate_after_ts=evaluate_after,
        )
        store.evaluate_ml_prediction(
            prediction_id=pred_id,
            future_close=1.1050,
            correct=True,
            evaluated_at_utc=evaluate_after,
        )
        # Should not appear in pending anymore
        pending = store.get_pending_predictions("EURUSD", "M1", before_ts=evaluate_after)
        assert len(pending) == 0

    def test_accuracy_stats_empty(self, store):
        stats = store.get_live_accuracy_stats("unknown_model", min_samples=10)
        assert stats["n_evaluated"] == 0
        assert stats["accuracy"] is None
        assert stats["is_reliable"] is False

    def test_accuracy_stats_below_min_samples(self, store):
        now = BASE_TS
        evaluate_after = now + timedelta(minutes=1)
        for i in range(5):
            pid = store.record_ml_prediction(
                model_version="v1",
                symbol="EURUSD",
                timeframe="M1",
                bar_ts_utc=now + timedelta(seconds=i),
                direction=1,
                confidence=0.7,
                entry_close=1.1,
                evaluate_after_ts=evaluate_after,
            )
            store.evaluate_ml_prediction(
                prediction_id=pid,
                future_close=1.11,
                correct=True,
                evaluated_at_utc=evaluate_after,
            )
        stats = store.get_live_accuracy_stats("v1", min_samples=10)
        assert stats["n_evaluated"] == 5
        assert stats["accuracy"] == pytest.approx(1.0)
        assert stats["is_reliable"] is False  # 5 < 10

    def test_accuracy_stats_above_min_samples(self, store):
        now = BASE_TS
        evaluate_after = now + timedelta(minutes=1)
        for i in range(12):
            correct = i % 3 != 0  # ~66% correct
            pid = store.record_ml_prediction(
                model_version="v2",
                symbol="EURUSD",
                timeframe="M1",
                bar_ts_utc=now + timedelta(seconds=i),
                direction=1,
                confidence=0.6,
                entry_close=1.1,
                evaluate_after_ts=evaluate_after,
            )
            store.evaluate_ml_prediction(
                prediction_id=pid,
                future_close=1.11 if correct else 1.09,
                correct=correct,
                evaluated_at_utc=evaluate_after,
            )
        stats = store.get_live_accuracy_stats("v2", min_samples=10)
        assert stats["n_evaluated"] == 12
        assert stats["is_reliable"] is True


# ── LiveAccuracyTracker ───────────────────────────────────────────────────────

class TestLiveAccuracyTracker:
    def test_hold_predictions_ignored(self, tracker, store):
        bar = _bar_at(0)
        tracker.record_prediction(bar, direction=0, confidence=0.9, model_version="v1")
        pending = store.get_pending_predictions("EURUSD", "M1", before_ts=BASE_TS + timedelta(hours=1))
        assert len(pending) == 0

    def test_buy_prediction_recorded(self, tracker, store):
        bar = _bar_at(0, close=1.10000)
        tracker.record_prediction(bar, direction=1, confidence=0.7, model_version="v1")
        pending = store.get_pending_predictions(
            "EURUSD", "M1", before_ts=BASE_TS + timedelta(minutes=10)
        )
        assert len(pending) == 1

    def test_evaluate_pending_correct_buy(self, tracker):
        entry_bar = _bar_at(0, close=1.10000)
        tracker.record_prediction(entry_bar, direction=1, confidence=0.7, model_version="v1")
        # 5 bars later price went up
        future_bar = _bar_at(5, close=1.10100)
        n = tracker.evaluate_pending(future_bar)
        assert n == 1

        stats = tracker.live_accuracy_stats("v1")
        assert stats.n_evaluated == 1
        assert stats.n_correct == 1

    def test_evaluate_pending_incorrect_buy(self, tracker):
        entry_bar = _bar_at(0, close=1.10000)
        tracker.record_prediction(entry_bar, direction=1, confidence=0.7, model_version="v1")
        future_bar = _bar_at(5, close=1.09900)  # price went down
        tracker.evaluate_pending(future_bar)

        stats = tracker.live_accuracy_stats("v1")
        assert stats.n_correct == 0

    def test_evaluate_pending_correct_sell(self, tracker):
        entry_bar = _bar_at(0, close=1.10000)
        tracker.record_prediction(entry_bar, direction=-1, confidence=0.65, model_version="v1")
        future_bar = _bar_at(5, close=1.09900)  # price went down = correct for SELL
        tracker.evaluate_pending(future_bar)

        stats = tracker.live_accuracy_stats("v1")
        assert stats.n_correct == 1

    def test_evaluate_not_yet_due(self, tracker):
        entry_bar = _bar_at(0, close=1.10000)
        tracker.record_prediction(entry_bar, direction=1, confidence=0.7, model_version="v1")
        # Only 3 bars later (forward_bars=5)
        future_bar = _bar_at(3, close=1.10100)
        n = tracker.evaluate_pending(future_bar)
        assert n == 0

        stats = tracker.live_accuracy_stats("v1")
        assert stats.n_evaluated == 0
        assert stats.n_pending == 1

    def test_reliability_below_min_samples(self, tracker):
        for i in range(5):
            bar = _bar_at(i, close=1.10000)
            tracker.record_prediction(bar, direction=1, confidence=0.7, model_version="v1")
        future = _bar_at(20, close=1.10100)
        tracker.evaluate_pending(future)

        stats = tracker.live_accuracy_stats("v1")
        assert stats.is_reliable is False  # 5 < min_samples=10

    def test_reliability_above_min_samples(self, tracker):
        for i in range(15):
            bar = _bar_at(i, close=1.10000)
            tracker.record_prediction(bar, direction=1, confidence=0.7, model_version="v1")
        future = _bar_at(30, close=1.10100)
        tracker.evaluate_pending(future)

        stats = tracker.live_accuracy_stats("v1")
        assert stats.is_reliable is True  # 15 >= min_samples=10
        assert stats.accuracy == pytest.approx(1.0)

    def test_accuracy_pct_formatting(self, tracker):
        bar = _bar_at(0, close=1.10000)
        tracker.record_prediction(bar, direction=1, confidence=0.7, model_version="v1")
        future = _bar_at(5, close=1.10100)
        tracker.evaluate_pending(future)

        stats = tracker.live_accuracy_stats("v1")
        assert "%" in stats.accuracy_pct

    def test_unreliable_shows_n_a(self, tracker):
        stats = tracker.live_accuracy_stats("v_empty")
        assert stats.accuracy_pct == "n/a"
