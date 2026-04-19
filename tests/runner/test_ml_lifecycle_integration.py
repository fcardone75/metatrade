"""Integration tests: LiveAccuracyTracker + ModelWatcher wired into BaseRunner."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.ml.config import MLConfig
from metatrade.ml.live_tracker import LiveAccuracyTracker
from metatrade.ml.model_watcher import ModelWatcher
from metatrade.ml.registry import ModelRegistry
from metatrade.observability.store import TelemetryStore
from metatrade.runner.base import BaseRunner
from metatrade.runner.config import RunnerConfig

BASE_TS = datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)


def _make_bars(n: int, start_close: float = 1.10000, step: float = 0.0001) -> list[Bar]:
    bars = []
    for i in range(n):
        close = Decimal(str(round(start_close + i * step, 5)))
        bars.append(
            Bar(
                symbol="EURUSD",
                timeframe=Timeframe.M1,
                timestamp_utc=BASE_TS + timedelta(minutes=i),
                open=close,
                high=close + Decimal("0.001"),
                low=close - Decimal("0.001"),
                close=close,
                volume=Decimal("1000"),
            )
        )
    return bars


def _make_runner(tmp_path, live_tracker=None, model_watcher=None) -> BaseRunner:
    cfg = RunnerConfig(signal_cooldown_bars=0)
    return BaseRunner(
        cfg,
        modules=[],
        live_tracker=live_tracker,
        model_watcher=model_watcher,
    )


class TestBackgroundTasksRunEveryBar:
    """Verify that background tasks execute on every bar (not gated by session/cooldown)."""

    def test_live_tracker_evaluates_pending_every_bar(self, tmp_path):
        store = TelemetryStore(str(tmp_path / "t.db"))
        tracker = LiveAccuracyTracker(
            symbol="EURUSD",
            timeframe="M1",
            forward_bars=5,
            tf_minutes=1,
            telemetry=store,
            min_samples=10,
        )
        runner = _make_runner(tmp_path, live_tracker=tracker)
        bars = _make_bars(60)

        # Manually insert a prediction that should be evaluated by bar 10
        store.record_ml_prediction(
            model_version="v1",
            symbol="EURUSD",
            timeframe="M1",
            bar_ts_utc=BASE_TS,
            direction=1,
            confidence=0.7,
            entry_close=1.10000,
            evaluate_after_ts=BASE_TS + timedelta(minutes=5),
        )

        # Process bars 0-9 (6 bars after the prediction's window opens)
        for i in range(10):
            runner.process_bar(
                bars=bars[: i + 1],
                account_balance=Decimal("10000"),
                account_equity=Decimal("10000"),
                free_margin=Decimal("10000"),
                timestamp_utc=bars[i].timestamp_utc,
            )

        stats = tracker.live_accuracy_stats("v1")
        assert stats.n_evaluated == 1  # prediction evaluated when window elapsed

    def test_model_watcher_called_outside_session(self, tmp_path):
        """Watcher must run even when session gate would block trade entries."""
        import json, pickle, time

        # Write a candidate snapshot
        (tmp_path / "EURUSD_v_candidate.pkl").write_bytes(pickle.dumps({"stub": True}))
        manifest = {
            "version": "v_candidate",
            "symbol": "EURUSD",
            "created_at": time.time(),
            "tags": {"holdout_accuracy": 0.58},
        }
        (tmp_path / "EURUSD_v_candidate_meta.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        ml_cfg = MLConfig(
            model_watcher_enabled=True,
            model_watcher_poll_sec=10,
            candidate_min_holdout=0.52,
            model_registry_dir=str(tmp_path),
        )
        registry = ModelRegistry(MLConfig(model_registry_dir=str(tmp_path)))
        watcher = ModelWatcher(registry, "EURUSD", ml_cfg)

        # Session filter that blocks the hours we're using (10:xx UTC)
        cfg = RunnerConfig(
            signal_cooldown_bars=0,
            session_filter_utc_start=14,
            session_filter_utc_end=20,
        )
        runner = BaseRunner(cfg, modules=[], model_watcher=watcher)
        bars = _make_bars(5)

        for i in range(5):
            runner.process_bar(
                bars=bars[: i + 1],
                account_balance=Decimal("10000"),
                account_equity=Decimal("10000"),
                free_margin=Decimal("10000"),
                timestamp_utc=bars[i].timestamp_utc,
            )

        # Watcher ran and found the candidate (switch should have happened
        # since open_positions=0 and candidate is better than nothing)
        assert watcher.state.total_switches == 1
        assert watcher.state.current_version == "v_candidate"
