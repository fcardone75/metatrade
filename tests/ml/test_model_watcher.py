"""Tests for ModelWatcher."""

from __future__ import annotations

import json
import pickle
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.ml.config import MLConfig
from metatrade.ml.live_tracker import LiveAccuracyStats
from metatrade.ml.model_watcher import ModelWatcher
from metatrade.ml.registry import ModelRegistry


BASE_TS = datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)


def _make_bar(minute: int = 0) -> Bar:
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        timestamp_utc=BASE_TS + timedelta(minutes=minute),
        open=Decimal("1.10000"),
        high=Decimal("1.10100"),
        low=Decimal("1.09900"),
        close=Decimal("1.10000"),
        volume=Decimal("1000"),
    )


def _make_config(tmp_path: Path, poll_sec: int = 10) -> MLConfig:
    return MLConfig(
        model_watcher_enabled=True,
        model_watcher_poll_sec=poll_sec,
        candidate_min_holdout=0.52,
        live_accuracy_warning_below=0.50,
        model_registry_dir=str(tmp_path),
    )


def _write_fake_snapshot(
    registry_dir: Path,
    symbol: str,
    version: str,
    holdout: float,
    created_at: float | None = None,
) -> None:
    """Write a minimal fake .pkl + _meta.json so the watcher can find it."""
    # Write a trivial pickle (the watcher only reads the manifest to evaluate
    # the policy; it only loads the pkl when executing a switch)
    import io
    fake_bytes = pickle.dumps({"_stub": True})
    stem = f"{symbol}_{version}"
    (registry_dir / f"{stem}.pkl").write_bytes(fake_bytes)
    manifest = {
        "version": version,
        "symbol": symbol,
        "created_at": created_at or time.time(),
        "tags": {"holdout_accuracy": holdout},
    }
    (registry_dir / f"{stem}_meta.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


class TestRegistryListDiskSnapshots:
    def test_empty_dir(self, tmp_path):
        result = ModelRegistry.list_disk_snapshots(str(tmp_path), "EURUSD")
        assert result == []

    def test_missing_dir(self, tmp_path):
        result = ModelRegistry.list_disk_snapshots(str(tmp_path / "no_such"), "EURUSD")
        assert result == []

    def test_finds_snapshot(self, tmp_path):
        _write_fake_snapshot(tmp_path, "EURUSD", "v1", 0.55)
        result = ModelRegistry.list_disk_snapshots(str(tmp_path), "EURUSD")
        assert len(result) == 1
        assert result[0]["version"] == "v1"
        assert result[0]["tags"]["holdout_accuracy"] == pytest.approx(0.55)

    def test_skips_pkl_without_meta(self, tmp_path):
        (tmp_path / "EURUSD_v_orphan.pkl").write_bytes(b"data")
        result = ModelRegistry.list_disk_snapshots(str(tmp_path), "EURUSD")
        assert result == []

    def test_skips_meta_without_pkl(self, tmp_path):
        (tmp_path / "EURUSD_v_ghost_meta.json").write_text('{"version":"v_ghost"}')
        result = ModelRegistry.list_disk_snapshots(str(tmp_path), "EURUSD")
        assert result == []

    def test_filters_by_symbol(self, tmp_path):
        _write_fake_snapshot(tmp_path, "EURUSD", "v1", 0.55)
        _write_fake_snapshot(tmp_path, "GBPUSD", "v2", 0.58)
        result = ModelRegistry.list_disk_snapshots(str(tmp_path), "EURUSD")
        assert all(r["symbol"] == "EURUSD" for r in result)
        assert len(result) == 1

    def test_sorted_newest_first(self, tmp_path):
        _write_fake_snapshot(tmp_path, "EURUSD", "v_old", 0.54, created_at=1000.0)
        _write_fake_snapshot(tmp_path, "EURUSD", "v_new", 0.57, created_at=2000.0)
        result = ModelRegistry.list_disk_snapshots(str(tmp_path), "EURUSD")
        assert result[0]["version"] == "v_new"


class TestModelWatcherNoCurrent:
    def test_no_candidates_no_switch(self, tmp_path):
        cfg = _make_config(tmp_path)
        registry = ModelRegistry(MLConfig(model_registry_dir=str(tmp_path)))
        watcher = ModelWatcher(registry, "EURUSD", cfg)
        switched = watcher.on_bar(_make_bar(), open_positions=0)
        assert switched is False

    def test_better_candidate_switches_when_no_positions(self, tmp_path):
        _write_fake_snapshot(tmp_path, "EURUSD", "v_candidate", 0.56)
        cfg = _make_config(tmp_path)
        registry = ModelRegistry(MLConfig(model_registry_dir=str(tmp_path)))
        watcher = ModelWatcher(registry, "EURUSD", cfg)

        switched = watcher.on_bar(_make_bar(), open_positions=0)
        assert switched is True
        assert watcher.state.current_version == "v_candidate"
        assert watcher.state.total_switches == 1

    def test_candidate_below_min_holdout_not_switched(self, tmp_path):
        _write_fake_snapshot(tmp_path, "EURUSD", "v_weak", 0.50)  # below min 0.52
        cfg = _make_config(tmp_path)
        registry = ModelRegistry(MLConfig(model_registry_dir=str(tmp_path)))
        watcher = ModelWatcher(registry, "EURUSD", cfg)
        switched = watcher.on_bar(_make_bar(), open_positions=0)
        assert switched is False

    def test_switch_deferred_when_positions_open(self, tmp_path):
        _write_fake_snapshot(tmp_path, "EURUSD", "v_candidate", 0.58)
        cfg = _make_config(tmp_path)
        registry = ModelRegistry(MLConfig(model_registry_dir=str(tmp_path)))
        watcher = ModelWatcher(registry, "EURUSD", cfg)

        switched = watcher.on_bar(_make_bar(0), open_positions=2)
        assert switched is False
        assert watcher.state.pending_switch_version == "v_candidate"

    def test_pending_switch_executes_when_positions_close(self, tmp_path):
        _write_fake_snapshot(tmp_path, "EURUSD", "v_candidate", 0.58)
        cfg = _make_config(tmp_path)
        registry = ModelRegistry(MLConfig(model_registry_dir=str(tmp_path)))
        watcher = ModelWatcher(registry, "EURUSD", cfg)

        # First bar: positions open → deferred
        watcher.on_bar(_make_bar(0), open_positions=1)
        assert watcher.state.pending_switch_version == "v_candidate"

        # Second bar: positions closed → execute switch
        switched = watcher.on_bar(_make_bar(1), open_positions=0)
        assert switched is True
        assert watcher.state.pending_switch_version is None
        assert watcher.state.current_version == "v_candidate"


class TestModelWatcherPollDebounce:
    def test_poll_debounce_prevents_redundant_disk_scans(self, tmp_path):
        cfg = _make_config(tmp_path, poll_sec=3600)  # effectively never re-poll
        registry = ModelRegistry(MLConfig(model_registry_dir=str(tmp_path)))
        watcher = ModelWatcher(registry, "EURUSD", cfg)

        # Simulate: first poll already happened (last_poll_ts set to now)
        watcher._state.last_poll_ts = time.monotonic()
        # Add a candidate after the last poll
        _write_fake_snapshot(tmp_path, "EURUSD", "v_late", 0.60)
        # Call within debounce window — should not see new candidate
        switched = watcher.on_bar(_make_bar(1), open_positions=0)
        assert switched is False


class TestModelWatcherWithEmergency:
    def test_emergency_switch_queued_when_positions_open(self, tmp_path):
        _write_fake_snapshot(tmp_path, "EURUSD", "v_candidate", 0.54)
        cfg = _make_config(tmp_path)
        registry = ModelRegistry(MLConfig(model_registry_dir=str(tmp_path)))
        watcher = ModelWatcher(registry, "EURUSD", cfg)
        watcher._state.current_holdout = 0.58  # set higher so upgrade doesn't fire
        watcher._state.current_version = "v_current"

        # Simulate degraded live accuracy (below 50%, 15 samples → reliable)
        degraded = LiveAccuracyStats(
            model_version="v_current",
            n_evaluated=15,
            n_correct=6,  # 40%
            n_pending=0,
            accuracy=0.40,
            is_reliable=True,
            min_samples=10,
        )
        watcher.on_bar(_make_bar(0), open_positions=1, live_stats=degraded)
        assert watcher.state.pending_switch_version == "v_candidate"
