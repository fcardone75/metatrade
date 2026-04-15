"""Tests for AdaptiveThresholdManager."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from metatrade.consensus.adaptive_threshold import AdaptiveThresholdManager, ModuleThresholdState
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion, SchemaVersion

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)
_VERSION = ModuleVersion(1, 0, 0)


def _signal(module_id: str, confidence: float, direction: SignalDirection = SignalDirection.BUY) -> AnalysisSignal:
    return AnalysisSignal(
        direction=direction,
        confidence=confidence,
        reason="test",
        module_id=module_id,
        module_version=_VERSION,
        timestamp_utc=_NOW,
    )


# ── Construction ───────────────────────────────────────────────────────────────

def test_default_threshold_applied_to_unknown_module():
    mgr = AdaptiveThresholdManager(store=None, default_threshold=0.60)
    assert mgr.get_threshold("new_module") == pytest.approx(0.60)


def test_store_is_loaded_on_init():
    store = MagicMock()
    store.list_module_thresholds.return_value = [
        {"module_id": "ema_m15", "threshold": 0.72, "eval_count": 10, "correct_count": 6, "mean_score": 0.58},
    ]
    mgr = AdaptiveThresholdManager(store=store, default_threshold=0.60)
    assert mgr.get_threshold("ema_m15") == pytest.approx(0.72)
    assert mgr.get_threshold("unknown") == pytest.approx(0.60)


def test_store_load_failure_is_silently_ignored():
    store = MagicMock()
    store.list_module_thresholds.side_effect = RuntimeError("DB down")
    mgr = AdaptiveThresholdManager(store=store)
    assert mgr.get_threshold("any") == pytest.approx(0.60)


# ── Threshold adjustment ───────────────────────────────────────────────────────

def test_correct_signal_lowers_threshold():
    mgr = AdaptiveThresholdManager(store=None, default_threshold=0.60, alpha=0.01)
    mgr.on_eval("mod_a", score=1.0)   # fully correct
    assert mgr.get_threshold("mod_a") < 0.60


def test_wrong_signal_raises_threshold():
    mgr = AdaptiveThresholdManager(store=None, default_threshold=0.60, alpha=0.01)
    mgr.on_eval("mod_a", score=0.0)   # fully wrong
    assert mgr.get_threshold("mod_a") > 0.60


def test_neutral_signal_barely_changes_threshold():
    mgr = AdaptiveThresholdManager(store=None, default_threshold=0.60, alpha=0.01)
    before = mgr.get_threshold("mod_a")
    mgr.on_eval("mod_a", score=0.5)
    after = mgr.get_threshold("mod_a")
    assert abs(after - before) < 1e-9


def test_threshold_respects_min_floor():
    mgr = AdaptiveThresholdManager(
        store=None, default_threshold=0.46, alpha=0.10, min_threshold=0.45
    )
    for _ in range(50):
        mgr.on_eval("mod_a", score=1.0)
    assert mgr.get_threshold("mod_a") >= 0.45


def test_threshold_respects_max_ceiling():
    mgr = AdaptiveThresholdManager(
        store=None, default_threshold=0.88, alpha=0.10, max_threshold=0.90
    )
    for _ in range(50):
        mgr.on_eval("mod_a", score=0.0)
    assert mgr.get_threshold("mod_a") <= 0.90


def test_proportional_adjustment_correct_stronger_than_mild():
    mgr_strong = AdaptiveThresholdManager(store=None, default_threshold=0.60, alpha=0.01)
    mgr_mild = AdaptiveThresholdManager(store=None, default_threshold=0.60, alpha=0.01)
    mgr_strong.on_eval("m", score=1.0)
    mgr_mild.on_eval("m", score=0.6)
    # Strong correction lowers threshold more than mild
    assert mgr_strong.get_threshold("m") < mgr_mild.get_threshold("m")


# ── Stats tracking ─────────────────────────────────────────────────────────────

def test_eval_count_increments():
    mgr = AdaptiveThresholdManager(store=None)
    mgr.on_eval("m", 0.7)
    mgr.on_eval("m", 0.3)
    states = {s.module_id: s for s in mgr.all_states()}
    assert states["m"].eval_count == 2


def test_correct_count_only_when_score_above_half():
    mgr = AdaptiveThresholdManager(store=None)
    mgr.on_eval("m", 0.8)  # correct
    mgr.on_eval("m", 0.4)  # wrong
    mgr.on_eval("m", 0.5)  # exactly 0.5 → correct
    states = {s.module_id: s for s in mgr.all_states()}
    assert states["m"].correct_count == 2


def test_mean_score_ema_updates():
    mgr = AdaptiveThresholdManager(store=None, score_ema_alpha=1.0)  # alpha=1 → mean = last score
    mgr.on_eval("m", 0.9)
    states = {s.module_id: s for s in mgr.all_states()}
    assert states["m"].mean_score == pytest.approx(0.9)


# ── Persistence ────────────────────────────────────────────────────────────────

def test_store_upsert_called_on_eval():
    store = MagicMock()
    store.list_module_thresholds.return_value = []
    mgr = AdaptiveThresholdManager(store=store)
    mgr.on_eval("ema_m5", 0.8)
    store.upsert_module_threshold.assert_called_once()
    call_kwargs = store.upsert_module_threshold.call_args.kwargs
    assert call_kwargs["module_id"] == "ema_m5"
    assert call_kwargs["eval_count"] == 1
    assert call_kwargs["correct_count"] == 1


def test_store_error_does_not_crash_runner():
    store = MagicMock()
    store.list_module_thresholds.return_value = []
    store.upsert_module_threshold.side_effect = RuntimeError("DB full")
    mgr = AdaptiveThresholdManager(store=store)
    mgr.on_eval("m", 0.7)   # must not raise


# ── Signal filtering ───────────────────────────────────────────────────────────

def test_filter_passes_signals_above_threshold():
    mgr = AdaptiveThresholdManager(store=None, default_threshold=0.60)
    signals = [
        _signal("mod_a", confidence=0.65),
        _signal("mod_b", confidence=0.55),
        _signal("mod_c", confidence=0.60),  # exactly at threshold — included
    ]
    accepted = mgr.filter_signals(signals)
    assert len(accepted) == 2
    assert all(s.module_id != "mod_b" for s in accepted)


def test_filter_empty_when_all_below():
    mgr = AdaptiveThresholdManager(store=None, default_threshold=0.80)
    signals = [_signal("m", confidence=0.55)]
    assert mgr.filter_signals(signals) == []


def test_filter_passes_all_for_new_modules():
    mgr = AdaptiveThresholdManager(store=None, default_threshold=0.60)
    signals = [_signal("new_module", confidence=0.62)]
    assert len(mgr.filter_signals(signals)) == 1


def test_filter_uses_per_module_threshold():
    store = MagicMock()
    store.list_module_thresholds.return_value = [
        {"module_id": "trusted", "threshold": 0.45, "eval_count": 100, "correct_count": 80, "mean_score": 0.75},
        {"module_id": "unreliable", "threshold": 0.85, "eval_count": 100, "correct_count": 20, "mean_score": 0.30},
    ]
    mgr = AdaptiveThresholdManager(store=store)
    signals = [
        _signal("trusted", confidence=0.50),        # above 0.45 — in
        _signal("unreliable", confidence=0.80),     # below 0.85 — out
    ]
    accepted = mgr.filter_signals(signals)
    assert len(accepted) == 1
    assert accepted[0].module_id == "trusted"


# ── Reset ──────────────────────────────────────────────────────────────────────

def test_reset_module_restores_default():
    mgr = AdaptiveThresholdManager(store=None, default_threshold=0.60)
    for _ in range(30):
        mgr.on_eval("m", 1.0)
    assert mgr.get_threshold("m") < 0.60
    mgr.reset_module("m")
    assert mgr.get_threshold("m") == pytest.approx(0.60)
