"""Tests for the Telegram /trained model-history command."""

from __future__ import annotations

import json

from metatrade.runner.base import BaseRunner
from tests.runner.test_runners import make_runner_config


class FakeTelemetry:
    def __init__(self) -> None:
        self.artifacts = [
            {
                "version": "v_active_m1",
                "symbol": "EURUSD",
                "timeframe": "M1",
                "created_at": 1_776_410_640,
                "source": "csv",
                "train_bars": 1000,
                "n_folds": 4,
                "mean_test_accuracy": 0.41,
                "best_test_accuracy": 0.55,
                "is_active": 1,
                "artifact_path": "data/models/EURUSD_v_active_m1.pkl",
                "details": json.dumps(
                    {
                        "best_fold": 2,
                        "holdout_accuracy": 0.53,
                    }
                ),
            },
            {
                "version": "v_old_m5",
                "symbol": "EURUSD",
                "timeframe": "M5",
                "created_at": 1_776_300_000,
                "source": "csv",
                "train_bars": 1000,
                "n_folds": 4,
                "mean_test_accuracy": 0.52,
                "best_test_accuracy": 0.61,
                "is_active": 0,
                "artifact_path": "data/models/EURUSD_v_old_m5.pkl",
                "details": "{}",
            },
        ]
        self.runs = [
            {
                "training_id": "train-1",
                "started_at": 1_776_410_640,
                "completed_at": 1_776_410_640,
                "status": "completed",
                "symbol": "EURUSD",
                "timeframe": "M1",
                "mean_test_accuracy": 0.41,
                "best_test_accuracy": 0.55,
                "best_fold": 2,
                "model_version": "v_active_m1",
                "details": "{}",
            }
        ]

    def list_module_thresholds(self) -> list[dict]:
        return []

    def list_model_artifacts(self, *, limit: int = 20) -> list[dict]:
        return self.artifacts[:limit]

    def list_training_runs(self, *, limit: int = 20) -> list[dict]:
        return self.runs[:limit]


def _runner(telemetry: FakeTelemetry) -> BaseRunner:
    return BaseRunner(
        config=make_runner_config(),
        modules=[],
        telemetry=telemetry,  # type: ignore[arg-type]
    )


def test_trained_command_lists_saved_and_discarded_models(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_MODEL_REGISTRY_DIR", str(tmp_path))
    reports = tmp_path / "training_reports"
    reports.mkdir()
    (reports / "summary_20260421_175602.json").write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-04-21T15:56:02+00:00",
                "symbol": "EURUSD",
                "min_accuracy": 0.52,
                "runs": [
                    {
                        "timeframe": "M1",
                        "success": False,
                        "error": "Test accuracy sotto la soglia min_accuracy",
                        "best_test_accuracy": 0.436,
                        "best_fold_index": 3,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    reply = _runner(FakeTelemetry())._cmd_trained("")

    assert "ACTIVE" in reply
    assert "v_active_m1" in reply
    assert "SCARTATO" in reply
    assert "43.6%" in reply
    assert "Test accuracy sotto la soglia" in reply


def test_trained_command_active_filter_excludes_failed_and_old(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_MODEL_REGISTRY_DIR", str(tmp_path))
    (tmp_path / "training_reports").mkdir()

    reply = _runner(FakeTelemetry())._cmd_trained("active")

    assert "ACTIVE" in reply
    assert "v_active_m1" in reply
    assert "v_old_m5" not in reply
    assert "SCARTATO" not in reply


def test_trained_command_timeframe_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_MODEL_REGISTRY_DIR", str(tmp_path))

    reply = _runner(FakeTelemetry())._cmd_trained("m5")

    assert "v_old_m5" in reply
    assert "v_active_m1" not in reply
