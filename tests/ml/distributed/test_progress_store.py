"""Tests for MongoProgressStore — uses MagicMock as DB."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from metatrade.ml.distributed.progress_store import MongoProgressStore


def _make_db() -> tuple[MagicMock, MagicMock]:
    db = MagicMock()
    col = MagicMock()
    db.__getitem__ = MagicMock(return_value=col)
    return db, col


class TestMongoProgressStore:
    def test_init_inserts_document(self):
        db, col = _make_db()
        store = MongoProgressStore(db)
        job = {"_id": "job_123", "symbol": "EURUSD", "timeframe": "M1"}
        store.init(job)
        col.replace_one.assert_called_once()
        doc = col.replace_one.call_args[0][1]
        assert doc["job_id"] == "job_123"
        assert doc["status"] == "running"
        assert doc["symbol"] == "EURUSD"

    def test_sync_from_file_updates_store(self):
        db, col = _make_db()
        store = MongoProgressStore(db)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            adaptive = p / "adaptive_progress.json"
            adaptive.write_text(json.dumps({
                "status": "running",
                "attempts_done": 5,
                "max_attempts": 20,
                "best_holdout": 0.58,
                "target": 0.60,
                "started_at_utc": "2026-04-20T10:00:00Z",
                "updated_at_utc": "2026-04-20T10:30:00Z",
                "attempts": [],
            }))
            store.sync_from_file("job_123", adaptive)

        col.update_one.assert_called_once()
        update = col.update_one.call_args[0][1]["$set"]
        assert update["attempts_done"] == 5
        assert update["best_holdout"] == 0.58

    def test_sync_from_file_skips_missing_file(self):
        db, col = _make_db()
        store = MongoProgressStore(db)
        store.sync_from_file("job_123", Path("/nonexistent/path.json"))
        col.update_one.assert_not_called()

    def test_mark_completed(self):
        db, col = _make_db()
        store = MongoProgressStore(db)
        store.mark_completed("job_123")
        update = col.update_one.call_args[0][1]["$set"]
        assert update["status"] == "completed"

    def test_mark_failed(self):
        db, col = _make_db()
        store = MongoProgressStore(db)
        store.mark_failed("job_123", "subprocess crashed")
        update = col.update_one.call_args[0][1]["$set"]
        assert update["status"] == "failed"
        assert "subprocess crashed" in update["error"]

    def test_read_delegates_to_find_one(self):
        db, col = _make_db()
        col.find_one.return_value = {"_id": "job_123", "status": "running"}
        store = MongoProgressStore(db)
        result = store.read("job_123")
        assert result is not None
        assert result["status"] == "running"
        col.find_one.assert_called_once_with({"_id": "job_123"})

    def test_sync_includes_fold_data(self):
        db, col = _make_db()
        store = MongoProgressStore(db)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            adaptive = p / "adaptive_progress.json"
            fold = p / "fold_progress.json"
            adaptive.write_text(json.dumps({
                "status": "running", "attempts_done": 1, "max_attempts": 20,
                "attempts": [], "started_at_utc": None, "updated_at_utc": None,
            }))
            fold.write_text(json.dumps({"current_fold": 3, "total_folds": 8}))
            store.sync_from_file("job_123", adaptive, fold)

        update = col.update_one.call_args[0][1]["$set"]
        assert update["fold_data"] == {"current_fold": 3, "total_folds": 8}
