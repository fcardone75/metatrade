"""Tests for MongoJobQueue — uses a MagicMock as DB collection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from metatrade.ml.distributed.job_queue import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    MongoJobQueue,
)


def _make_db() -> MagicMock:
    db = MagicMock()
    col = MagicMock()
    db.__getitem__ = MagicMock(return_value=col)
    db["training_jobs"] = col
    return db, col


class TestMongoJobQueue:
    def test_push_job_inserts_document(self):
        db = MagicMock()
        col = MagicMock()
        db.__getitem__ = MagicMock(return_value=col)
        queue = MongoJobQueue(db)
        job_id = queue.push_job("EURUSD", "M1", "histgbm", ["--symbol", "EURUSD"])
        assert col.insert_one.called
        doc = col.insert_one.call_args[0][0]
        assert doc["status"] == STATUS_PENDING
        assert doc["symbol"] == "EURUSD"
        assert doc["timeframe"] == "M1"
        assert doc["backend"] == "histgbm"
        assert job_id.startswith("job_eurusd_m1_")

    def test_push_job_returns_job_id(self):
        db = MagicMock()
        col = MagicMock()
        db.__getitem__ = MagicMock(return_value=col)
        queue = MongoJobQueue(db)
        job_id = queue.push_job("GBPUSD", "M5", "lightgbm", [])
        assert "gbpusd" in job_id
        assert "m5" in job_id

    def test_poll_pending_calls_find_and_update(self):
        db = MagicMock()
        col = MagicMock()
        col.find_one_and_update.return_value = {"_id": "job_123", "symbol": "EURUSD"}
        db.__getitem__ = MagicMock(return_value=col)
        queue = MongoJobQueue(db)
        result = queue.poll_pending()
        assert result is not None
        assert result["_id"] == "job_123"
        assert col.find_one_and_update.called

    def test_poll_pending_returns_none_when_no_job(self):
        db = MagicMock()
        col = MagicMock()
        col.find_one_and_update.return_value = None
        db.__getitem__ = MagicMock(return_value=col)
        queue = MongoJobQueue(db)
        result = queue.poll_pending()
        assert result is None

    def test_complete_updates_status(self):
        db = MagicMock()
        col = MagicMock()
        db.__getitem__ = MagicMock(return_value=col)
        queue = MongoJobQueue(db)
        queue.complete("job_123", "gridfs_id_abc", "v20260420_1200_M1", 0.62)
        col.update_one.assert_called_once()
        args = col.update_one.call_args
        update = args[0][1]["$set"]
        assert update["status"] == STATUS_COMPLETED
        assert update["holdout_accuracy"] == 0.62
        assert update["model_version"] == "v20260420_1200_M1"

    def test_fail_updates_status(self):
        db = MagicMock()
        col = MagicMock()
        db.__getitem__ = MagicMock(return_value=col)
        queue = MongoJobQueue(db)
        queue.fail("job_123", "no model found")
        update = col.update_one.call_args[0][1]["$set"]
        assert update["status"] == STATUS_FAILED
        assert "no model found" in update["error"]

    def test_get_active_job_queries_pending_and_running(self):
        db = MagicMock()
        col = MagicMock()
        col.find_one.return_value = None
        db.__getitem__ = MagicMock(return_value=col)
        queue = MongoJobQueue(db)
        queue.get_active_job()
        call_args = col.find_one.call_args[0][0]
        assert STATUS_PENDING in call_args["status"]["$in"]
        assert STATUS_RUNNING in call_args["status"]["$in"]

    def test_reset_stale_jobs(self):
        db = MagicMock()
        col = MagicMock()
        col.update_many.return_value = MagicMock(modified_count=2)
        db.__getitem__ = MagicMock(return_value=col)
        queue = MongoJobQueue(db)
        count = queue.reset_stale_jobs()
        assert count == 2
        assert col.update_many.called

    def test_mark_data_ready(self):
        db = MagicMock()
        col = MagicMock()
        db.__getitem__ = MagicMock(return_value=col)
        queue = MongoJobQueue(db)
        queue.mark_data_ready("job_123", "some_gridfs_oid")
        col.update_one.assert_called_once()
        filt, update = col.update_one.call_args[0]
        assert filt == {"_id": "job_123"}
        assert update["$set"]["data_gridfs_id"] == "some_gridfs_oid"
        assert update["$set"]["data_ready"] is True

    def test_poll_pending_filters_data_ready(self):
        db = MagicMock()
        col = MagicMock()
        col.find_one_and_update.return_value = None
        db.__getitem__ = MagicMock(return_value=col)
        queue = MongoJobQueue(db)
        queue.poll_pending()
        filt = col.find_one_and_update.call_args[0][0]
        assert filt["data_ready"] is True

    def test_push_job_sets_data_ready_false(self):
        db = MagicMock()
        col = MagicMock()
        db.__getitem__ = MagicMock(return_value=col)
        queue = MongoJobQueue(db)
        queue.push_job("EURUSD", "M1", "histgbm", [])
        doc = col.insert_one.call_args[0][0]
        assert doc["data_ready"] is False
