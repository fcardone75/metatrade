"""Verifica percorsi retrain distribuito: massive sul worker vs mt5 via GridFS."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.ml.config import MLConfig
from metatrade.ml.retrain_scheduler import RetrainScheduler, build_ml_retrain_train_args


def _sample_bar() -> Bar:
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        timestamp_utc=datetime(2026, 4, 23, 12, 0, tzinfo=UTC),
        open=Decimal("1.1000"),
        high=Decimal("1.1005"),
        low=Decimal("1.0995"),
        close=Decimal("1.1002"),
        volume=Decimal("100"),
    )


def test_retrain_remote_massive_skips_gridfs_upload() -> None:
    ml = MLConfig(
        retrain_source="massive",
        retrain_enabled=True,
        backend="histgbm",
    )
    args = build_ml_retrain_train_args(
        ml, symbol="EURUSD", timeframe="M1", model_dir="/tmp/models"
    )
    sched = RetrainScheduler(config=ml, train_args=args)
    sched._mongo_cfg = MagicMock()
    sched._mongo_db = MagicMock()

    with (
        patch("metatrade.ml.distributed.job_queue.MongoJobQueue") as mjq_cls,
        patch("metatrade.ml.distributed.data_transfer.DataTransfer") as dt_cls,
    ):
        mock_queue = MagicMock()
        mock_queue.push_job.return_value = "job_eurusd_m1_test"
        mjq_cls.return_value = mock_queue
        mock_transfer = MagicMock()
        dt_cls.return_value = mock_transfer

        ok = sched._trigger_remote("EURUSD", "M1", advance_slot=False)

    assert ok is True
    mock_transfer.upload_bars.assert_not_called()
    mock_queue.push_job.assert_called_once()
    kw = mock_queue.push_job.call_args[1]
    assert kw["data_mode"] == "massive"
    mock_queue.mark_data_ready.assert_called_once_with("job_eurusd_m1_test", None)


def test_retrain_remote_mt5_uploads_then_marks_gridfs() -> None:
    ml = MLConfig(
        retrain_source="mt5",
        retrain_enabled=True,
        backend="histgbm",
    )
    args = build_ml_retrain_train_args(
        ml, symbol="EURUSD", timeframe="M1", model_dir="/tmp/models"
    )
    sched = RetrainScheduler(config=ml, train_args=args)
    sched._mongo_cfg = MagicMock()
    sched._mongo_db = MagicMock()
    sched._bar_buffer.append(_sample_bar())

    with (
        patch("metatrade.ml.distributed.job_queue.MongoJobQueue") as mjq_cls,
        patch("metatrade.ml.distributed.data_transfer.DataTransfer") as dt_cls,
    ):
        mock_queue = MagicMock()
        mock_queue.push_job.return_value = "job_mt5_1"
        mjq_cls.return_value = mock_queue
        mock_transfer = MagicMock()
        mock_transfer.upload_bars.return_value = "gridfs_abc123"
        dt_cls.return_value = mock_transfer

        ok = sched._trigger_remote("EURUSD", "M1", advance_slot=False)

    assert ok is True
    mock_transfer.upload_bars.assert_called_once()
    mock_queue.push_job.assert_called_once()
    kw = mock_queue.push_job.call_args[1]
    assert kw["data_mode"] == "gridfs"
    mock_queue.mark_data_ready.assert_called_once_with("job_mt5_1", "gridfs_abc123")
