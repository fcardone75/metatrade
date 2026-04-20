"""MongoDB job queue for distributed ML training."""

from __future__ import annotations

import datetime
from typing import Any

from metatrade.core.log import get_logger

log = get_logger(__name__)

# Status constants
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

STALE_TIMEOUT_MINUTES = 30


def _utcnow() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_job_id(symbol: str, timeframe: str) -> str:
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M")
    return f"job_{symbol.lower()}_{timeframe.lower()}_{ts}"


class MongoJobQueue:
    """Push, poll, and update training jobs in MongoDB."""

    def __init__(self, db: Any) -> None:
        self._col = db["training_jobs"]

    def push_job(
        self,
        symbol: str,
        timeframe: str,
        backend: str,
        train_args: list[str],
        triggered_by: str = "manual",
    ) -> str:
        job_id = _make_job_id(symbol, timeframe)
        doc = {
            "_id": job_id,
            "status": STATUS_PENDING,
            "data_ready": False,       # master sets True after bars upload completes
            "symbol": symbol,
            "timeframe": timeframe,
            "backend": backend,
            "train_args": train_args,
            "data_gridfs_id": None,
            "model_gridfs_id": None,
            "model_version": None,
            "holdout_accuracy": None,
            "triggered_by": triggered_by,
            "created_at": _utcnow(),
            "started_at": None,
            "completed_at": None,
            "error": None,
        }
        self._col.insert_one(doc)
        log.info("job_queue_pushed", job_id=job_id, symbol=symbol, timeframe=timeframe)
        return job_id

    def mark_data_ready(self, job_id: str, gridfs_id: Any) -> None:
        """Call after bars upload completes. Worker will not pick up job until this is called."""
        self._col.update_one(
            {"_id": job_id},
            {"$set": {"data_gridfs_id": gridfs_id, "data_ready": True}},
        )
        log.info("job_queue_data_ready", job_id=job_id)

    def poll_pending(self) -> dict[str, Any] | None:
        """Atomically claim one pending job → running. Returns the job doc or None.

        Only picks up jobs where the master has finished uploading bar data
        (data_ready=True). This prevents the worker from starting before the
        full dataset is available in GridFS.
        """
        now = _utcnow()
        job = self._col.find_one_and_update(
            {"status": STATUS_PENDING, "data_ready": True},
            {"$set": {"status": STATUS_RUNNING, "started_at": now}},
            sort=[("created_at", 1)],
            return_document=True,
        )
        if job:
            log.info("job_queue_claimed", job_id=job["_id"])
        return job  # type: ignore[return-value]

    def complete(self, job_id: str, model_gridfs_id: Any, model_version: str, holdout_accuracy: float) -> None:
        self._col.update_one(
            {"_id": job_id},
            {
                "$set": {
                    "status": STATUS_COMPLETED,
                    "model_gridfs_id": model_gridfs_id,
                    "model_version": model_version,
                    "holdout_accuracy": holdout_accuracy,
                    "completed_at": _utcnow(),
                }
            },
        )
        log.info("job_queue_completed", job_id=job_id, model_version=model_version)

    def fail(self, job_id: str, error: str) -> None:
        self._col.update_one(
            {"_id": job_id},
            {"$set": {"status": STATUS_FAILED, "error": error, "completed_at": _utcnow()}},
        )
        log.warning("job_queue_failed", job_id=job_id, error=error)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job. Returns True if the job was found and cancelled."""
        result = self._col.update_one(
            {"_id": job_id, "status": {"$in": [STATUS_PENDING, STATUS_RUNNING]}},
            {"$set": {"status": STATUS_CANCELLED, "completed_at": _utcnow(), "error": "cancelled by user"}},
        )
        if result.modified_count:
            log.info("job_queue_cancelled", job_id=job_id)
            return True
        return False  # type: ignore[return-value]

    def cancel_active(self, symbol: str | None = None, timeframe: str | None = None) -> list[str]:
        """Cancel all pending/running jobs, optionally filtered by symbol/timeframe.

        Returns list of cancelled job IDs.
        """
        filt: dict[str, Any] = {"status": {"$in": [STATUS_PENDING, STATUS_RUNNING]}}
        if symbol:
            filt["symbol"] = symbol
        if timeframe:
            filt["timeframe"] = timeframe

        docs = list(self._col.find(filt, {"_id": 1}))
        job_ids = [d["_id"] for d in docs]
        if job_ids:
            self._col.update_many(
                {"_id": {"$in": job_ids}},
                {"$set": {"status": STATUS_CANCELLED, "completed_at": _utcnow(), "error": "cancelled by user"}},
            )
            log.info("job_queue_cancelled_all", count=len(job_ids), jobs=job_ids)
        return job_ids  # type: ignore[return-value]

    def get_active_job(self, symbol: str | None = None, timeframe: str | None = None) -> dict[str, Any] | None:
        """Return the most recent pending or running job, optionally filtered by symbol/timeframe."""
        filt: dict[str, Any] = {"status": {"$in": [STATUS_PENDING, STATUS_RUNNING]}}
        if symbol:
            filt["symbol"] = symbol
        if timeframe:
            filt["timeframe"] = timeframe
        return self._col.find_one(filt, sort=[("created_at", -1)])  # type: ignore[return-value]

    def get_latest_completed(self, symbol: str, timeframe: str) -> dict[str, Any] | None:
        return self._col.find_one(  # type: ignore[return-value]
            {"status": STATUS_COMPLETED, "symbol": symbol, "timeframe": timeframe},
            sort=[("completed_at", -1)],
        )

    def reset_stale_jobs(self) -> int:
        """Reset running jobs that haven't been updated in STALE_TIMEOUT_MINUTES."""
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=STALE_TIMEOUT_MINUTES)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
        result = self._col.update_many(
            {"status": STATUS_RUNNING, "started_at": {"$lt": cutoff_str}},
            {"$set": {"status": STATUS_PENDING, "started_at": None}},
        )
        if result.modified_count:
            log.warning("job_queue_reset_stale", count=result.modified_count)
        return result.modified_count  # type: ignore[return-value]
