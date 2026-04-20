"""MongoDB progress store — worker writes, master reads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from metatrade.core.log import get_logger

log = get_logger(__name__)

_COLLECTION = "training_progress"


class MongoProgressStore:
    """Sync adaptive_progress.json ↔ MongoDB training_progress collection."""

    def __init__(self, db: Any) -> None:
        self._col = db[_COLLECTION]

    # ── Worker side ───────────────────────────────────────────────────────────

    def init(self, job: dict[str, Any]) -> None:
        """Create initial progress document for a job."""
        doc = {
            "_id": job["_id"],
            "job_id": job["_id"],
            "status": "running",
            "symbol": job["symbol"],
            "timeframe": job["timeframe"],
            "attempts_done": 0,
            "max_attempts": 20,
            "best_holdout": None,
            "target": None,
            "started_at_utc": None,
            "updated_at_utc": None,
            "attempts": [],
            "fold_data": None,
        }
        self._col.replace_one({"_id": job["_id"]}, doc, upsert=True)

    def sync_from_file(self, job_id: str, adaptive_progress_path: Path, fold_progress_path: Path | None = None) -> None:
        """Read local JSON files and push to MongoDB."""
        if not adaptive_progress_path.exists():
            return
        try:
            with open(adaptive_progress_path) as f:
                data = json.load(f)

            fold_data: dict[str, Any] | None = None
            if fold_progress_path and fold_progress_path.exists():
                with open(fold_progress_path) as f:
                    fold_data = json.load(f)

            update: dict[str, Any] = {
                "status": data.get("status", "running"),
                "attempts_done": data.get("attempts_done", 0),
                "max_attempts": data.get("max_attempts", 20),
                "best_holdout": data.get("best_holdout"),
                "target": data.get("target"),
                "started_at_utc": data.get("started_at_utc"),
                "updated_at_utc": data.get("updated_at_utc"),
                "attempts": data.get("attempts", []),
                "fold_data": fold_data,
            }
            self._col.update_one({"_id": job_id}, {"$set": update})
        except Exception as exc:
            log.warning("progress_store_sync_error", job_id=job_id, error=str(exc))

    def mark_completed(self, job_id: str) -> None:
        self._col.update_one({"_id": job_id}, {"$set": {"status": "completed"}})

    def mark_failed(self, job_id: str, error: str) -> None:
        self._col.update_one({"_id": job_id}, {"$set": {"status": "failed", "error": error}})

    # ── Master side ───────────────────────────────────────────────────────────

    def read(self, job_id: str) -> dict[str, Any] | None:
        return self._col.find_one({"_id": job_id})  # type: ignore[return-value]

    def read_active(self, symbol: str | None = None, timeframe: str | None = None) -> dict[str, Any] | None:
        filt: dict[str, Any] = {"status": "running"}
        if symbol:
            filt["symbol"] = symbol
        if timeframe:
            filt["timeframe"] = timeframe
        return self._col.find_one(filt, sort=[("started_at_utc", -1)])  # type: ignore[return-value]
