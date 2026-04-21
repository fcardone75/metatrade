"""Worker daemon — polls MongoDB for pending training jobs and executes them."""

from __future__ import annotations

import concurrent.futures
import io
import json
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from metatrade.core.log import get_logger
from metatrade.ml.distributed.config import MongoTrainConfig
from metatrade.ml.distributed.data_transfer import DataTransfer
from metatrade.ml.distributed.job_queue import MongoJobQueue
from metatrade.ml.distributed.progress_store import MongoProgressStore

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

_TRAIN_SCRIPT = Path(__file__).parent.parent.parent.parent.parent / "scripts" / "train.py"
_PROGRESS_SYNC_INTERVAL = 3  # seconds between file→MongoDB syncs
_JOBS_BASE_DIR = Path.home() / ".metatrade" / "worker_jobs"


def _job_work_dir(job_id: str) -> Path:
    """Return a persistent directory for a job, creating it if needed."""
    d = _JOBS_BASE_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _connect(cfg: MongoTrainConfig) -> Any:
    """Return a pymongo Database instance."""
    try:
        import pymongo  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "pymongo is required for distributed training. "
            "Install it with: pip install 'pymongo[srv]>=4.6'"
        ) from exc
    client: Any = pymongo.MongoClient(cfg.mongo_uri, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    return client[cfg.mongo_db]


def _sync_loop(
    job_id: str,
    db: Any,
    progress_store: MongoProgressStore,
    adaptive_path: Path,
    fold_path: Path,
    stop_event: threading.Event,
    cancel_event: threading.Event,
) -> None:
    """Background thread: sync progress to MongoDB and detect remote cancellation."""
    from metatrade.ml.distributed.job_queue import STATUS_CANCELLED

    while not stop_event.is_set():
        stop_event.wait(timeout=_PROGRESS_SYNC_INTERVAL)
        try:
            progress_store.sync_from_file(job_id, adaptive_path, fold_path)
        except Exception as exc:
            log.warning("worker_sync_error", job_id=job_id, error=str(exc))

        # Check if master has cancelled this job
        if not cancel_event.is_set():
            try:
                col = db["training_jobs"]
                doc = col.find_one({"_id": job_id, "status": STATUS_CANCELLED}, {"_id": 1})
                if doc is not None:
                    log.info("worker_job_cancelled_detected", job_id=job_id)
                    cancel_event.set()
            except Exception:
                pass


def _find_best_model(work_dir: Path, symbol: str) -> Path | None:
    """Return the newest .pkl file in work_dir for the given symbol, or None."""
    candidates = sorted(work_dir.glob(f"models/{symbol}_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(work_dir.glob("**/*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _run_job(
    job: dict[str, Any],
    db: Any,
    work_dir: Path,
) -> None:
    job_id: str = job["_id"]
    symbol: str = job["symbol"]
    timeframe: str = job["timeframe"]

    queue = MongoJobQueue(db)
    progress_store = MongoProgressStore(db)
    transfer = DataTransfer(db)

    # ── Download bar data (skip if already present from a previous run) ──────
    csv_path = work_dir / f"{symbol}_{timeframe}_{job_id}.csv"
    if csv_path.exists():
        log.info("worker_bars_reusing_cached", job_id=job_id, path=str(csv_path))
    else:
        try:
            row_count = transfer.bars_to_csv_file(job["data_gridfs_id"], str(csv_path))
            log.info("worker_bars_downloaded", job_id=job_id, rows=row_count)
        except Exception as exc:
            queue.fail(job_id, f"bar download failed: {exc}")
            return

    # ── Detect resume: how many adaptive attempts already completed? ──────────
    adaptive_path = work_dir / "models" / "adaptive_progress.json"
    skip_attempts = 0
    if adaptive_path.exists():
        try:
            prev = json.loads(adaptive_path.read_text())
            skip_attempts = int(prev.get("attempts_done") or 0)
            if skip_attempts:
                log.info("worker_resuming_job", job_id=job_id, skip_attempts=skip_attempts)
        except Exception:
            skip_attempts = 0

    # ── Init / re-init progress doc ───────────────────────────────────────────
    progress_store.init(job)

    # ── Build train.py command ────────────────────────────────────────────────
    # Drop all environment-specific flags from the master's train_args
    # (source, file, symbol, timeframe, model-dir are all master-side paths or
    # data sources that are irrelevant on the worker). Inject correct values
    # from the job document and the worker's own temp directory.
    _ENV_FLAGS = {"--source", "--file", "--symbol", "--timeframe", "--model-dir"}
    base_args: list[str] = list(job.get("train_args", []))
    filtered: list[str] = []
    skip_next = False
    for arg in base_args:
        if skip_next:
            skip_next = False
            continue
        if arg in _ENV_FLAGS:
            skip_next = True
            continue
        filtered.append(arg)

    model_dir = work_dir / "models"
    model_dir.mkdir(exist_ok=True)
    fold_path = model_dir / "training_progress.json"

    resume_args = ["--adaptive-skip-attempts", str(skip_attempts)] if skip_attempts > 0 else []

    # Inject --backend from job doc (overrides whatever was in train_args)
    job_backend = job.get("backend")
    backend_args = ["--backend", job_backend] if job_backend else []

    cmd = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--source", "csv",
        "--file", str(csv_path),
        "--symbol", symbol,
        "--timeframe", timeframe,
        "--model-dir", str(model_dir),
        *filtered,
        *backend_args,
        *resume_args,
    ]

    # ── Launch sync + cancel-watch thread ────────────────────────────────────
    stop_event = threading.Event()
    cancel_event = threading.Event()
    sync_thread = threading.Thread(
        target=_sync_loop,
        args=(job_id, db, progress_store, adaptive_path, fold_path, stop_event, cancel_event),
        daemon=True,
    )
    sync_thread.start()

    # ── Run training subprocess (non-blocking Popen so we can cancel it) ──────
    log_buf = io.open(work_dir / "train.log", "w", encoding="utf-8")
    exit_code: int | None = None
    cancelled = False
    try:
        log.info(
            "worker_training_start",
            job_id=job_id,
            symbol=symbol,
            timeframe=timeframe,
            cmd=" ".join(cmd[:8]) + " ...",
        )
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=log_buf,
            stderr=log_buf,
            cwd=str(work_dir),
        )
        # Poll until subprocess finishes or cancellation is requested
        while proc.poll() is None:
            if cancel_event.is_set():
                log.info("worker_cancelling_subprocess", job_id=job_id)
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except Exception:
                    proc.kill()
                cancelled = True
                break
            time.sleep(2)

        exit_code = proc.returncode if not cancelled else -1
    except Exception as exc:
        stop_event.set()
        sync_thread.join(timeout=15)
        queue.fail(job_id, f"subprocess error: {exc}")
        return
    finally:
        log_buf.close()

    stop_event.set()
    sync_thread.join(timeout=15)

    if cancelled:
        log.info("worker_job_cancelled", job_id=job_id)
        progress_store.mark_failed(job_id, "cancelled by user")
        return  # job already has status=cancelled set by master; nothing else to do

    # Final sync
    progress_store.sync_from_file(job_id, adaptive_path, fold_path)

    if exit_code != 0:
        # Read last 50 lines of train.log to surface the error in the worker log
        train_log_path = work_dir / "train.log"
        tail = ""
        try:
            lines = train_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-50:])
        except Exception:
            pass
        log.warning("worker_training_failed", job_id=job_id, exit_code=exit_code, tail=tail)
        queue.fail(job_id, f"train.py exited with code {exit_code}")
        progress_store.mark_failed(job_id, f"exit_code={exit_code}")
        return

    # ── Upload model ──────────────────────────────────────────────────────────
    model_path = _find_best_model(work_dir, symbol)
    if model_path is None:
        queue.fail(job_id, "no model .pkl found after training")
        progress_store.mark_failed(job_id, "no_model")
        return

    version = f"v{datetime.now(UTC).strftime('%Y%m%d_%H%M')}_{timeframe}"
    try:
        model_bytes = model_path.read_bytes()
        model_gridfs_id = transfer.upload_model(job_id, symbol, version, model_bytes)
    except Exception as exc:
        queue.fail(job_id, f"model upload failed: {exc}")
        return

    # Extract result_type, holdout, signal_precision from adaptive_progress.json
    holdout: float = 0.0
    result_type: str = "full_success"
    signal_precision: float | None = None
    if adaptive_path.exists():
        try:
            ap = json.loads(adaptive_path.read_text())
            holdout = float(ap.get("best_holdout") or 0.0)
            result_type = ap.get("result_type") or "full_success"
            # Best signal_precision across all attempts
            attempts = ap.get("attempts") or []
            precs = [a.get("signal_precision") for a in attempts if a.get("signal_precision") is not None]
            if precs:
                signal_precision = max(precs)
        except Exception:
            pass

    queue.complete(job_id, model_gridfs_id, version, holdout, result_type=result_type, signal_precision=signal_precision)
    progress_store.mark_completed(job_id)
    log.info("worker_job_done", job_id=job_id, version=version, holdout=holdout)


class WorkerDaemon:
    """Polls MongoDB for pending training jobs and executes them sequentially."""

    def __init__(self, cfg: MongoTrainConfig | None = None) -> None:
        self._cfg = cfg or MongoTrainConfig()

    def run(self) -> None:
        """Blocking poll loop. Runs until interrupted."""
        max_parallel = max(1, self._cfg.ml_worker_max_parallel_jobs)
        log.info("worker_daemon_start", poll_sec=self._cfg.ml_worker_poll_sec, max_parallel=max_parallel)
        db = _connect(self._cfg)
        queue = MongoJobQueue(db)

        # On startup, immediately reset any jobs stuck in RUNNING state so they
        # are re-picked-up as resumable PENDING jobs (don't wait for stale timeout).
        reset_count = queue.reset_running_jobs()
        if reset_count:
            log.info("worker_reset_interrupted_jobs", count=reset_count)

        def _run_and_cleanup(job: dict[str, Any]) -> None:
            work_dir = _job_work_dir(job["_id"])
            try:
                _run_job(job, db, work_dir)
                # Clean up only on success (keep dir on failure for debugging/resume)
                import shutil
                shutil.rmtree(work_dir, ignore_errors=True)
                log.info("worker_workdir_cleaned", job_id=job["_id"])
            except Exception as exc:
                log.error("worker_job_error", job_id=job["_id"], error=str(exc))

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel)
        active_futures: set[concurrent.futures.Future] = set()  # type: ignore[type-arg]

        try:
            while True:
                try:
                    queue.reset_stale_jobs()

                    # Fill up to max_parallel slots
                    active_futures = {f for f in active_futures if not f.done()}
                    while len(active_futures) < max_parallel:
                        job = queue.poll_pending()
                        if job is None:
                            break
                        future = executor.submit(_run_and_cleanup, job)
                        active_futures.add(future)
                        log.info("worker_job_dispatched", job_id=job["_id"], active=len(active_futures))

                    time.sleep(self._cfg.ml_worker_poll_sec)

                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    log.error("worker_daemon_error", error=str(exc))
                    time.sleep(self._cfg.ml_worker_poll_sec)

        except KeyboardInterrupt:
            log.info("worker_daemon_stopping", active_jobs=len(active_futures))
            executor.shutdown(wait=False, cancel_futures=False)
            log.info("worker_daemon_stopped")
