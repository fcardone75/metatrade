"""Background retraining scheduler.

Launches ``scripts/train.py`` as a subprocess at configured daily time slots.
Only one training process runs at a time.

Trigger modes (ML_RETRAIN_TRIGGER env var)
------------------------------------------
    "hours"  — fire at fixed UTC time slots.
               First slot = ML_RETRAIN_SCHEDULE_START_HOUR (default 09:00 UTC).
               Subsequent slots every ML_RETRAIN_EVERY_HOURS hours.
               Example: start=9, every=3 → 09:00, 12:00, 15:00, 18:00, 21:00 UTC.
               Weekends are skipped (no market = no point training on stale bars).
               On restart the scheduler picks up from the next due slot, so it
               never duplicates a run that already happened earlier in the day.

    "bars"   — fire every ML_RETRAIN_EVERY_BARS bars processed by the runner.
               Useful for backtests or when wall-clock time is less meaningful.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections import deque
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from metatrade.alerting.telegram_alerter import TelegramAlerter
from metatrade.core.contracts.market import Bar
from metatrade.core.log import get_logger
from metatrade.ml.config import MLConfig

if TYPE_CHECKING:
    from metatrade.ml.registry import ModelRegistry

log = get_logger(__name__)

_TRAIN_SCRIPT = Path(__file__).parent.parent.parent.parent / "scripts" / "train.py"
_BAR_BUFFER_SIZE = 50_000  # ~34 days of M1 bars


def _is_weekend(dt: datetime) -> bool:
    """True if dt falls on Saturday or Sunday (UTC)."""
    return dt.weekday() >= 5  # 5=Sat, 6=Sun


def _extract_arg(args: list[str], flag: str) -> str | None:
    """Return the value after ``flag`` in an argparse-style list, or None."""
    try:
        idx = args.index(flag)
        return args[idx + 1]
    except (ValueError, IndexError):
        return None


def _daily_slots(day: date, start_hour: int, interval_hours: int) -> list[datetime]:
    """Return all UTC training slots for a given calendar day."""
    slots = []
    h = start_hour
    while h < 24:
        slots.append(datetime(day.year, day.month, day.day, h, 0, 0, tzinfo=UTC))
        h += interval_hours
    return slots


def next_scheduled_slot(after: datetime, start_hour: int, interval_hours: int) -> datetime:
    """Return the first trading-day slot strictly after ``after``.

    Skips Saturday and Sunday entirely.
    """
    candidate = after
    for _ in range(10):  # search up to 10 days forward (handles long weekends)
        for slot in _daily_slots(candidate.date(), start_hour, interval_hours):
            if slot > after and not _is_weekend(slot):
                return slot
        candidate = datetime(
            candidate.year, candidate.month, candidate.day, 0, 0, 0, tzinfo=UTC
        ) + timedelta(days=1)
    # Fallback — should never happen
    return after + timedelta(hours=interval_hours)


class RetrainScheduler:
    """Trigger background retraining at fixed daily time slots.

    Args:
        config:         MLConfig (retrain_trigger, retrain_every_hours,
                        retrain_schedule_start_hour, retrain_every_bars).
        train_args:     Extra CLI arguments passed to train.py (e.g.
                        ["--source", "mt5", "--symbol", "EURUSD", "--auto-tune"]).
        alerter:        Optional TelegramAlerter for retrain notifications.
        train_script:   Override path to train.py (default: auto-detected).
    """

    def __init__(
        self,
        config: MLConfig | None = None,
        train_args: list[str] | None = None,
        alerter: TelegramAlerter | None = None,
        train_script: Path | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self._cfg = config or MLConfig()
        self._train_args = train_args or []
        self._alerter = alerter
        self._script = train_script or _TRAIN_SCRIPT
        self._model_registry = model_registry

        self._bars_since_last: int = 0
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]

        # Wall-clock slot tracking: the UTC datetime of the last slot we fired.
        # Initialised to "now" so we don't immediately re-run a slot that
        # already fired before this process started today.
        self._last_triggered_slot: datetime = datetime.now(UTC)

        # ── Bar buffer (for remote training — uploaded to GridFS) ──────────────
        self._bar_buffer: deque[Bar] = deque(maxlen=_BAR_BUFFER_SIZE)

        # ── Distributed training state ─────────────────────────────────────────
        self._mongo_cfg: Any = None
        self._mongo_db: Any = None
        self._remote_job_id: str | None = None
        self._remote_best_precision_buy: float = 0.0
        self._remote_best_precision_sell: float = 0.0
        self._init_mongo()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_training(self) -> bool:
        """True if a training process is currently running (local or remote)."""
        if self._remote_job_id is not None:
            return self._is_remote_job_active()
        return self._process is not None and self._process.poll() is None

    @property
    def next_slot(self) -> datetime:
        """UTC datetime of the next scheduled training slot."""
        return next_scheduled_slot(
            self._last_triggered_slot,
            self._cfg.retrain_schedule_start_hour,
            self._cfg.retrain_every_hours,
        )

    def on_bar(self, bar: Bar) -> bool:
        """Advance counters and fire retraining if the trigger fires.

        Args:
            bar: Current bar — provides the UTC timestamp for slot matching.

        Returns:
            True if a new training process was launched this bar.
        """
        self._bar_buffer.append(bar)

        if not self._cfg.retrain_enabled:
            return False

        self._bars_since_last += 1

        if self._process is not None and not (self._process.poll() is None):
            self._harvest_process(bar)

        if self.is_training:
            # Check if a remote job just completed
            if self._remote_job_id:
                self._maybe_download_completed_model()
                self._check_precision_notifications()
            return False

        if self._should_trigger(bar.timestamp_utc):
            return self._launch(bar)
        return False

    # ── Private ───────────────────────────────────────────────────────────────

    def _harvest_process(self, bar: Bar) -> None:
        """Check exit code of completed process, alert on failure, clean up."""
        exit_code = self._process.poll()
        symbol = bar.symbol
        tf = str(bar.timeframe.value) if hasattr(bar.timeframe, "value") else str(bar.timeframe)
        if exit_code != 0 and self._alerter is not None:
            self._alerter.alert_retrain_failed(
                symbol=symbol,
                timeframe=tf,
                exit_code=exit_code if exit_code is not None else -1,
            )
            log.warning(
                "retrain_scheduler_process_failed",
                exit_code=exit_code,
                symbol=symbol,
            )
        self._process = None

    def _should_trigger(self, bar_utc: datetime) -> bool:
        if self._cfg.retrain_trigger.lower() == "bars":
            return self._bars_since_last >= self._cfg.retrain_every_bars

        # Slot-based: trigger if bar time has passed the next scheduled slot
        # AND we're not on a weekend (no new bars to train on anyway)
        if _is_weekend(bar_utc):
            return False
        return bar_utc >= self.next_slot

    def trigger_now(self) -> bool:
        """Force an immediate training run, bypassing the schedule.

        Returns True if the subprocess was launched successfully.
        """
        if self.is_training:
            return False
        symbol = _extract_arg(self._train_args, "--symbol") or "manual"
        timeframe = _extract_arg(self._train_args, "--timeframe") or "manual"
        return self._launch_raw(symbol=symbol, timeframe=timeframe, advance_slot=False)

    def stop_training(self) -> bool:
        """Terminate the running training subprocess.

        Returns True if a process was running and was killed, False otherwise.
        """
        if self._process is None or self._process.poll() is not None:
            return False
        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except Exception:
            self._process.kill()
        log.warning("retrain_scheduler_stopped_by_user")
        return True

    def _launch(self, bar: Bar) -> bool:
        symbol = bar.symbol
        tf = str(bar.timeframe.value) if hasattr(bar.timeframe, "value") else str(bar.timeframe)
        return self._launch_raw(symbol=symbol, timeframe=tf, advance_slot=True)

    # ── Distributed training (master side) ───────────────────────────────────

    def _init_mongo(self) -> None:
        try:
            from metatrade.ml.distributed.config import MongoTrainConfig
            cfg = MongoTrainConfig()
            if not (cfg.is_enabled and cfg.is_master and cfg.mongo_uri):
                return
            try:
                import pymongo  # type: ignore[import-untyped]
            except ImportError:
                log.error(
                    "retrain_scheduler_mongo_pymongo_missing",
                    hint="Run: pip install 'pymongo[srv]>=4.6'",
                )
                return
            try:
                client: Any = pymongo.MongoClient(cfg.mongo_uri, serverSelectionTimeoutMS=5_000)
                client.admin.command("ping")
            except Exception as exc:
                log.error("retrain_scheduler_mongo_connection_failed", error=str(exc))
                return
            self._mongo_cfg = cfg
            self._mongo_db = client[cfg.mongo_db]
            log.info("retrain_scheduler_mongo_connected", db=cfg.mongo_db)
        except Exception as exc:
            log.error("retrain_scheduler_mongo_init_error", error=str(exc))

    def _trigger_remote(self, symbol: str, timeframe: str, advance_slot: bool) -> bool:
        from metatrade.ml.distributed.data_transfer import DataTransfer
        from metatrade.ml.distributed.job_queue import MongoJobQueue

        backend = _extract_arg(self._train_args, "--backend") or self._cfg.backend
        queue = MongoJobQueue(self._mongo_db)
        transfer = DataTransfer(self._mongo_db)

        bars = list(self._bar_buffer)
        if not bars:
            log.warning("retrain_scheduler_remote_no_bars", symbol=symbol)
            return False
        if len(bars) < 500:
            log.warning(
                "retrain_scheduler_remote_few_bars",
                count=len(bars),
                hint="Run for longer before retraining, or increase --warmup-bars",
            )

        try:
            job_id = queue.push_job(
                symbol=symbol,
                timeframe=timeframe,
                backend=backend,
                train_args=self._train_args,
                triggered_by="scheduled" if advance_slot else "manual",
            )
            # Upload bars and mark data_ready=True only after upload completes.
            # Worker's poll_pending() checks data_ready, so it cannot start before this.
            gridfs_id = transfer.upload_bars(job_id, symbol, timeframe, bars)
            queue.mark_data_ready(job_id, gridfs_id)
            self._remote_job_id = job_id
        except Exception as exc:
            log.error("retrain_scheduler_remote_push_failed", error=str(exc))
            return False

        self._bars_since_last = 0
        if advance_slot:
            self._last_triggered_slot = self.next_slot

        if self._alerter is not None:
            unit_val = (
                self._cfg.retrain_every_bars
                if self._cfg.retrain_trigger.lower() == "bars"
                else self._cfg.retrain_every_hours
            )
            self._alerter.alert_retrain_started(
                symbol=symbol,
                timeframe=timeframe,
                trigger=self._cfg.retrain_trigger,
                bars_or_hours=unit_val,
            )
        log.info("retrain_scheduler_remote_job_queued", job_id=self._remote_job_id)
        return True

    def _is_remote_job_active(self) -> bool:
        if self._mongo_db is None or self._remote_job_id is None:
            return False
        try:
            from metatrade.ml.distributed.job_queue import STATUS_PENDING, STATUS_RUNNING, MongoJobQueue
            queue = MongoJobQueue(self._mongo_db)
            job = queue._col.find_one({"_id": self._remote_job_id, "status": {"$in": [STATUS_PENDING, STATUS_RUNNING]}})
            return job is not None
        except Exception:
            return False

    def _maybe_download_completed_model(self) -> None:
        if self._mongo_db is None or self._remote_job_id is None:
            return
        try:
            from metatrade.ml.distributed.data_transfer import DataTransfer
            from metatrade.ml.distributed.job_queue import STATUS_COMPLETED, MongoJobQueue
            queue = MongoJobQueue(self._mongo_db)
            job = queue._col.find_one({"_id": self._remote_job_id, "status": STATUS_COMPLETED})
            if job is None:
                return

            transfer = DataTransfer(self._mongo_db)
            model_bytes = transfer.download_model(job["model_gridfs_id"])

            if self._model_registry is not None:
                from metatrade.ml.classifier import MLClassifier
                classifier = MLClassifier.deserialize(model_bytes, self._cfg)
                snapshot = self._model_registry.register(
                    classifier,
                    symbol=job["symbol"],
                    version=job.get("model_version"),
                    tags={"backend": job.get("backend"), "holdout": job.get("holdout_accuracy"), "source": "remote"},
                )
                self._model_registry.promote(snapshot.version)
                log.info("retrain_scheduler_remote_model_loaded", version=snapshot.version)

            if self._alerter is not None:
                self._alerter.alert_retrain_complete(
                    symbol=job["symbol"],
                    timeframe=job["timeframe"],
                    accuracy=job.get("holdout_accuracy") or 0.0,
                    version=job.get("model_version") or "",
                )
            self._remote_job_id = None
        except Exception as exc:
            log.error("retrain_scheduler_model_download_failed", error=str(exc))

    def _check_precision_notifications(self) -> None:
        """Read latest progress from MongoDB and alert if BUY or SELL precision improved."""
        if self._mongo_db is None or self._remote_job_id is None or self._alerter is None:
            return
        try:
            col = self._mongo_db["training_progress"]
            doc = col.find_one({"_id": self._remote_job_id})
            if doc is None:
                return
            attempts = doc.get("attempts") or []
            if not attempts:
                return
            latest = attempts[-1]
            cur_buy: float = latest.get("precision_buy") or 0.0
            cur_sell: float = latest.get("precision_sell") or 0.0
            if cur_buy > self._remote_best_precision_buy or cur_sell > self._remote_best_precision_sell:
                self._remote_best_precision_buy = max(self._remote_best_precision_buy, cur_buy)
                self._remote_best_precision_sell = max(self._remote_best_precision_sell, cur_sell)
                target_buy: float = self._cfg.target_buy_precision
                target_sell: float = self._cfg.target_sell_precision
                self._alerter.alert_training_precision_improved(
                    symbol=doc.get("symbol", "?"),
                    timeframe=doc.get("timeframe", "?"),
                    attempt=latest.get("attempt", len(attempts)),
                    precision_buy=cur_buy if cur_buy else None,
                    precision_sell=cur_sell if cur_sell else None,
                    target_buy=target_buy,
                    target_sell=target_sell,
                    holdout=latest.get("holdout"),
                )
        except Exception as exc:
            log.debug("precision_notification_check_failed", error=str(exc))

    def get_remote_job_id(self) -> str | None:
        return self._remote_job_id

    def get_mongo_db(self) -> Any:
        return self._mongo_db

    def cancel_remote_job(self) -> list[str]:
        """Cancel all active remote jobs on MongoDB and reset local state.

        Returns list of cancelled job IDs (empty if mongo not enabled or no active jobs).
        """
        if self._mongo_db is None:
            return []
        from metatrade.ml.distributed.job_queue import MongoJobQueue
        queue = MongoJobQueue(self._mongo_db)
        symbol = _extract_arg(self._train_args, "--symbol")
        timeframe = _extract_arg(self._train_args, "--timeframe")
        cancelled = queue.cancel_active(symbol=symbol, timeframe=timeframe)
        if cancelled:
            self._remote_job_id = None
        return cancelled

    # ── Local subprocess launch ────────────────────────────────────────────────

    def _launch_raw(self, symbol: str, timeframe: str, advance_slot: bool = True) -> bool:
        if self._mongo_cfg is not None:
            return self._trigger_remote(symbol, timeframe, advance_slot)

        if not self._script.exists():
            log.warning("retrain_scheduler_script_not_found", path=str(self._script))
            return False

        cmd = [sys.executable, str(self._script)] + self._train_args
        log.info(
            "retrain_scheduler_launch",
            trigger=self._cfg.retrain_trigger,
            bars_since_last=self._bars_since_last,
            now_utc=datetime.now(UTC).isoformat(),
            cmd=" ".join(cmd[:6]) + " ...",
        )
        # Redirect training output to a rotating log file so errors are visible
        # even when the scheduler runs in background (stdout/stderr are not a tty).
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        train_log = log_dir / "retrain.log"
        try:
            log_fh = open(train_log, "a", encoding="utf-8")  # noqa: SIM115
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=log_fh,
                stderr=log_fh,
            )
        except OSError as exc:
            log.error("retrain_scheduler_launch_failed", error=str(exc))
            return False

        self._bars_since_last = 0
        # Record which slot we just consumed so next_slot advances correctly.
        # Manual triggers (advance_slot=False) must not consume a scheduled slot.
        if advance_slot:
            self._last_triggered_slot = self.next_slot

        if self._alerter is not None:
            unit_val = (
                self._cfg.retrain_every_bars
                if self._cfg.retrain_trigger.lower() == "bars"
                else self._cfg.retrain_every_hours
            )
            self._alerter.alert_retrain_started(
                symbol=symbol,
                timeframe=timeframe,
                trigger=self._cfg.retrain_trigger,
                bars_or_hours=unit_val,
            )

        return True
