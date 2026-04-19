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
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from metatrade.alerting.telegram_alerter import TelegramAlerter
from metatrade.core.contracts.market import Bar
from metatrade.core.log import get_logger
from metatrade.ml.config import MLConfig

log = get_logger(__name__)

_TRAIN_SCRIPT = Path(__file__).parent.parent.parent.parent / "scripts" / "train.py"


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
    ) -> None:
        self._cfg = config or MLConfig()
        self._train_args = train_args or []
        self._alerter = alerter
        self._script = train_script or _TRAIN_SCRIPT

        self._bars_since_last: int = 0
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]

        # Wall-clock slot tracking: the UTC datetime of the last slot we fired.
        # Initialised to "now" so we don't immediately re-run a slot that
        # already fired before this process started today.
        self._last_triggered_slot: datetime = datetime.now(UTC)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_training(self) -> bool:
        """True if a training subprocess is currently running."""
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
        if not self._cfg.retrain_enabled:
            return False

        self._bars_since_last += 1

        if self._process is not None and not self.is_training:
            self._harvest_process(bar)

        if self.is_training:
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

    def _launch_raw(self, symbol: str, timeframe: str, advance_slot: bool = True) -> bool:
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
