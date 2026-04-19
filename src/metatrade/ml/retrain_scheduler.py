"""Background retraining scheduler.

Launches ``scripts/train.py`` as a subprocess when the configured trigger
fires (every X hours or every X bars).  Only one training process runs at
a time; a new trigger is ignored while training is in progress.

Trigger modes (ML_RETRAIN_TRIGGER env var)
------------------------------------------
    "hours"  — fire every ML_RETRAIN_EVERY_HOURS wall-clock hours.
    "bars"   — fire every ML_RETRAIN_EVERY_BARS bars processed by the runner.

The scheduler is intentionally side-effect-free when disabled
(ML_RETRAIN_ENABLED=false) so it can be instantiated unconditionally.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from metatrade.alerting.telegram_alerter import TelegramAlerter
from metatrade.core.contracts.market import Bar
from metatrade.core.log import get_logger
from metatrade.ml.config import MLConfig

log = get_logger(__name__)

# Path to train.py relative to the package root (two levels up from this file)
_TRAIN_SCRIPT = Path(__file__).parent.parent.parent.parent / "scripts" / "train.py"


class RetrainScheduler:
    """Trigger background retraining at configurable intervals.

    Args:
        config:         MLConfig with retrain_trigger, retrain_every_hours/bars.
        train_args:     Extra CLI arguments passed to train.py (e.g.
                        ["--source", "mt5", "--symbol", "EURUSD", "--multires"]).
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
        self._last_trigger_ts: float = time.monotonic()
        self._process: subprocess.Popen | None = None  # type: ignore[type-arg]

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_training(self) -> bool:
        """True if a training subprocess is currently running."""
        return self._process is not None and self._process.poll() is None

    def on_bar(self, bar: Bar) -> bool:
        """Advance counters and fire retraining if the trigger fires.

        Args:
            bar: Current bar (used for logging context).

        Returns:
            True if a new training process was launched this bar.
        """
        if not self._cfg.retrain_enabled:
            return False

        self._bars_since_last += 1

        # Don't overlap: wait for the previous run to finish
        if self.is_training:
            return False
        # Harvest completed process
        if self._process is not None:
            self._process = None

        if self._should_trigger():
            return self._launch(bar)
        return False

    # ── Private ───────────────────────────────────────────────────────────────

    def _should_trigger(self) -> bool:
        mode = self._cfg.retrain_trigger.lower()
        if mode == "bars":
            return self._bars_since_last >= self._cfg.retrain_every_bars
        # default: hours
        elapsed_hours = (time.monotonic() - self._last_trigger_ts) / 3600.0
        return elapsed_hours >= self._cfg.retrain_every_hours

    def _launch(self, bar: Bar) -> bool:
        if not self._script.exists():
            log.warning("retrain_scheduler_script_not_found", path=str(self._script))
            return False

        cmd = [sys.executable, str(self._script)] + self._train_args
        log.info(
            "retrain_scheduler_launch",
            trigger=self._cfg.retrain_trigger,
            bars_since_last=self._bars_since_last,
            cmd=" ".join(cmd[:6]) + " ...",
        )
        try:
            self._process = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            log.error("retrain_scheduler_launch_failed", error=str(exc))
            return False

        self._bars_since_last = 0
        self._last_trigger_ts = time.monotonic()
        return True
