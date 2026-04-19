"""Model watcher — polls the registry directory for new/better trained snapshots.

Called once per bar from the runner.  When a candidate snapshot is found that
satisfies the PromotionPolicy, the switch is queued and executed as soon as
there are no open positions.

Flow
----
on_bar(bar, open_positions, live_stats)
    1. Check if poll interval has elapsed (time-based debounce)
    2. Scan registry dir for snapshots newer than the current model
    3. Find the best candidate (highest holdout that passes policy)
    4. If switch approved and open_positions == 0 → execute switch
    5. If switch approved but positions open → hold pending switch
    6. Send Telegram alert on switch
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from metatrade.alerting.telegram_alerter import TelegramAlerter
from metatrade.core.contracts.market import Bar
from metatrade.core.log import get_logger
from metatrade.ml.classifier import MLClassifier
from metatrade.ml.config import MLConfig
from metatrade.ml.live_tracker import LiveAccuracyStats
from metatrade.ml.promotion_policy import PromotionDecision, PromotionPolicy
from metatrade.ml.registry import ModelRegistry

log = get_logger(__name__)


@dataclass
class WatcherState:
    """Public state snapshot from the last on_bar() call."""

    last_poll_ts: float = 0.0
    current_version: str | None = None
    current_holdout: float | None = None
    pending_switch_version: str | None = None
    last_switch_ts: float = 0.0
    total_switches: int = 0


class ModelWatcher:
    """Poll the registry directory for new snapshots and promote candidates.

    Args:
        registry:   Live ModelRegistry used by the running MLModule.
        symbol:     Instrument symbol (used to filter snapshot filenames).
        config:     MLConfig (controls poll interval, min holdout, etc.).
        alerter:    Optional TelegramAlerter for switch notifications.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        symbol: str,
        config: MLConfig | None = None,
        alerter: TelegramAlerter | None = None,
    ) -> None:
        self._registry = registry
        self._symbol = symbol
        self._cfg = config or MLConfig()
        self._alerter = alerter
        self._policy = PromotionPolicy(
            min_holdout=self._cfg.candidate_min_holdout,
            degradation_threshold=self._cfg.live_accuracy_warning_below,
        )
        self._state = WatcherState()

        # Seed current version from registry
        active = registry.get_active()
        if active is not None:
            self._state.current_version = active.version
            self._state.current_holdout = active.tags.get("holdout_accuracy") or active.tags.get("best_test_acc")

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> WatcherState:
        return self._state

    def on_bar(
        self,
        bar: Bar,
        open_positions: int,
        live_stats: LiveAccuracyStats | None = None,
    ) -> bool:
        """Evaluate model switch on each bar.

        Args:
            bar:            Current bar (used for logging/alerting context).
            open_positions: Number of currently open positions.
            live_stats:     Current live accuracy stats (from LiveAccuracyTracker).

        Returns:
            True if a model switch was executed this bar.
        """
        # Execute a pending switch when positions close
        if self._state.pending_switch_version is not None and open_positions == 0:
            return self._execute_switch(self._state.pending_switch_version, bar)

        # Debounce: only scan disk every poll_sec seconds
        now = time.monotonic()
        if now - self._state.last_poll_ts < self._cfg.model_watcher_poll_sec:
            return False
        self._state.last_poll_ts = now

        # Live accuracy degradation alert (independent of switch decision)
        if (
            live_stats is not None
            and live_stats.is_reliable
            and live_stats.accuracy is not None
            and live_stats.accuracy < self._cfg.live_accuracy_warning_below
            and self._alerter is not None
        ):
            self._alerter.alert_model_degraded(
                symbol=self._symbol,
                model_version=live_stats.model_version,
                live_accuracy=live_stats.accuracy,
                threshold=self._cfg.live_accuracy_warning_below,
                n_evaluated=live_stats.n_evaluated,
            )

        # Scan for candidates
        decision = self._find_best_candidate(live_stats)
        if not decision.should_switch or decision.candidate_version is None:
            return False

        if open_positions == 0:
            return self._execute_switch(decision.candidate_version, bar)

        # Queue for when positions close
        log.info(
            "model_watcher_switch_pending",
            candidate=decision.candidate_version,
            reason=decision.reason,
            open_positions=open_positions,
        )
        self._state.pending_switch_version = decision.candidate_version
        return False

    # ── Private ───────────────────────────────────────────────────────────────

    def _find_best_candidate(
        self, live_stats: LiveAccuracyStats | None
    ) -> PromotionDecision:
        """Scan disk snapshots and return the best promotable candidate decision."""
        snapshots = ModelRegistry.list_disk_snapshots(
            self._cfg.model_registry_dir, self._symbol
        )
        # Skip the currently active version
        candidates = [
            s for s in snapshots
            if s["version"] != self._state.current_version
        ]
        if not candidates:
            return PromotionDecision(should_switch=False, reason="no new candidates on disk")

        # Evaluate each candidate; take the best-holdout one that triggers a switch
        best_decision: PromotionDecision | None = None
        for snap in candidates:
            holdout = self._extract_holdout(snap)
            if holdout is None:
                continue
            decision = self._policy.evaluate(
                current_holdout=self._state.current_holdout,
                candidate_version=snap["version"],
                candidate_holdout=holdout,
                live_stats=live_stats,
            )
            if decision.should_switch:
                if (
                    best_decision is None
                    or holdout > self._extract_holdout(
                        next(s for s in candidates if s["version"] == best_decision.candidate_version),
                    )
                ):
                    best_decision = decision

        return best_decision or PromotionDecision(
            should_switch=False,
            reason="no candidate passed promotion policy",
        )

    def _execute_switch(self, candidate_version: str, bar: Bar) -> bool:
        """Load candidate from disk, promote, update state, alert."""
        snapshot = self._registry.load_from_disk(self._symbol, candidate_version)
        if snapshot is None:
            log.warning(
                "model_watcher_load_failed",
                candidate=candidate_version,
                symbol=self._symbol,
            )
            self._state.pending_switch_version = None
            return False

        # Re-read metadata for holdout
        disk_snapshots = ModelRegistry.list_disk_snapshots(
            self._cfg.model_registry_dir, self._symbol
        )
        meta = next((s for s in disk_snapshots if s["version"] == candidate_version), {})
        new_holdout = self._extract_holdout(meta)

        old_version = self._state.current_version
        old_holdout = self._state.current_holdout

        self._registry.promote(candidate_version)
        self._state.current_version = candidate_version
        self._state.current_holdout = new_holdout
        self._state.pending_switch_version = None
        self._state.last_switch_ts = time.monotonic()
        self._state.total_switches += 1

        log.info(
            "model_watcher_switched",
            old=old_version,
            new=candidate_version,
            old_holdout=old_holdout,
            new_holdout=new_holdout,
            symbol=self._symbol,
        )

        if self._alerter is not None:
            self._alerter.alert_model_switched(
                symbol=self._symbol,
                old_version=old_version or "none",
                new_version=candidate_version,
                old_holdout=old_holdout,
                new_holdout=new_holdout or 0.0,
                live_accuracy=None,
                reason="model watcher promotion",
            )

        return True

    @staticmethod
    def _extract_holdout(manifest: dict[str, Any]) -> float | None:
        tags = manifest.get("tags", {})
        v = tags.get("holdout_accuracy") or tags.get("best_test_acc")
        return float(v) if v is not None else None
