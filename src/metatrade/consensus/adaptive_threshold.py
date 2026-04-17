"""AdaptiveThresholdManager — per-module confidence thresholds that self-adjust.

Each analysis module has its own confidence threshold.  A signal whose
confidence is below that threshold is excluded from the consensus vote.

Adjustment rule (called after each market evaluation):

    delta = -alpha × (score - 0.5) × 2

Where ``score`` is the market-accuracy score in [0, 1] produced by
``MarketAccuracyTracker``:

    score ≈ 1.0  →  market moved strongly in the predicted direction
    score ≈ 0.5  →  market was flat or ambiguous
    score ≈ 0.0  →  market moved strongly against the prediction

Effect:
    - Correct signal  (score > 0.5):  delta < 0 → threshold decreases
    - Wrong signal    (score < 0.5):  delta > 0 → threshold increases
    - Neutral         (score ≈ 0.5):  delta ≈ 0 → no change

A module that is consistently right will converge to a low threshold (it
gets included even with modest confidence).  A module that is consistently
wrong will converge to a high threshold (it needs very high conviction to
be included).

The mean_score EMA is stored so the dashboard can show the running
accuracy estimate for each module without re-scanning the decision history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# ── DB protocol ───────────────────────────────────────────────────────────────

class IThresholdStore(Protocol):
    """Minimal storage interface required by AdaptiveThresholdManager."""

    def upsert_module_threshold(
        self,
        module_id: str,
        threshold: float,
        eval_count: int,
        correct_count: int,
        mean_score: float,
    ) -> None: ...

    def list_module_thresholds(self) -> list[dict[str, Any]]: ...


# ── State snapshot ────────────────────────────────────────────────────────────

@dataclass
class ModuleThresholdState:
    """Current state for one module."""

    module_id: str
    threshold: float
    eval_count: int = 0
    correct_count: int = 0
    mean_score: float = 0.5   # EMA of accuracy scores


# ── Manager ───────────────────────────────────────────────────────────────────

class AdaptiveThresholdManager:
    """Maintains and adjusts per-module confidence thresholds.

    Usage in BaseRunner:

        # Create once:
        self._threshold_mgr = AdaptiveThresholdManager(store=telemetry)

        # Filter signals before consensus:
        accepted = self._threshold_mgr.filter_signals(signals)

        # Called by MarketAccuracyTracker (as second callback):
        self._threshold_mgr.on_eval(module_id, score)

    Args:
        store:             Persistence back-end (TelemetryStore).
                           Pass None to run in-memory only (useful for tests
                           and backtest runs where you don't want DB writes).
        default_threshold: Starting threshold for any unseen module.
        alpha:             Step size per evaluation (0.01 = 1 % per bar evaluated).
        min_threshold:     Hard floor — threshold never drops below this.
        max_threshold:     Hard ceiling — threshold never rises above this.
        score_ema_alpha:   EMA factor for the mean_score estimate (0.1 = slow).
    """

    def __init__(
        self,
        store: IThresholdStore | None = None,
        default_threshold: float = 0.60,
        alpha: float = 0.01,
        min_threshold: float = 0.45,
        max_threshold: float = 0.90,
        score_ema_alpha: float = 0.10,
    ) -> None:
        self._store = store
        self._default = default_threshold
        self._alpha = alpha
        self._min = min_threshold
        self._max = max_threshold
        self._ema_alpha = score_ema_alpha
        self._states: dict[str, ModuleThresholdState] = {}

        if store is not None:
            self._load_from_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_threshold(self, module_id: str) -> float:
        """Return the current threshold for a module (default if unseen)."""
        return self._state(module_id).threshold

    def filter_signals(
        self,
        signals: list[Any],   # list[AnalysisSignal]
    ) -> list[Any]:
        """Return only signals whose confidence >= their module's threshold.

        Signals whose module is not yet tracked pass through (no false early
        exclusions during warm-up — the default threshold applies).
        """
        return [
            sig for sig in signals
            if sig.confidence >= self.get_threshold(sig.module_id)
        ]

    def on_eval(self, module_id: str, score: float) -> None:
        """Update threshold and stats for ``module_id`` given an accuracy score.

        Args:
            module_id: The module being evaluated.
            score:     Market-accuracy score in [0, 1] from MarketAccuracyTracker.
        """
        state = self._state(module_id)

        # Adjust threshold proportional to how right/wrong the signal was
        delta = -self._alpha * (score - 0.5) * 2.0
        new_threshold = max(self._min, min(self._max, state.threshold + delta))

        # Update stats
        state.eval_count += 1
        if score >= 0.5:
            state.correct_count += 1
        state.mean_score = (
            self._ema_alpha * score + (1.0 - self._ema_alpha) * state.mean_score
        )
        state.threshold = new_threshold

        if self._store is not None:
            try:
                self._store.upsert_module_threshold(
                    module_id=module_id,
                    threshold=round(new_threshold, 6),
                    eval_count=state.eval_count,
                    correct_count=state.correct_count,
                    mean_score=round(state.mean_score, 6),
                )
            except Exception:
                pass  # Never let storage failures block trading

    def all_states(self) -> list[ModuleThresholdState]:
        """Return a snapshot of all tracked module states."""
        return list(self._states.values())

    def reset_module(self, module_id: str) -> None:
        """Reset a module's threshold to the default value."""
        self._states[module_id] = ModuleThresholdState(
            module_id=module_id,
            threshold=self._default,
        )
        if self._store is not None:
            try:
                self._store.upsert_module_threshold(
                    module_id=module_id,
                    threshold=self._default,
                    eval_count=0,
                    correct_count=0,
                    mean_score=0.5,
                )
            except Exception:
                pass

    # ── Internals ─────────────────────────────────────────────────────────────

    def _state(self, module_id: str) -> ModuleThresholdState:
        if module_id not in self._states:
            self._states[module_id] = ModuleThresholdState(
                module_id=module_id,
                threshold=self._default,
            )
        return self._states[module_id]

    def _load_from_db(self) -> None:
        """Populate in-memory state from persisted DB rows on startup."""
        if self._store is None:
            return
        try:
            rows = self._store.list_module_thresholds()
        except Exception:
            return
        for row in rows:
            module_id = row.get("module_id", "")
            if not module_id:
                continue
            self._states[module_id] = ModuleThresholdState(
                module_id=module_id,
                threshold=float(row.get("threshold", self._default)),
                eval_count=int(row.get("eval_count", 0)),
                correct_count=int(row.get("correct_count", 0)),
                mean_score=float(row.get("mean_score", 0.5)),
            )
