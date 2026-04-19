"""Live prediction accuracy tracker.

Records every BUY/SELL prediction made by the active model and evaluates
correctness once ``forward_bars`` bars have elapsed.  Accuracy is tracked in
SQLite and exposed as a reliability-gated metric.

Correctness criterion (directional):
    BUY  → correct if future_close > entry_close
    SELL → correct if future_close < entry_close

This is simpler and more interpretable than an ATR-threshold check at
evaluation time (the ATR at prediction time is not stored).  The resulting
metric — "directional accuracy" — is the natural analogue of the 3-class
holdout accuracy produced by walk-forward training.

Only BUY and SELL predictions are tracked; HOLD predictions carry no
directional claim and are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from metatrade.core.contracts.market import Bar
from metatrade.core.log import get_logger
from metatrade.observability.store import TelemetryStore

log = get_logger(__name__)


@dataclass
class LiveAccuracyStats:
    """Snapshot of evaluated accuracy for one model version."""

    model_version: str
    n_evaluated: int
    n_correct: int
    n_pending: int
    accuracy: float | None        # None if n_evaluated < min_samples
    is_reliable: bool             # True once n_evaluated >= min_samples
    min_samples: int

    @property
    def accuracy_pct(self) -> str:
        if self.accuracy is None:
            return "n/a"
        return f"{self.accuracy:.1%}"


class LiveAccuracyTracker:
    """Track live prediction accuracy for a single symbol/timeframe.

    Args:
        symbol:          Instrument symbol (e.g. "EURUSD").
        timeframe:       Timeframe label (e.g. "M1").
        forward_bars:    Same value used during training — how many bars
                         ahead to check if the prediction was correct.
        tf_minutes:      Duration of one bar in minutes (e.g. 1 for M1).
        telemetry:       TelemetryStore for persistence.
        min_samples:     Minimum evaluated predictions before accuracy is
                         considered reliable for switch decisions.
    """

    def __init__(
        self,
        *,
        symbol: str,
        timeframe: str,
        forward_bars: int,
        tf_minutes: int,
        telemetry: TelemetryStore,
        min_samples: int = 200,
    ) -> None:
        self._symbol = symbol
        self._timeframe = timeframe
        self._forward_bars = forward_bars
        self._tf_minutes = tf_minutes
        self._telemetry = telemetry
        self._min_samples = min_samples
        self._eval_window = timedelta(minutes=forward_bars * tf_minutes)

    # ── Public API ────────────────────────────────────────────────────────────

    def record_prediction(
        self,
        bar: Bar,
        direction: int,
        confidence: float,
        model_version: str,
    ) -> None:
        """Persist a BUY (1) or SELL (-1) prediction for future evaluation.

        HOLD predictions (direction == 0) are silently ignored.
        """
        if direction == 0:
            return
        evaluate_after = bar.timestamp_utc + self._eval_window
        self._telemetry.record_ml_prediction(
            model_version=model_version,
            symbol=self._symbol,
            timeframe=self._timeframe,
            bar_ts_utc=bar.timestamp_utc,
            direction=direction,
            confidence=confidence,
            entry_close=float(bar.close),
            evaluate_after_ts=evaluate_after,
        )

    def evaluate_pending(self, current_bar: Bar) -> int:
        """Evaluate all pending predictions whose window has elapsed.

        Should be called once per bar in the runner's main loop.

        Returns:
            Number of predictions evaluated in this call.
        """
        now = current_bar.timestamp_utc
        pending = self._telemetry.get_pending_predictions(
            symbol=self._symbol,
            timeframe=self._timeframe,
            before_ts=now,
        )
        if not pending:
            return 0

        future_close = float(current_bar.close)
        n_evaluated = 0
        for row in pending:
            direction = int(row["direction"])
            entry_close = float(row["entry_close"])
            correct = (
                (direction == 1 and future_close > entry_close)
                or (direction == -1 and future_close < entry_close)
            )
            self._telemetry.evaluate_ml_prediction(
                prediction_id=int(row["id"]),
                future_close=future_close,
                correct=correct,
                evaluated_at_utc=now,
            )
            n_evaluated += 1

        if n_evaluated > 0:
            log.debug(
                "live_tracker_evaluated",
                symbol=self._symbol,
                timeframe=self._timeframe,
                n=n_evaluated,
            )
        return n_evaluated

    def live_accuracy_stats(self, model_version: str) -> LiveAccuracyStats:
        """Return current accuracy statistics for the given model version."""
        raw = self._telemetry.get_live_accuracy_stats(
            model_version=model_version,
            min_samples=self._min_samples,
        )
        return LiveAccuracyStats(
            model_version=raw["model_version"],
            n_evaluated=raw["n_evaluated"],
            n_correct=raw["n_correct"],
            n_pending=raw["n_pending"],
            accuracy=raw["accuracy"],
            is_reliable=raw["is_reliable"],
            min_samples=self._min_samples,
        )
