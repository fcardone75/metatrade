"""Market accuracy tracker — automatic signal evaluation against price movement.

After a module emits a BUY or SELL signal, this tracker waits ``forward_bars``
bars and then measures whether the market actually moved in the predicted
direction.  The result is a continuous score in [0, 1]:

    score = (1 + tanh(direction_return × 200)) / 2

Where ``direction_return`` is the actual percent return of the asset,
sign-flipped for SELL signals so that a correct prediction always yields a
positive return.

Score interpretation:
    ~1.0  →  market moved strongly in the predicted direction
    ~0.5  →  market was flat or ambiguous
    ~0.0  →  market moved strongly against the prediction

This score is fed to ``ConsensusEngine.update_performance_scored()`` which
updates the weight of the module in the ``DynamicVoteEngine`` via EMA:

    accuracy_ema = alpha × score + (1 - alpha) × accuracy_ema
    weight       = max(min_weight, base_weight × 2 × accuracy_ema)

Proportionality guarantee:
    A small error (market barely moved the wrong way, score≈0.45) shrinks the
    weight only slightly.  A large error (market moved hard against the signal,
    score≈0.02) shrinks it significantly.  This is the key difference from the
    binary ``update_performance(correct: bool)`` API.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection

# ── Contracts ──────────────────────────────────────────────────────────────────

@dataclass
class PendingEvaluation:
    """A signal queued for market-accuracy evaluation."""

    module_id: str
    direction: SignalDirection      # BUY or SELL
    entry_close: Decimal            # close price when signal was recorded
    bars_remaining: int             # bars left before evaluation fires
    recorded_at: datetime


@dataclass(frozen=True)
class EvalResult:
    """Outcome of a single signal evaluation.

    Attributes:
        module_id:       The module whose signal was evaluated.
        direction:       The signal direction (BUY / SELL).
        entry_close:     Price when the signal was recorded.
        exit_close:      Price ``forward_bars`` bars later.
        market_return:   Raw percent return (positive = price rose).
        direction_return: ``market_return`` sign-adjusted for direction.
                         Positive means the prediction was correct.
        score:           Continuous accuracy score in [0, 1].
        recorded_at:     When the signal was originally recorded.
        evaluated_at:    When this evaluation was produced.
    """

    module_id: str
    direction: SignalDirection
    entry_close: Decimal
    exit_close: Decimal
    market_return: float        # raw: positive = price rose
    direction_return: float     # adjusted for signal direction
    score: float                # [0, 1]
    recorded_at: datetime
    evaluated_at: datetime


# ── Score computation ──────────────────────────────────────────────────────────

def _compute_score(
    direction: SignalDirection,
    entry_close: Decimal,
    exit_close: Decimal,
) -> tuple[float, float, float]:
    """Compute (market_return, direction_return, score) for a signal evaluation.

    The score uses a tanh to map any size of price move to [0, 1]:
    - A 1% move in the correct direction  → score ≈ 0.98
    - A 0.1% move in the correct direction → score ≈ 0.73
    - Flat market (0% move)                → score  = 0.50
    - A 0.1% move in the wrong direction   → score ≈ 0.27
    - A 1% move in the wrong direction     → score ≈ 0.02

    The multiplier 200 calibrates tanh so that ±1% moves saturate the scale.
    Adjust if trading higher-volatility instruments.
    """
    if entry_close == 0:
        return 0.0, 0.0, 0.5

    market_return = float(exit_close - entry_close) / float(entry_close)

    if direction == SignalDirection.SELL:
        direction_return = -market_return
    else:
        direction_return = market_return

    score = (1.0 + math.tanh(direction_return * 200)) / 2.0
    score = max(0.0, min(1.0, score))
    return market_return, direction_return, score


# ── Tracker ────────────────────────────────────────────────────────────────────

class MarketAccuracyTracker:
    """Automatically evaluates module signals against subsequent market movement.

    Usage pattern (called by BaseRunner on every bar):

        # 1. Evaluate signals recorded forward_bars ago:
        results = tracker.on_bar(current_close, timestamp_utc)

        # 2. Record new signals for future evaluation:
        tracker.record_signals(new_signals, current_close, timestamp_utc)

    Args:
        update_callback:  Called with (module_id, score) for each evaluation.
                          Typically ``ConsensusEngine.update_performance_scored``.
        forward_bars:     How many bars ahead to evaluate a signal.
                          Must match the labelling horizon used for ML training.
        skip_hold:        If True (default), HOLD signals are not recorded.
                          HOLD does not predict a direction, so evaluating it
                          makes no sense.
    """

    def __init__(
        self,
        update_callback: Callable[[str, float], None],
        forward_bars: int = 5,
        skip_hold: bool = True,
    ) -> None:
        self._callback = update_callback
        self._forward_bars = forward_bars
        self._skip_hold = skip_hold
        self._pending: list[PendingEvaluation] = []

    # ── Write path ────────────────────────────────────────────────────────────

    def record_signals(
        self,
        signals: list[AnalysisSignal],
        current_close: Decimal,
        timestamp_utc: datetime | None = None,
    ) -> None:
        """Queue directional signals for evaluation after ``forward_bars`` bars.

        Args:
            signals:       Signals produced by analyse() on the current bar.
            current_close: Close price of the current bar.
            timestamp_utc: Timestamp of the current bar.
        """
        ts = timestamp_utc or datetime.now(UTC)
        for sig in signals:
            if self._skip_hold and sig.direction == SignalDirection.HOLD:
                continue
            self._pending.append(PendingEvaluation(
                module_id=sig.module_id,
                direction=sig.direction,
                entry_close=current_close,
                bars_remaining=self._forward_bars,
                recorded_at=ts,
            ))

    # ── Read / advance path ────────────────────────────────────────────────────

    def on_bar(
        self,
        current_close: Decimal,
        timestamp_utc: datetime | None = None,
    ) -> list[EvalResult]:
        """Advance one bar and evaluate any matured pending signals.

        Returns the list of evaluations produced this bar (may be empty).
        Each evaluation also triggers ``update_callback`` with the module's
        accuracy score.

        Args:
            current_close: Close price of the current (newly closed) bar.
            timestamp_utc: Timestamp of the current bar.
        """
        ts = timestamp_utc or datetime.now(UTC)
        results: list[EvalResult] = []
        still_pending: list[PendingEvaluation] = []

        for pe in self._pending:
            pe.bars_remaining -= 1
            if pe.bars_remaining <= 0:
                market_ret, dir_ret, score = _compute_score(
                    pe.direction, pe.entry_close, current_close
                )
                result = EvalResult(
                    module_id=pe.module_id,
                    direction=pe.direction,
                    entry_close=pe.entry_close,
                    exit_close=current_close,
                    market_return=market_ret,
                    direction_return=dir_ret,
                    score=score,
                    recorded_at=pe.recorded_at,
                    evaluated_at=ts,
                )
                results.append(result)
                self._callback(pe.module_id, score)
            else:
                still_pending.append(pe)

        self._pending = still_pending
        return results

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def pending_count(self) -> int:
        """Number of signals still awaiting evaluation."""
        return len(self._pending)

    def pending_for_module(self, module_id: str) -> int:
        """Number of pending evaluations for a specific module."""
        return sum(1 for p in self._pending if p.module_id == module_id)

    def reset(self) -> None:
        """Discard all pending evaluations (e.g. on session restart)."""
        self._pending.clear()
