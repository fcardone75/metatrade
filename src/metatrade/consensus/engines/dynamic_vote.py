"""Dynamic voting engine — weights adapt based on recent module accuracy.

Each module starts with a configurable base weight. After each trade outcome
is known, call update_performance() to feed the result back. The engine
uses an exponential moving average (EMA) to maintain a running accuracy
estimate per module and scales weights accordingly.

Weight update rule:
    accuracy_ema = alpha * outcome + (1 - alpha) * accuracy_ema
    weight       = max(min_weight, base_weight * (2 * accuracy_ema))

    With alpha=0.10 and accuracy_ema initialised at 0.5 (neutral):
    - A module that is always correct converges to weight = base_weight * 2.
    - A module that is always wrong converges to weight → min_weight.
    - An unknown module starts at weight = base_weight.

This enables automatic up-weighting of high-accuracy modules and
down-weighting of underperforming ones without manual reconfiguration.
"""

from __future__ import annotations

from datetime import datetime

from metatrade.core.contracts.consensus import ConsensusResult, VoteRecord
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import ConsensusMode
from metatrade.consensus.interface import IConsensusEngine
from metatrade.consensus.engines._base import _build_result, _hold_result


class DynamicVoteEngine(IConsensusEngine):
    """Self-adjusting weighted vote engine.

    Args:
        module_weights:   Initial base weights per module_id.
        default_weight:   Base weight for modules not in module_weights.
        threshold:        Minimum weighted vote share to be actionable.
        min_signals:      Minimum signals required for an actionable result.
        alpha:            EMA smoothing factor (0.0, 1.0].
                          Higher = faster adaptation to recent outcomes.
        min_weight:       Weight floor (prevents zero-weighting a module).
    """

    def __init__(
        self,
        module_weights: dict[str, float] | None = None,
        default_weight: float = 1.0,
        threshold: float = 0.60,
        min_signals: int = 1,
        alpha: float = 0.10,
        min_weight: float = 0.10,
    ) -> None:
        self._base_weights: dict[str, float] = module_weights or {}
        self._default = default_weight
        self._threshold = threshold
        self._min_signals = min_signals
        self._alpha = alpha
        self._min_weight = min_weight
        # Running EMA of accuracy per module_id (initialised at 0.5 = neutral).
        self._accuracy: dict[str, float] = {}

    # ── Performance feedback ──────────────────────────────────────────────────

    def update_performance(self, module_id: str, correct: bool) -> None:
        """Feed a binary trade outcome back to adjust module weight."""
        self.update_performance_scored(module_id, 1.0 if correct else 0.0)

    def update_performance_scored(self, module_id: str, score: float) -> None:
        """Feed a continuous accuracy score in [0, 1] to adjust weight.

        Score semantics:
            1.0  → perfectly correct  (market moved strongly in predicted direction)
            0.5  → neutral            (market flat or ambiguous)
            0.0  → completely wrong   (market moved strongly against prediction)

        Weight update:
            accuracy_ema = alpha * score + (1 - alpha) * accuracy_ema
            weight       = max(min_weight, base_weight * 2 * accuracy_ema)

        Proportional: a small error reduces weight less than a large error.
        A module always correct converges to weight = 2 × base.
        A module always wrong converges toward min_weight.

        Args:
            module_id: Module identifier.
            score:     Accuracy score clamped to [0.0, 1.0].
        """
        score = max(0.0, min(1.0, score))
        current = self._accuracy.get(module_id, 0.5)
        self._accuracy[module_id] = (
            self._alpha * score + (1.0 - self._alpha) * current
        )

    def get_accuracy(self, module_id: str) -> float:
        """Return the current EMA accuracy estimate for module_id (default 0.5)."""
        return self._accuracy.get(module_id, 0.5)

    def get_weight(self, module_id: str) -> float:
        """Current effective weight for module_id."""
        base = self._base_weights.get(module_id, self._default)
        accuracy = self._accuracy.get(module_id, 0.5)
        raw = base * (2.0 * accuracy)
        return max(self._min_weight, raw)

    # ── IConsensusEngine ──────────────────────────────────────────────────────

    def evaluate(
        self,
        symbol: str,
        signals: list[AnalysisSignal],
        timestamp_utc: datetime,
    ) -> ConsensusResult:
        if not signals:
            return _hold_result(
                symbol, signals, ConsensusMode.DYNAMIC_VOTE,
                self._threshold, timestamp_utc,
                "No signals provided",
            )
        if len(signals) < self._min_signals:
            return _hold_result(
                symbol, signals, ConsensusMode.DYNAMIC_VOTE,
                self._threshold, timestamp_utc,
                f"Insufficient signals: {len(signals)} < {self._min_signals}",
            )

        vote_records = tuple(
            VoteRecord(
                signal=s,
                weight=self.get_weight(s.module_id),
                weighted_score=self.get_weight(s.module_id) * s.confidence,
            )
            for s in signals
        )
        return _build_result(
            symbol, vote_records,
            ConsensusMode.DYNAMIC_VOTE, self._threshold, timestamp_utc,
        )
