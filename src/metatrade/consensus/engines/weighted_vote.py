"""Weighted voting engine.

Each module has a fixed weight assigned at construction time.
Modules not listed in the weight map receive the default_weight.
The weighted score for each signal = weight * confidence.
"""

from __future__ import annotations

from datetime import datetime

from metatrade.consensus.engines._base import _build_result, _hold_result
from metatrade.consensus.interface import IConsensusEngine
from metatrade.core.contracts.consensus import ConsensusResult, VoteRecord
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import ConsensusMode


class WeightedVoteEngine(IConsensusEngine):
    """Fixed per-module weight voting.

    Args:
        module_weights:  Map of {module_id: weight}. Values must be > 0.
        default_weight:  Weight for modules not in module_weights.
        threshold:       Minimum weighted vote share to be actionable.
        min_signals:     Minimum signals required for an actionable result.
    """

    def __init__(
        self,
        module_weights: dict[str, float] | None = None,
        default_weight: float = 1.0,
        threshold: float = 0.60,
        min_signals: int = 1,
    ) -> None:
        self._weights: dict[str, float] = module_weights or {}
        self._default = default_weight
        self._threshold = threshold
        self._min_signals = min_signals

    def get_weight(self, module_id: str) -> float:
        return self._weights.get(module_id, self._default)

    def evaluate(
        self,
        symbol: str,
        signals: list[AnalysisSignal],
        timestamp_utc: datetime,
    ) -> ConsensusResult:
        if not signals:
            return _hold_result(
                symbol, signals, ConsensusMode.WEIGHTED_VOTE,
                self._threshold, timestamp_utc,
                "No signals provided",
            )
        if len(signals) < self._min_signals:
            return _hold_result(
                symbol, signals, ConsensusMode.WEIGHTED_VOTE,
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
            ConsensusMode.WEIGHTED_VOTE, self._threshold, timestamp_utc,
        )
