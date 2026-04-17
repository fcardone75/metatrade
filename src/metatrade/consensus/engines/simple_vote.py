"""Simple majority voting engine.

Every signal receives equal weight (1.0).
The direction with the highest vote count wins.
Confidence is incorporated only when computing aggregate_confidence of the winner.
"""

from __future__ import annotations

from datetime import datetime

from metatrade.consensus.engines._base import _build_result, _hold_result
from metatrade.consensus.interface import IConsensusEngine
from metatrade.core.contracts.consensus import ConsensusResult, VoteRecord
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import ConsensusMode


class SimpleVoteEngine(IConsensusEngine):
    """Equal-weight majority vote.

    Args:
        threshold:   Minimum vote share in (0.0, 1.0] for a direction to be
                     actionable (e.g. 0.60 means 60% of votes required).
        min_signals: Minimum number of signals required for an actionable result.
    """

    def __init__(self, threshold: float = 0.60, min_signals: int = 1) -> None:
        self._threshold = threshold
        self._min_signals = min_signals

    def evaluate(
        self,
        symbol: str,
        signals: list[AnalysisSignal],
        timestamp_utc: datetime,
    ) -> ConsensusResult:
        if not signals:
            return _hold_result(
                symbol, signals, ConsensusMode.SIMPLE_VOTE,
                self._threshold, timestamp_utc,
                "No signals provided",
            )
        if len(signals) < self._min_signals:
            return _hold_result(
                symbol, signals, ConsensusMode.SIMPLE_VOTE,
                self._threshold, timestamp_utc,
                f"Insufficient signals: {len(signals)} < {self._min_signals}",
            )

        vote_records = tuple(
            VoteRecord(
                signal=s,
                weight=1.0,
                weighted_score=s.confidence,  # 1.0 * confidence
            )
            for s in signals
        )
        return _build_result(
            symbol, vote_records,
            ConsensusMode.SIMPLE_VOTE, self._threshold, timestamp_utc,
        )
