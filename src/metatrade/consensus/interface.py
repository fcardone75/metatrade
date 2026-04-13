"""Abstract interface for all consensus engine implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from metatrade.core.contracts.consensus import ConsensusResult
from metatrade.core.contracts.signal import AnalysisSignal


class IConsensusEngine(ABC):
    """Contract that every consensus engine must satisfy.

    A consensus engine receives a list of AnalysisSignals — one per registered
    analysis module — and returns a single ConsensusResult representing the
    aggregated trading decision.

    Implementations:
        SimpleVoteEngine    — equal weight, majority wins.
        WeightedVoteEngine  — fixed per-module weights from config.
        DynamicVoteEngine   — weights adapt based on module accuracy history.
    """

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        signals: list[AnalysisSignal],
        timestamp_utc: datetime,
    ) -> ConsensusResult:
        """Aggregate signals into a consensus decision.

        Args:
            symbol:        The Forex pair being evaluated (e.g. "EURUSD").
            signals:       All signals produced for this evaluation cycle.
                           Empty list → HOLD with is_actionable=False.
            timestamp_utc: UTC timestamp for the resulting ConsensusResult.

        Returns:
            ConsensusResult with full vote audit trail.
        """
