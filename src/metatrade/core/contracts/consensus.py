"""ConsensusResult — output of the consensus engine.

The consensus engine takes a list of AnalysisSignals and produces a single
ConsensusResult. This is what the risk manager receives as input.

VoteRecord tracks how each signal contributed to the final result,
enabling full auditability of every trading decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import ConsensusMode, SignalDirection


@dataclass(frozen=True)
class VoteRecord:
    """How one AnalysisSignal was weighted in the consensus calculation."""

    signal: AnalysisSignal
    weight: float       # Effective weight applied to this signal [0.0, 1.0]
    weighted_score: float  # Contribution to direction total (weight * confidence)

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"VoteRecord.weight cannot be negative, got {self.weight}")
        if self.weighted_score < 0:
            raise ValueError(
                f"VoteRecord.weighted_score cannot be negative, got {self.weighted_score}"
            )


@dataclass(frozen=True)
class ConsensusResult:
    """Aggregated decision from all analysis modules.

    buy_pct / sell_pct / hold_pct   Weighted vote shares. Sum == 1.0.
    aggregate_confidence            Weighted average confidence of signals
                                    voting for final_direction.
    is_actionable                   True when a direction met the threshold.
                                    False = the risk layer receives HOLD.
    vote_records                    Full audit trail of every vote cast.
    explanation                     Human-readable summary for logs/UI.
    """

    symbol: str
    final_direction: SignalDirection
    buy_pct: float
    sell_pct: float
    hold_pct: float
    aggregate_confidence: float
    consensus_mode: ConsensusMode
    threshold_used: float
    vote_records: tuple[VoteRecord, ...]
    explanation: str
    timestamp_utc: datetime
    is_actionable: bool

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("ConsensusResult.timestamp_utc must be timezone-aware (UTC)")
        total = self.buy_pct + self.sell_pct + self.hold_pct
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"ConsensusResult percentages must sum to 1.0, got {total:.6f}"
            )
        if not 0.0 <= self.aggregate_confidence <= 1.0:
            raise ValueError(
                f"ConsensusResult.aggregate_confidence must be in [0.0, 1.0], "
                f"got {self.aggregate_confidence}"
            )
        if not 0.0 < self.threshold_used <= 1.0:
            raise ValueError(
                f"ConsensusResult.threshold_used must be in (0.0, 1.0], "
                f"got {self.threshold_used}"
            )
        if not self.symbol:
            raise ValueError("ConsensusResult.symbol cannot be empty")

    # ── Derived counts ────────────────────────────────────────────────────────

    @property
    def total_signals(self) -> int:
        return len(self.vote_records)

    @property
    def buy_count(self) -> int:
        return sum(
            1 for vr in self.vote_records if vr.signal.direction == SignalDirection.BUY
        )

    @property
    def sell_count(self) -> int:
        return sum(
            1 for vr in self.vote_records if vr.signal.direction == SignalDirection.SELL
        )

    @property
    def hold_count(self) -> int:
        return sum(
            1 for vr in self.vote_records if vr.signal.direction == SignalDirection.HOLD
        )

    @property
    def winning_direction_pct(self) -> float:
        """Vote share of the final_direction."""
        mapping = {
            SignalDirection.BUY: self.buy_pct,
            SignalDirection.SELL: self.sell_pct,
            SignalDirection.HOLD: self.hold_pct,
        }
        return mapping[self.final_direction]

    def __repr__(self) -> str:
        return (
            f"ConsensusResult("
            f"symbol={self.symbol!r}, "
            f"direction={self.final_direction.value}, "
            f"buy={self.buy_pct:.0%}, sell={self.sell_pct:.0%}, hold={self.hold_pct:.0%}, "
            f"confidence={self.aggregate_confidence:.2f}, "
            f"actionable={self.is_actionable})"
        )
