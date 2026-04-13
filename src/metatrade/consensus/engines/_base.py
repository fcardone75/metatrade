"""Shared helpers for consensus engine implementations."""

from __future__ import annotations

from datetime import datetime

from metatrade.core.contracts.consensus import ConsensusResult, VoteRecord
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import ConsensusMode, SignalDirection


def _hold_result(
    symbol: str,
    signals: list[AnalysisSignal],
    mode: ConsensusMode,
    threshold: float,
    timestamp_utc: datetime,
    reason: str,
) -> ConsensusResult:
    """Return a HOLD result (no actionable decision)."""
    vote_records = tuple(
        VoteRecord(signal=s, weight=0.0, weighted_score=0.0) for s in signals
    )
    n = len(signals)
    return ConsensusResult(
        symbol=symbol,
        final_direction=SignalDirection.HOLD,
        buy_pct=0.0,
        sell_pct=0.0,
        hold_pct=1.0,
        aggregate_confidence=0.0,
        consensus_mode=mode,
        threshold_used=threshold,
        vote_records=vote_records,
        explanation=reason,
        timestamp_utc=timestamp_utc,
        is_actionable=False,
    )


def _build_result(
    symbol: str,
    vote_records: tuple[VoteRecord, ...],
    mode: ConsensusMode,
    threshold: float,
    timestamp_utc: datetime,
) -> ConsensusResult:
    """Build a ConsensusResult from completed vote records.

    Computes direction percentages from weighted scores, determines the winner,
    and decides actionability based on threshold.
    """
    buy_score = sum(
        vr.weighted_score
        for vr in vote_records
        if vr.signal.direction == SignalDirection.BUY
    )
    sell_score = sum(
        vr.weighted_score
        for vr in vote_records
        if vr.signal.direction == SignalDirection.SELL
    )
    hold_score = sum(
        vr.weighted_score
        for vr in vote_records
        if vr.signal.direction == SignalDirection.HOLD
    )
    total_score = buy_score + sell_score + hold_score

    if total_score == 0.0:
        buy_pct, sell_pct, hold_pct = 0.0, 0.0, 1.0
    else:
        buy_pct = buy_score / total_score
        sell_pct = sell_score / total_score
        hold_pct = hold_score / total_score

    # Normalise to avoid floating-point drift
    hold_pct = max(0.0, 1.0 - buy_pct - sell_pct)

    # Winner
    scores = {
        SignalDirection.BUY: buy_pct,
        SignalDirection.SELL: sell_pct,
        SignalDirection.HOLD: hold_pct,
    }
    final_direction = max(scores, key=lambda d: scores[d])

    # Aggregate confidence = weighted avg confidence of signals backing the winner
    winning_votes = [
        vr for vr in vote_records if vr.signal.direction == final_direction
    ]
    if winning_votes:
        total_w = sum(vr.weight for vr in winning_votes)
        if total_w > 0:
            agg_conf = sum(
                vr.weight * vr.signal.confidence for vr in winning_votes
            ) / total_w
        else:
            agg_conf = sum(vr.signal.confidence for vr in winning_votes) / len(
                winning_votes
            )
    else:
        agg_conf = 0.0

    winning_pct = scores[final_direction]
    is_actionable = (
        final_direction != SignalDirection.HOLD and winning_pct >= threshold
    )

    explanation = (
        f"{final_direction.value}: {winning_pct:.0%} vote share "
        f"(threshold={threshold:.0%}, actionable={is_actionable})"
    )

    return ConsensusResult(
        symbol=symbol,
        final_direction=final_direction,
        buy_pct=buy_pct,
        sell_pct=sell_pct,
        hold_pct=hold_pct,
        aggregate_confidence=agg_conf,
        consensus_mode=mode,
        threshold_used=threshold,
        vote_records=vote_records,
        explanation=explanation,
        timestamp_utc=timestamp_utc,
        is_actionable=is_actionable,
    )
