"""Tests for core/contracts/consensus.py."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.consensus import ConsensusResult, VoteRecord
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import ConsensusMode, SignalDirection
from metatrade.core.versioning import ModuleVersion

UTC = timezone.utc
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
V1 = ModuleVersion(1, 0, 0)


def make_signal(direction: SignalDirection, module_id: str = "m1") -> AnalysisSignal:
    return AnalysisSignal(
        direction=direction,
        confidence=0.70,
        reason="test",
        module_id=module_id,
        module_version=V1,
        timestamp_utc=T0,
    )


def make_vote(direction: SignalDirection, weight: float = 1.0, module_id: str = "m1") -> VoteRecord:
    signal = make_signal(direction, module_id)
    return VoteRecord(signal=signal, weight=weight, weighted_score=weight * 0.70)


def make_consensus(**kwargs) -> ConsensusResult:
    defaults = dict(
        symbol="EURUSD",
        final_direction=SignalDirection.BUY,
        buy_pct=0.70,
        sell_pct=0.20,
        hold_pct=0.10,
        aggregate_confidence=0.72,
        consensus_mode=ConsensusMode.SIMPLE_VOTE,
        threshold_used=0.60,
        vote_records=(
            make_vote(SignalDirection.BUY, module_id="m1"),
            make_vote(SignalDirection.BUY, module_id="m2"),
            make_vote(SignalDirection.SELL, module_id="m3"),
        ),
        explanation="7/10 modules voted BUY",
        timestamp_utc=T0,
        is_actionable=True,
    )
    defaults.update(kwargs)
    return ConsensusResult(**defaults)


class TestVoteRecord:
    def test_valid(self) -> None:
        vr = make_vote(SignalDirection.BUY)
        assert vr.weight == 1.0

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match="weight"):
            VoteRecord(
                signal=make_signal(SignalDirection.BUY),
                weight=-0.1,
                weighted_score=0.5,
            )

    def test_negative_weighted_score_raises(self) -> None:
        with pytest.raises(ValueError, match="weighted_score"):
            VoteRecord(
                signal=make_signal(SignalDirection.BUY),
                weight=1.0,
                weighted_score=-0.1,
            )

    def test_frozen(self) -> None:
        vr = make_vote(SignalDirection.BUY)
        with pytest.raises(Exception):
            vr.weight = 2.0  # type: ignore[misc]


class TestConsensusResult:
    def test_valid_construction(self) -> None:
        cr = make_consensus()
        assert cr.is_actionable
        assert cr.final_direction == SignalDirection.BUY

    def test_percentages_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            make_consensus(buy_pct=0.50, sell_pct=0.30, hold_pct=0.30)

    def test_percentages_close_enough_accepted(self) -> None:
        # Floating point tolerance
        make_consensus(buy_pct=0.7, sell_pct=0.2, hold_pct=0.1)

    def test_invalid_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="aggregate_confidence"):
            make_consensus(aggregate_confidence=1.5)

    def test_invalid_threshold_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_used"):
            make_consensus(threshold_used=0.0)

    def test_invalid_threshold_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold_used"):
            make_consensus(threshold_used=1.1)

    def test_empty_symbol_raises(self) -> None:
        with pytest.raises(ValueError, match="symbol"):
            make_consensus(symbol="")

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            make_consensus(timestamp_utc=datetime(2024, 1, 1))

    def test_total_signals(self) -> None:
        cr = make_consensus()
        assert cr.total_signals == 3

    def test_buy_count(self) -> None:
        cr = make_consensus()
        assert cr.buy_count == 2

    def test_sell_count(self) -> None:
        cr = make_consensus()
        assert cr.sell_count == 1

    def test_hold_count(self) -> None:
        cr = make_consensus()
        assert cr.hold_count == 0

    def test_winning_direction_pct_buy(self) -> None:
        cr = make_consensus(final_direction=SignalDirection.BUY, buy_pct=0.70, sell_pct=0.20, hold_pct=0.10)
        assert cr.winning_direction_pct == 0.70

    def test_winning_direction_pct_sell(self) -> None:
        cr = make_consensus(
            final_direction=SignalDirection.SELL,
            buy_pct=0.20,
            sell_pct=0.70,
            hold_pct=0.10,
        )
        assert cr.winning_direction_pct == 0.70

    def test_not_actionable_hold(self) -> None:
        cr = make_consensus(
            final_direction=SignalDirection.HOLD,
            buy_pct=0.40,
            sell_pct=0.30,
            hold_pct=0.30,
            is_actionable=False,
        )
        assert not cr.is_actionable

    def test_frozen(self) -> None:
        cr = make_consensus()
        with pytest.raises(Exception):
            cr.buy_pct = 0.99  # type: ignore[misc]

    def test_repr_contains_key_info(self) -> None:
        cr = make_consensus()
        r = repr(cr)
        assert "EURUSD" in r
        assert "BUY" in r
