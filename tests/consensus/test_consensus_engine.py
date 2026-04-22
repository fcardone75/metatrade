"""Tests for the top-level ConsensusEngine factory and config."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from metatrade.consensus.config import ConsensusConfig
from metatrade.consensus.engine import ConsensusEngine
from metatrade.consensus.engines.dynamic_vote import DynamicVoteEngine
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import ConsensusMode, SignalDirection
from metatrade.core.versioning import ModuleVersion

UTC = UTC
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
V1 = ModuleVersion(1, 0, 0)


def make_signal(
    direction: SignalDirection,
    module_id: str = "m1",
    confidence: float = 0.80,
) -> AnalysisSignal:
    return AnalysisSignal(
        direction=direction,
        confidence=confidence,
        reason="test",
        module_id=module_id,
        module_version=V1,
        timestamp_utc=T0,
    )


class TestConsensusConfig:
    def test_default_mode_is_simple_vote(self) -> None:
        cfg = ConsensusConfig()
        assert cfg.mode == ConsensusMode.SIMPLE_VOTE

    def test_default_threshold(self) -> None:
        cfg = ConsensusConfig()
        assert cfg.threshold == 0.65

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(Exception):
            ConsensusConfig(threshold=0.0)

    def test_invalid_threshold_above_one_raises(self) -> None:
        with pytest.raises(Exception):
            ConsensusConfig(threshold=1.1)

    def test_invalid_default_weight_raises(self) -> None:
        with pytest.raises(Exception):
            ConsensusConfig(default_weight=0.0)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(Exception):
            ConsensusConfig(performance_alpha=0.0)

    def test_invalid_min_signals_raises(self) -> None:
        with pytest.raises(Exception):
            ConsensusConfig(min_signals=0)


class TestConsensusEngineFactory:
    def test_creates_simple_mode(self) -> None:
        cfg = ConsensusConfig(mode=ConsensusMode.SIMPLE_VOTE)
        engine = ConsensusEngine(cfg)
        assert engine.mode == ConsensusMode.SIMPLE_VOTE

    def test_creates_weighted_mode(self) -> None:
        cfg = ConsensusConfig(mode=ConsensusMode.WEIGHTED_VOTE)
        engine = ConsensusEngine(cfg)
        assert engine.mode == ConsensusMode.WEIGHTED_VOTE

    def test_creates_dynamic_mode(self) -> None:
        cfg = ConsensusConfig(mode=ConsensusMode.DYNAMIC_VOTE)
        engine = ConsensusEngine(cfg)
        assert engine.mode == ConsensusMode.DYNAMIC_VOTE

    def test_evaluate_returns_result(self) -> None:
        # Use explicit min_signals=1 so this unit test is not coupled to the
        # production default (which is 3 to reduce false positives).
        engine = ConsensusEngine(ConsensusConfig(min_signals=1))
        signals = [make_signal(SignalDirection.BUY)]
        result = engine.evaluate("EURUSD", signals, T0)
        assert result.symbol == "EURUSD"
        assert result.final_direction == SignalDirection.BUY

    def test_update_performance_noop_for_simple(self) -> None:
        engine = ConsensusEngine(ConsensusConfig(mode=ConsensusMode.SIMPLE_VOTE))
        engine.update_performance("m1", correct=True)  # must not raise

    def test_update_performance_noop_for_weighted(self) -> None:
        engine = ConsensusEngine(ConsensusConfig(mode=ConsensusMode.WEIGHTED_VOTE))
        engine.update_performance("m1", correct=True)  # must not raise

    def test_update_performance_works_for_dynamic(self) -> None:
        cfg = ConsensusConfig(mode=ConsensusMode.DYNAMIC_VOTE)
        engine = ConsensusEngine(cfg)
        # Should not raise and should affect internal state
        engine.update_performance("m1", correct=True)
        assert isinstance(engine._engine, DynamicVoteEngine)
        assert engine._engine.get_accuracy("m1") != 0.5

    def test_default_config_used_when_none(self) -> None:
        engine = ConsensusEngine()
        assert engine.mode == ConsensusMode.SIMPLE_VOTE

    def test_module_weights_passed_to_weighted_engine(self) -> None:
        cfg = ConsensusConfig(
            mode=ConsensusMode.WEIGHTED_VOTE,
            module_weights={"m1": 3.0},
        )
        engine = ConsensusEngine(cfg)
        signals = [
            make_signal(SignalDirection.BUY, module_id="m1", confidence=1.0),
            make_signal(SignalDirection.SELL, module_id="m2", confidence=1.0),
            make_signal(SignalDirection.SELL, module_id="m3", confidence=1.0),
        ]
        # m1 weight=3, m2/m3 weight=1 → BUY score=3, SELL score=2 → BUY wins
        result = engine.evaluate("EURUSD", signals, T0)
        assert result.final_direction == SignalDirection.BUY
