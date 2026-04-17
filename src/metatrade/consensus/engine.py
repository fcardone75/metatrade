"""ConsensusEngine — factory that selects the correct voting implementation.

Usage:
    config = ConsensusConfig()
    engine = ConsensusEngine(config)
    result = engine.evaluate(symbol="EURUSD", signals=signals, timestamp_utc=now)

    # For dynamic mode only — feed outcomes back after each trade:
    engine.update_performance("ema_crossover_h1", correct=True)
"""

from __future__ import annotations

from datetime import datetime

from metatrade.consensus.config import ConsensusConfig
from metatrade.consensus.engines.dynamic_vote import DynamicVoteEngine
from metatrade.consensus.engines.simple_vote import SimpleVoteEngine
from metatrade.consensus.engines.weighted_vote import WeightedVoteEngine
from metatrade.consensus.errors import ConsensusError
from metatrade.consensus.interface import IConsensusEngine
from metatrade.core.contracts.consensus import ConsensusResult
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import ConsensusMode
from metatrade.core.log import get_logger

log = get_logger(__name__)


class ConsensusEngine:
    """Top-level consensus engine — delegates to the configured sub-engine.

    This class is the main entry point for the rest of the system.
    It wraps one of the three engine implementations and exposes a stable
    API regardless of which voting strategy is active.

    Args:
        config: ConsensusConfig (loaded from env or passed explicitly).
    """

    def __init__(self, config: ConsensusConfig | None = None) -> None:
        self._config = config or ConsensusConfig()
        self._engine: IConsensusEngine = self._create_engine(self._config)
        log.info(
            "consensus_engine_initialised",
            mode=self._config.mode.value,
            threshold=self._config.threshold,
            min_signals=self._config.min_signals,
        )

    def _create_engine(self, cfg: ConsensusConfig) -> IConsensusEngine:
        if cfg.mode == ConsensusMode.SIMPLE_VOTE:
            return SimpleVoteEngine(
                threshold=cfg.threshold,
                min_signals=cfg.min_signals,
            )
        if cfg.mode == ConsensusMode.WEIGHTED_VOTE:
            return WeightedVoteEngine(
                module_weights=cfg.module_weights,
                default_weight=cfg.default_weight,
                threshold=cfg.threshold,
                min_signals=cfg.min_signals,
            )
        if cfg.mode == ConsensusMode.DYNAMIC_VOTE:
            return DynamicVoteEngine(
                module_weights=cfg.module_weights,
                default_weight=cfg.default_weight,
                threshold=cfg.threshold,
                min_signals=cfg.min_signals,
                alpha=cfg.performance_alpha,
                min_weight=cfg.min_dynamic_weight,
            )
        raise ConsensusError(
            message=f"Unknown ConsensusMode: {cfg.mode}",
            code="CONSENSUS_UNKNOWN_MODE",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        symbol: str,
        signals: list[AnalysisSignal],
        timestamp_utc: datetime,
    ) -> ConsensusResult:
        """Aggregate signals into a consensus decision.

        Logs the result at INFO level (direction, confidence, actionable).
        """
        result = self._engine.evaluate(symbol, signals, timestamp_utc)
        log.info(
            "consensus_evaluated",
            symbol=symbol,
            direction=result.final_direction.value,
            confidence=round(result.aggregate_confidence, 4),
            actionable=result.is_actionable,
            n_signals=len(signals),
            buy_pct=round(result.buy_pct, 3),
            sell_pct=round(result.sell_pct, 3),
        )
        return result

    def update_performance(self, module_id: str, correct: bool) -> None:
        """Feed a binary trade outcome back (only effective for DYNAMIC_VOTE).

        Safe to call for all modes — a no-op for SimpleVote and WeightedVote.
        """
        if isinstance(self._engine, DynamicVoteEngine):
            self._engine.update_performance(module_id, correct)

    def update_performance_scored(self, module_id: str, score: float) -> None:
        """Feed a continuous accuracy score [0, 1] back (only for DYNAMIC_VOTE).

        Preferred over update_performance() when the magnitude of the error
        matters (e.g. market moved 2% against the signal vs 0.01%).
        Safe no-op for SimpleVote and WeightedVote modes.
        """
        if isinstance(self._engine, DynamicVoteEngine):
            self._engine.update_performance_scored(module_id, score)

    def get_module_weight(self, module_id: str) -> float | None:
        """Return the current effective weight for a module (DYNAMIC_VOTE only).

        Returns None for SimpleVote mode.
        """
        if isinstance(self._engine, (DynamicVoteEngine, WeightedVoteEngine)):
            return self._engine.get_weight(module_id)
        return None

    @property
    def mode(self) -> ConsensusMode:
        return self._config.mode
