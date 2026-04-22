"""Consensus engine configuration."""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings

from metatrade.core.enums import ConsensusMode


class ConsensusConfig(BaseSettings):
    """Configuration for the consensus engine.

    All fields are overridable via environment variables prefixed CONSENSUS_.

    Example .env:
        CONSENSUS_MODE=WEIGHTED_VOTE
        CONSENSUS_THRESHOLD=0.65
        CONSENSUS_MIN_SIGNALS=3
    """

    model_config = {"env_prefix": "CONSENSUS_", "case_sensitive": False}

    # Voting strategy
    mode: ConsensusMode = ConsensusMode.SIMPLE_VOTE

    # Minimum vote share (0.0, 1.0] for the winning direction to be actionable.
    # If no direction reaches this threshold → HOLD (is_actionable=False).
    # 0.65 is more conservative than the previous 0.60 default, reducing false signals
    # from correlated module clusters that would otherwise reach 0.60 easily.
    threshold: float = 0.65

    # Minimum number of signals required to produce an actionable result.
    # Fewer signals → HOLD regardless of vote share.
    # Require at least 3 independent module signals before opening a trade.
    min_signals: int = 3

    # Per-module weights for WEIGHTED_VOTE / DYNAMIC_VOTE.
    # Keys are module_id strings. Unknown modules fall back to default_weight.
    # Example: {"ema_crossover_h1": 1.5, "news_sentiment": 0.8}
    module_weights: dict[str, float] = {}

    # Fallback weight for modules not listed in module_weights.
    default_weight: float = 1.0

    # EMA smoothing factor for DYNAMIC_VOTE weight adaptation (0.0, 1.0].
    # Higher alpha = faster adaptation to recent performance.
    performance_alpha: float = 0.10

    # Minimum weight floor for DYNAMIC_VOTE (prevents a module from being zeroed).
    min_dynamic_weight: float = 0.10

    @field_validator("threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"threshold must be in (0.0, 1.0], got {v}")
        return v

    @field_validator("default_weight")
    @classmethod
    def _validate_default_weight(cls, v: float) -> float:
        if v <= 0.0:
            raise ValueError(f"default_weight must be positive, got {v}")
        return v

    @field_validator("performance_alpha")
    @classmethod
    def _validate_alpha(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"performance_alpha must be in (0.0, 1.0], got {v}")
        return v

    @field_validator("min_signals")
    @classmethod
    def _validate_min_signals(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"min_signals must be >= 1, got {v}")
        return v
