"""Intermarket analysis configuration."""

from __future__ import annotations

from pydantic import Field

from metatrade.core.config_base import BaseConfig


class IntermarketConfig(BaseConfig):
    """Configuration for the intermarket analysis module.

    Override with environment variables prefixed ``INTERMARKET_``.
    """

    # Primary symbol being traded
    primary_symbol: str = Field(default="EURUSD")

    # Reference instruments used as correlation anchors
    reference_symbols: list[str] = Field(
        default_factory=lambda: ["USDJPY", "GBPUSD", "USDCHF"]
    )

    # Rolling window for correlation computation (in bars)
    correlation_window: int = Field(default=50, ge=10)

    # Maximum lag to test in cross-correlation analysis (in bars)
    lag_max_bars: int = Field(default=20, ge=1)

    # Window over which to measure correlation stability (in bars)
    stability_window: int = Field(default=100, ge=20)

    # Minimum |correlation| to consider a dependency significant
    min_correlation_strength: float = Field(default=0.4, ge=0.0, le=1.0)

    # Minimum stability score to use a dependency in decision making
    min_stability_score: float = Field(default=0.4, ge=0.0, le=1.0)

    # Minimum confidence to emit a non-HOLD signal from this module
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Maximum weight this module contributes to risk engine (0–1)
    max_module_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # Threshold for regime-break detection: mean correlation shift
    regime_break_threshold: float = Field(default=0.4, ge=0.1, le=1.0)

    model_config = {
        "env_prefix": "INTERMARKET_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "frozen": True,
    }
