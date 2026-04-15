"""Intermarket analysis configuration.

Two layers of configuration are bundled here under one ``INTERMARKET_`` prefix:

1. **Signal-enrichment layer** (original ``IntermarketModule`` / ``ITechnicalModule``):
   Fields: ``primary_symbol``, ``reference_symbols``, ``correlation_window``,
   ``lag_max_bars``, ``stability_window``, ``min_correlation_strength``,
   ``min_stability_score``, ``min_confidence``, ``max_module_weight``,
   ``regime_break_threshold``.

2. **Portfolio-risk layer** (new ``IntermarketEngine``):
   Fields: ``enabled``, ``symbols``, ``corr_*``, ``exposure_*``,
   ``portfolio_*``, ``risk_*``.

All are overridable via environment variables prefixed ``INTERMARKET_``,
e.g. ``INTERMARKET_ENABLED=true``, ``INTERMARKET_CORR_LOOKBACK_BARS=200``.
"""

from __future__ import annotations

from pydantic import Field

from metatrade.core.config_base import BaseConfig


class IntermarketConfig(BaseConfig):
    """Unified configuration for both the signal-enrichment and portfolio-risk
    intermarket layers.

    Override with environment variables prefixed ``INTERMARKET_``.
    """

    # ── Signal-enrichment layer (IntermarketModule / ITechnicalModule) ─────────

    primary_symbol: str = Field(default="EURUSD")
    reference_symbols: list[str] = Field(
        default_factory=lambda: ["USDJPY", "GBPUSD", "USDCHF"]
    )
    correlation_window: int = Field(default=50, ge=10)
    lag_max_bars: int = Field(default=20, ge=1)
    stability_window: int = Field(default=100, ge=20)
    min_correlation_strength: float = Field(default=0.4, ge=0.0, le=1.0)
    min_stability_score: float = Field(default=0.4, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_module_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    regime_break_threshold: float = Field(default=0.4, ge=0.1, le=1.0)

    # ── Portfolio-risk layer master switch ────────────────────────────────────

    # Set INTERMARKET_ENABLED=true to activate the portfolio-risk engine.
    enabled: bool = Field(default=False)

    # Symbols monitored by the IntermarketEngine (all must feed bar data).
    symbols: list[str] = Field(
        default_factory=lambda: [
            "EURUSD", "GBPUSD", "USDCHF", "USDJPY",
            "AUDUSD", "NZDUSD", "USDCAD", "EURGBP", "EURJPY", "GBPJPY",
        ]
    )

    # ── Correlation settings ──────────────────────────────────────────────────

    # "log" (log returns) | "pct" (percentage returns)
    corr_returns_mode: str = Field(default="pct")

    # Number of bars in the rolling correlation window.
    corr_lookback_bars: int = Field(default=200, ge=10)

    # Minimum bars of overlap required to produce a valid correlation.
    # Correlations with fewer overlapping bars are discarded.
    corr_min_overlap_bars: int = Field(default=150, ge=10)

    # |correlation| >= this → strong correlation (high risk-reduction).
    corr_strong_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    # |correlation| >= this → medium correlation (moderate risk-reduction).
    corr_medium_threshold: float = Field(default=0.60, ge=0.0, le=1.0)

    # ── Exposure limits ───────────────────────────────────────────────────────

    # Maximum net lot exposure per currency (signed sum across all positions).
    # E.g. 2.0 means at most 2 lots net long or 2 lots net short in EUR.
    exposure_max_net: float = Field(default=2.0, gt=0.0)

    # Maximum gross lot exposure per currency (absolute sum).
    exposure_max_gross: float = Field(default=3.0, gt=0.0)

    # ── Portfolio / cluster limits ────────────────────────────────────────────

    # Maximum concurrent positions in the same risk cluster.
    portfolio_max_corr_positions_per_cluster: int = Field(default=2, ge=1)

    # |correlation| threshold used to identify "same idea" duplicate trades.
    portfolio_duplicate_idea_abs_corr: float = Field(default=0.75, ge=0.0, le=1.0)

    # ── Risk-adjustment rules ─────────────────────────────────────────────────

    # lot_size multiplier when medium correlation is detected.
    risk_scale_down_medium_corr: float = Field(default=0.75, gt=0.0, le=1.0)

    # lot_size multiplier when strong correlation is detected.
    risk_scale_down_high_corr: float = Field(default=0.50, gt=0.0, le=1.0)

    # If True, veto when a new trade duplicates an open position's idea and
    # the cluster position limit has been reached.
    risk_veto_same_idea_same_dir: bool = Field(default=True)

    # If True, veto when adding this trade would exceed exposure limits.
    risk_veto_if_exposure_limit_exceeded: bool = Field(default=True)

    model_config = {
        "env_prefix": "INTERMARKET_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "frozen": True,
    }
