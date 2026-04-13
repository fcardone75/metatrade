"""Data contracts for intermarket analysis output.

Every detected dependency is represented as a DependencySnapshot — a
point-in-time description of the statistical relationship between two
instruments. DependencySnapshot objects are intentionally immutable:
a new one is produced on each analysis cycle.

IntermarketAnalysis aggregates all snapshots from a single analysis pass
into a single directional signal and a flat feature vector for ML use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class DependencySnapshot:
    """Statistical dependency between two instruments at a point in time.

    Attributes:
        source_instrument:    Instrument whose movement is observed.
        target_instrument:    Instrument whose future movement is predicted.
        relationship_type:    "pearson" | "spearman" | "cross_corr" | "covariance".
        lag_bars:             Bars by which source leads target
                              (positive = source leads, negative = target leads,
                               0 = contemporaneous).
        directionality:       +1 (positive correlation), -1 (inverse), 0 (undefined).
        strength_score:       Absolute correlation magnitude, in [0, 1].
        stability_score:      Stability over the tracking window, in [0, 1].
                              High = relationship has been consistent recently.
        confidence:           Combined reliability score = strength * stability.
        valid_from:           UTC timestamp of the oldest bar used in the window.
        valid_to:             None if currently active, else expiry timestamp.
        current_signal_impact: Directional signal contribution, in [-1, 1].
                              Accounts for the reference instrument's recent move.
        warnings:             Human-readable alerts (e.g. regime break detected).
        timestamp_utc:        When this snapshot was produced.
    """

    source_instrument: str
    target_instrument: str
    relationship_type: str
    lag_bars: int
    directionality: int                # +1, -1, or 0
    strength_score: float              # [0, 1]
    stability_score: float             # [0, 1]
    confidence: float                  # [0, 1]
    valid_from: datetime
    valid_to: datetime | None
    current_signal_impact: float       # [-1, 1]
    warnings: tuple[str, ...]          # frozen-friendly
    timestamp_utc: datetime


@dataclass(frozen=True)
class LeadLagResult:
    """Result of cross-correlation analysis across multiple lags.

    Attributes:
        dominant_lag:      Lag (in bars) with highest absolute correlation.
                           Positive = source leads target.
        max_correlation:   Correlation value at the dominant lag (signed).
        lag_profile:       Full map of lag → correlation for all tested lags.
        confidence:        Sharpness of the dominant peak vs. average, in [0, 1].
                           High = clear dominant lag; low = noisy / no lag.
    """

    dominant_lag: int
    max_correlation: float
    lag_profile: dict[int, float]
    confidence: float


@dataclass
class IntermarketAnalysis:
    """Aggregated output of one intermarket analysis pass.

    Attributes:
        primary_symbol:     The instrument being traded.
        dependencies:       All detected DependencySnapshots.
        signal_direction:   -1 (SELL), 0 (HOLD), +1 (BUY).
        overall_confidence: Mean confidence of contributing dependencies.
        features:           Flat dict of ML-ready features.
        warnings:           All warnings from all active dependencies.
        timestamp_utc:      Analysis timestamp.
    """

    primary_symbol: str
    dependencies: list[DependencySnapshot] = field(default_factory=list)
    signal_direction: int = 0
    overall_confidence: float = 0.0
    features: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now())
