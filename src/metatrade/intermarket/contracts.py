"""Data contracts for intermarket analysis output.

Two distinct layers:

1. Signal-enrichment layer (original):
   DependencySnapshot, LeadLagResult, IntermarketAnalysis
   Used by IntermarketModule (ITechnicalModule bridge).

2. Portfolio-risk layer (new):
   CurrencyCode, CurrencyExposure, PairCorrelation,
   IntermarketSnapshot, IntermarketDecision
   Used by IntermarketEngine (inserted between Consensus and RiskManager).

All objects are immutable dataclasses.  Mutable dict fields use
``field(hash=False, compare=False)`` so that ``frozen=True`` still works.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


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


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio-risk layer — IntermarketEngine contracts
# ══════════════════════════════════════════════════════════════════════════════

# CurrencyCode: a plain str alias for readability.
# Valid values: "EUR", "USD", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"
# Kept as str (not Enum) so the set is open for future instruments.
CurrencyCode = str


@dataclass(frozen=True)
class CurrencyExposure:
    """Net and gross exposure to a single currency, measured in lots.

    Attributes:
        currency:    Currency code, e.g. "EUR".
        net_lots:    Signed sum of lot exposure (+ = net long, - = net short).
        gross_lots:  Absolute sum of lot exposure (always >= 0).
    """

    currency: CurrencyCode
    net_lots: Decimal
    gross_lots: Decimal

    def __post_init__(self) -> None:
        if self.gross_lots < Decimal("0"):
            raise ValueError("CurrencyExposure.gross_lots must be >= 0")


@dataclass(frozen=True)
class PairCorrelation:
    """Rolling Pearson correlation between two symbols at a point in time.

    Attributes:
        symbol_a:        First symbol (alphabetically earlier by convention).
        symbol_b:        Second symbol.
        correlation:     Signed Pearson r in [-1, 1].
        abs_correlation: |correlation|, for threshold comparisons.
        lookback_bars:   Window size requested.
        overlap_bars:    Actual bars used (may be < lookback_bars).
        returns_mode:    "log" | "pct" — how returns were computed.
        timestamp_utc:   When this was computed.
    """

    symbol_a: str
    symbol_b: str
    correlation: float
    abs_correlation: float
    lookback_bars: int
    overlap_bars: int
    returns_mode: str
    timestamp_utc: datetime

    def __post_init__(self) -> None:
        if not (-1.0 - 1e-9 <= self.correlation <= 1.0 + 1e-9):
            raise ValueError(
                f"PairCorrelation.correlation must be in [-1, 1], got {self.correlation}"
            )
        if self.overlap_bars < 2:
            raise ValueError(
                f"PairCorrelation.overlap_bars must be >= 2, got {self.overlap_bars}"
            )


@dataclass(frozen=True)
class IntermarketSnapshot:
    """Point-in-time state of cross-pair correlations, currency exposures, and risk clusters.

    Produced by IntermarketEngine.evaluate() for observability and telemetry.
    All contained types are immutable, so this snapshot is safe to pass
    across thread/process boundaries without copying.
    """

    correlations: tuple[PairCorrelation, ...]
    exposures: tuple[CurrencyExposure, ...]
    risk_clusters: tuple[frozenset[str], ...]
    timestamp_utc: datetime


@dataclass(frozen=True)
class IntermarketDecision:
    """Output of IntermarketEngine for one candidate trade.

    Semantics
    ---------
    - ``approved=False``  →  veto: do not open the trade (check ``reason``).
    - ``approved=True``   →  proceed, scaling lot_size by ``risk_multiplier``.
    - ``risk_multiplier = 1.0``  →  no size change.
    - ``risk_multiplier < 1.0``  →  reduce lot_size proportionally.

    Attributes:
        approved:              Whether the trade is permitted.
        risk_multiplier:       Decimal multiplier to apply to the base lot size.
        correlated_positions:  Symbols of open positions that are correlated
                               with the candidate (empty if none).
        reason:                Human-readable explanation of the decision.
        warnings:              Non-blocking observations (e.g. medium correlation).
        exposure_delta_by_ccy: Currency exposure change if this trade is opened.
                               Excluded from hash/compare (dict is mutable).
        snapshot:              Full intermarket snapshot at decision time (optional).
    """

    approved: bool
    risk_multiplier: Decimal
    correlated_positions: tuple[str, ...]
    reason: str
    warnings: tuple[str, ...]
    # dict excluded from hash/compare so frozen=True works correctly
    exposure_delta_by_ccy: dict[CurrencyCode, Decimal] = field(
        default_factory=dict, hash=False, compare=False
    )
    snapshot: IntermarketSnapshot | None = None

    def __post_init__(self) -> None:
        if self.risk_multiplier <= Decimal("0"):
            raise ValueError(
                f"IntermarketDecision.risk_multiplier must be > 0, got {self.risk_multiplier}"
            )
        if not self.reason:
            raise ValueError("IntermarketDecision.reason cannot be empty")
