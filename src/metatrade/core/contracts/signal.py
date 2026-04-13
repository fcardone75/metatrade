"""AnalysisSignal — the output contract for every analysis module.

Every IAnalysisModule (technical, fundamental, ML) must return an AnalysisSignal.
The consensus engine aggregates a list of AnalysisSignals into a ConsensusResult.

Schema version is bumped only on breaking changes to the field layout.
Module version tracks which release of the analysis code produced this signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion, SchemaVersion

# Current schema version — increment on any breaking contract change.
SIGNAL_SCHEMA_VERSION = SchemaVersion(1)


@dataclass(frozen=True)
class AnalysisSignal:
    """Output of a single analysis module.

    Fields:
        direction       BUY, SELL, or HOLD.
        confidence      Certainty of the direction, in [0.0, 1.0].
                        0.5 = coin-flip; 0.9 = high conviction.
                        Note: HOLD signals should also carry a confidence
                        (how confident the module is that *nothing* should happen).
        reason          Human-readable one-line explanation for audit logs.
        module_id       Unique stable identifier for the producing module
                        (e.g. "ema_crossover_h1", "rsi_oversold_m15").
        module_version  Semver of the module that produced this signal.
        timestamp_utc   UTC time at which the signal was generated.
        schema_version  Contract version — used for compatibility checks.
        metadata        Optional extra data (indicator values, feature importances).
                        Excluded from equality and hashing.
    """

    direction: SignalDirection
    confidence: float
    reason: str
    module_id: str
    module_version: ModuleVersion
    timestamp_utc: datetime
    schema_version: SchemaVersion = field(default=SIGNAL_SCHEMA_VERSION)
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("AnalysisSignal.timestamp_utc must be timezone-aware (UTC)")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"AnalysisSignal.confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if not self.module_id:
            raise ValueError("AnalysisSignal.module_id cannot be empty")
        if not self.reason:
            raise ValueError("AnalysisSignal.reason cannot be empty")
        if not self.schema_version.is_compatible_with(SIGNAL_SCHEMA_VERSION):
            raise ValueError(
                f"AnalysisSignal schema mismatch: "
                f"got {self.schema_version}, current is {SIGNAL_SCHEMA_VERSION}"
            )

    @property
    def is_actionable(self) -> bool:
        """True for BUY or SELL signals (not HOLD)."""
        return self.direction != SignalDirection.HOLD

    def __repr__(self) -> str:
        return (
            f"AnalysisSignal("
            f"module={self.module_id!r}, "
            f"direction={self.direction.value}, "
            f"confidence={self.confidence:.2f}, "
            f"reason={self.reason!r})"
        )
