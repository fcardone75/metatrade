"""ML analysis module — implements ITechnicalModule using a trained classifier.

This module bridges the ML subsystem with the consensus voting engine.
It extracts features from the current bar window, queries the active
classifier, and emits an AnalysisSignal that can be weighted alongside
technical indicators in the consensus engine.

The module returns HOLD with low confidence (0.50) if:
  - No trained model is available in the registry
  - Feature extraction fails (insufficient bars)
  - Predicted confidence is below the minimum threshold
"""

from __future__ import annotations

from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.ml.features import MIN_FEATURE_BARS, extract_features
from metatrade.ml.registry import ModelRegistry
from metatrade.technical_analysis.interface import ITechnicalModule

_VERSION = ModuleVersion(1, 0, 0)

# Minimum classifier confidence to emit a non-HOLD signal
_MIN_CONFIDENCE_THRESHOLD = 0.55


class MLModule(ITechnicalModule):
    """ML signal module backed by a trained RandomForest classifier.

    Args:
        registry:            ModelRegistry holding the trained model.
        timeframe_id:        Suffix for module_id (default "h1").
        min_confidence:      Minimum class probability to emit a directional signal.
                             Below this threshold, HOLD is returned.
    """

    def __init__(
        self,
        registry: ModelRegistry | None,
        timeframe_id: str = "h1",
        min_confidence: float = _MIN_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._registry = registry
        self._timeframe_id = timeframe_id
        self._min_confidence = min_confidence

    @property
    def module_id(self) -> str:
        return f"ml_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        return MIN_FEATURE_BARS

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        if self._registry is None:
            return AnalysisSignal(
                direction=SignalDirection.HOLD,
                confidence=0.50,
                reason=f"{self.module_id}: registry not available",
                module_id=self.module_id,
                module_version=_VERSION,
                timestamp_utc=timestamp_utc,
                metadata={"model_available": False, "registry_available": False},
            )

        snapshot = self._registry.get_active()

        if snapshot is None or not snapshot.classifier.is_trained:
            return AnalysisSignal(
                direction=SignalDirection.HOLD,
                confidence=0.50,
                reason=f"{self.module_id}: no trained model available",
                module_id=self.module_id,
                module_version=_VERSION,
                timestamp_utc=timestamp_utc,
                metadata={"model_available": False},
            )

        fv = extract_features(bars, timestamp_utc=timestamp_utc)
        if fv is None:
            return AnalysisSignal(
                direction=SignalDirection.HOLD,
                confidence=0.50,
                reason=f"{self.module_id}: feature extraction failed",
                module_id=self.module_id,
                module_version=_VERSION,
                timestamp_utc=timestamp_utc,
                metadata={"model_available": True, "features_ok": False},
            )

        try:
            raw_direction, confidence = snapshot.classifier.predict(fv)
        except RuntimeError:
            return AnalysisSignal(
                direction=SignalDirection.HOLD,
                confidence=0.50,
                reason=f"{self.module_id}: prediction error",
                module_id=self.module_id,
                module_version=_VERSION,
                timestamp_utc=timestamp_utc,
                metadata={"model_available": True, "features_ok": True},
            )

        # Low-confidence predictions are downgraded to HOLD
        if confidence < self._min_confidence:
            direction = SignalDirection.HOLD
            reason = (
                f"{self.module_id}: low confidence ({confidence:.2f} < "
                f"{self._min_confidence:.2f}), HOLD"
            )
        else:
            direction = {
                1: SignalDirection.BUY,
                -1: SignalDirection.SELL,
                0: SignalDirection.HOLD,
            }.get(raw_direction, SignalDirection.HOLD)
            reason = (
                f"{self.module_id}: predicted {direction.value} "
                f"(confidence={confidence:.2f}, model={snapshot.version})"
            )

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "model_version": snapshot.version,
                "raw_direction": raw_direction,
                "confidence": confidence,
                "model_available": True,
                "features_ok": True,
            },
        )
