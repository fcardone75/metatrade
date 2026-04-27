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
from metatrade.core.log import get_logger
from metatrade.core.versioning import ModuleVersion
from metatrade.ml.features import MIN_FEATURE_BARS, extract_features
from metatrade.ml.registry import ModelRegistry
from metatrade.ml.signal_enricher import MlSignalEnricher
from metatrade.technical_analysis.interface import ITechnicalModule

log = get_logger(__name__)

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
        enricher: MlSignalEnricher | None = None,
    ) -> None:
        self._registry = registry
        self._timeframe_id = timeframe_id
        self._min_confidence = min_confidence
        self._enricher = enricher

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
            prediction = snapshot.classifier.predict_raw(fv)
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

        direction_int = prediction.direction
        confidence = prediction.confidence

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
            }.get(direction_int, SignalDirection.HOLD)
            reason = (
                f"{self.module_id}: predicted {direction.value} "
                f"(confidence={confidence:.2f}, model={snapshot.version})"
            )

        log.debug(
            "ml_prediction",
            module=self.module_id,
            direction=direction.value,
            confidence=round(confidence, 4),
            raw_direction=prediction.raw_direction,
            model_version=snapshot.version,
        )

        meta: dict[str, object] = {
            "model_version": snapshot.version,
            "direction_int": direction_int,
            "raw_direction": prediction.raw_direction,
            "confidence": confidence,
            "model_available": True,
            "features_ok": True,
            "feature_count": len(fv.to_list()),
        }
        if prediction.p_buy is not None:
            meta["p_buy"] = round(prediction.p_buy, 4)
        if prediction.p_hold is not None:
            meta["p_hold"] = round(prediction.p_hold, 4)
        if prediction.p_sell is not None:
            meta["p_sell"] = round(prediction.p_sell, 4)
        if prediction.confidence_margin is not None:
            meta["confidence_margin"] = round(prediction.confidence_margin, 4)

        if snapshot.ev_trainer is not None and snapshot.ev_trainer.is_trained:
            try:
                ev_score = snapshot.ev_trainer.predict(fv)
                meta["expected_value_score"] = round(ev_score, 4)
            except RuntimeError:
                pass

        signal = AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata=meta,
        )

        if self._enricher is not None:
            signal = self._enricher.enrich(signal)

        return signal
