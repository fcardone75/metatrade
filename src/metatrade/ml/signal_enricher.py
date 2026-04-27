"""ML signal enricher — typed access layer and optional gating for MLModule signals."""

from __future__ import annotations

from dataclasses import dataclass

from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection


@dataclass(frozen=True)
class EnrichedMlSignal:
    """Typed view of ML-specific metadata attached to an AnalysisSignal.

    Use ``EnrichedMlSignal.from_signal()`` to extract fields from metadata
    without casting or key-access boilerplate at call sites.
    """

    signal: AnalysisSignal
    p_buy: float | None
    p_hold: float | None
    p_sell: float | None
    confidence_margin: float | None
    expected_value_score: float | None

    @classmethod
    def from_signal(cls, signal: AnalysisSignal) -> EnrichedMlSignal:
        meta = signal.metadata
        return cls(
            signal=signal,
            p_buy=meta.get("p_buy"),
            p_hold=meta.get("p_hold"),
            p_sell=meta.get("p_sell"),
            confidence_margin=meta.get("confidence_margin"),
            expected_value_score=meta.get("expected_value_score"),
        )

    @property
    def has_probabilities(self) -> bool:
        return self.p_buy is not None

    @property
    def has_ev_score(self) -> bool:
        return self.expected_value_score is not None


class MlSignalEnricher:
    """Optional gating layer for MLModule signals consumed by the consensus engine.

    Two independent gates, each configurable independently:

    *Confidence-margin gate* — downgrades a directional signal to HOLD when the
    gap between the top-1 and top-2 class probabilities is below
    ``min_confidence_margin``.  Filters ambiguous, near-tie predictions before
    they enter the consensus vote.

    *EV gate* — downgrades a directional signal to HOLD when the expected-value
    regressor score either contradicts the classifier direction or its magnitude
    is below ``ev_gate_min_r``.  Fires only when an EV score is present in the
    signal metadata (i.e. an EV model was trained and loaded).

    When a gate fires the original confidence is preserved; only ``direction``
    and ``reason`` change.  A flag ``"enricher_gate_fired": True`` is added to
    the returned signal's metadata so downstream consumers can distinguish
    enricher-downgraded HOLDs from native HOLDs.
    """

    def __init__(
        self,
        ev_gate_enabled: bool = False,
        ev_gate_min_r: float = 0.3,
        confidence_margin_gate_enabled: bool = False,
        min_confidence_margin: float = 0.10,
    ) -> None:
        if ev_gate_min_r <= 0.0:
            raise ValueError(f"ev_gate_min_r must be > 0, got {ev_gate_min_r}")
        if not 0.0 <= min_confidence_margin <= 1.0:
            raise ValueError(
                f"min_confidence_margin must be in [0, 1], got {min_confidence_margin}"
            )
        self._ev_gate_enabled = ev_gate_enabled
        self._ev_gate_min_r = ev_gate_min_r
        self._cm_gate_enabled = confidence_margin_gate_enabled
        self._min_cm = min_confidence_margin

    # ------------------------------------------------------------------
    def enrich(self, signal: AnalysisSignal) -> AnalysisSignal:
        """Apply gating rules; returns a (possibly downgraded) AnalysisSignal.

        HOLD signals are returned unchanged — gates only filter directional
        signals that don't meet the quality bar.
        """
        if signal.direction == SignalDirection.HOLD:
            return signal

        enriched = EnrichedMlSignal.from_signal(signal)
        new_direction = signal.direction
        new_reason = signal.reason

        # Gate 1: confidence margin
        if (
            self._cm_gate_enabled
            and enriched.confidence_margin is not None
            and enriched.confidence_margin < self._min_cm
        ):
            new_direction = SignalDirection.HOLD
            new_reason = (
                f"{signal.reason} [enricher: low margin "
                f"{enriched.confidence_margin:.3f} < {self._min_cm:.3f}]"
            )

        # Gate 2: EV alignment (only evaluated when direction still directional)
        if (
            new_direction != SignalDirection.HOLD
            and self._ev_gate_enabled
            and enriched.expected_value_score is not None
        ):
            ev = enriched.expected_value_score
            aligned = (
                signal.direction == SignalDirection.BUY and ev >= self._ev_gate_min_r
            ) or (
                signal.direction == SignalDirection.SELL and ev <= -self._ev_gate_min_r
            )
            if not aligned:
                new_direction = SignalDirection.HOLD
                new_reason = (
                    f"{signal.reason} [enricher: EV {ev:+.3f} conflicts direction]"
                )

        if new_direction == signal.direction:
            return signal

        return AnalysisSignal(
            direction=new_direction,
            confidence=signal.confidence,
            reason=new_reason,
            module_id=signal.module_id,
            module_version=signal.module_version,
            timestamp_utc=signal.timestamp_utc,
            schema_version=signal.schema_version,
            metadata={**signal.metadata, "enricher_gate_fired": True},
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: object) -> MlSignalEnricher:
        """Construct from an MLConfig (or any object with the expected attributes)."""
        return cls(
            ev_gate_enabled=getattr(cfg, "ev_gate_enabled", False),
            ev_gate_min_r=getattr(cfg, "ev_gate_min_r", 0.3),
            confidence_margin_gate_enabled=getattr(cfg, "cm_gate_enabled", False),
            min_confidence_margin=getattr(cfg, "min_confidence_margin", 0.10),
        )
