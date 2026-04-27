"""Tests for ml/signal_enricher.py — EnrichedMlSignal and MlSignalEnricher."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.ml.signal_enricher import EnrichedMlSignal, MlSignalEnricher


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_VER = ModuleVersion(1, 0, 0)


def _signal(
    direction: SignalDirection = SignalDirection.BUY,
    confidence: float = 0.70,
    metadata: dict | None = None,
) -> AnalysisSignal:
    return AnalysisSignal(
        direction=direction,
        confidence=confidence,
        reason="test signal",
        module_id="ml_h1",
        module_version=_VER,
        timestamp_utc=_NOW,
        metadata=metadata or {},
    )


def _buy_signal(p_buy=0.65, p_hold=0.20, p_sell=0.15, margin=0.45, ev=None):
    meta: dict = {
        "p_buy": p_buy,
        "p_hold": p_hold,
        "p_sell": p_sell,
        "confidence_margin": margin,
    }
    if ev is not None:
        meta["expected_value_score"] = ev
    return _signal(direction=SignalDirection.BUY, metadata=meta)


def _sell_signal(p_buy=0.15, p_hold=0.20, p_sell=0.65, margin=0.45, ev=None):
    meta: dict = {
        "p_buy": p_buy,
        "p_hold": p_hold,
        "p_sell": p_sell,
        "confidence_margin": margin,
    }
    if ev is not None:
        meta["expected_value_score"] = ev
    return _signal(direction=SignalDirection.SELL, metadata=meta)


# ---------------------------------------------------------------------------
# EnrichedMlSignal.from_signal()
# ---------------------------------------------------------------------------


class TestEnrichedMlSignalFromSignal:
    def test_extracts_probabilities(self) -> None:
        sig = _buy_signal(p_buy=0.60, p_hold=0.25, p_sell=0.15, margin=0.35)
        e = EnrichedMlSignal.from_signal(sig)
        assert abs(e.p_buy - 0.60) < 1e-9
        assert abs(e.p_hold - 0.25) < 1e-9
        assert abs(e.p_sell - 0.15) < 1e-9

    def test_extracts_confidence_margin(self) -> None:
        sig = _buy_signal(margin=0.22)
        e = EnrichedMlSignal.from_signal(sig)
        assert abs(e.confidence_margin - 0.22) < 1e-9

    def test_extracts_ev_score(self) -> None:
        sig = _buy_signal(ev=0.45)
        e = EnrichedMlSignal.from_signal(sig)
        assert abs(e.expected_value_score - 0.45) < 1e-9

    def test_none_when_fields_absent(self) -> None:
        sig = _signal(metadata={})
        e = EnrichedMlSignal.from_signal(sig)
        assert e.p_buy is None
        assert e.p_hold is None
        assert e.p_sell is None
        assert e.confidence_margin is None
        assert e.expected_value_score is None

    def test_signal_reference_preserved(self) -> None:
        sig = _buy_signal()
        e = EnrichedMlSignal.from_signal(sig)
        assert e.signal is sig


class TestEnrichedMlSignalProperties:
    def test_has_probabilities_true(self) -> None:
        e = EnrichedMlSignal.from_signal(_buy_signal())
        assert e.has_probabilities is True

    def test_has_probabilities_false_when_absent(self) -> None:
        e = EnrichedMlSignal.from_signal(_signal(metadata={}))
        assert e.has_probabilities is False

    def test_has_ev_score_true(self) -> None:
        e = EnrichedMlSignal.from_signal(_buy_signal(ev=0.5))
        assert e.has_ev_score is True

    def test_has_ev_score_false_when_absent(self) -> None:
        e = EnrichedMlSignal.from_signal(_buy_signal())
        assert e.has_ev_score is False


# ---------------------------------------------------------------------------
# MlSignalEnricher — constructor validation
# ---------------------------------------------------------------------------


class TestMlSignalEnricherInit:
    def test_invalid_ev_gate_min_r_raises(self) -> None:
        with pytest.raises(ValueError, match="ev_gate_min_r"):
            MlSignalEnricher(ev_gate_enabled=True, ev_gate_min_r=0.0)

    def test_invalid_min_confidence_margin_raises(self) -> None:
        with pytest.raises(ValueError, match="min_confidence_margin"):
            MlSignalEnricher(min_confidence_margin=1.5)

    def test_valid_defaults_ok(self) -> None:
        e = MlSignalEnricher()
        assert e is not None


# ---------------------------------------------------------------------------
# MlSignalEnricher.enrich() — no gates
# ---------------------------------------------------------------------------


class TestEnrichNoGates:
    def test_buy_returned_unchanged(self) -> None:
        enricher = MlSignalEnricher()
        sig = _buy_signal()
        result = enricher.enrich(sig)
        assert result is sig

    def test_sell_returned_unchanged(self) -> None:
        enricher = MlSignalEnricher()
        sig = _sell_signal()
        result = enricher.enrich(sig)
        assert result is sig

    def test_hold_returned_unchanged(self) -> None:
        enricher = MlSignalEnricher(
            ev_gate_enabled=True,
            confidence_margin_gate_enabled=True,
        )
        sig = _signal(direction=SignalDirection.HOLD, metadata={"confidence_margin": 0.0})
        result = enricher.enrich(sig)
        assert result is sig


# ---------------------------------------------------------------------------
# Confidence-margin gate
# ---------------------------------------------------------------------------


class TestConfidenceMarginGate:
    def _enricher(self, threshold: float = 0.15) -> MlSignalEnricher:
        return MlSignalEnricher(
            confidence_margin_gate_enabled=True,
            min_confidence_margin=threshold,
        )

    def test_buy_downgraded_when_margin_too_low(self) -> None:
        sig = _buy_signal(margin=0.05)
        result = self._enricher(threshold=0.15).enrich(sig)
        assert result.direction == SignalDirection.HOLD

    def test_sell_downgraded_when_margin_too_low(self) -> None:
        sig = _sell_signal(margin=0.05)
        result = self._enricher(threshold=0.15).enrich(sig)
        assert result.direction == SignalDirection.HOLD

    def test_buy_kept_when_margin_sufficient(self) -> None:
        sig = _buy_signal(margin=0.20)
        result = self._enricher(threshold=0.15).enrich(sig)
        assert result.direction == SignalDirection.BUY

    def test_no_fire_when_margin_absent(self) -> None:
        sig = _signal(direction=SignalDirection.BUY, metadata={})
        result = self._enricher().enrich(sig)
        assert result is sig

    def test_confidence_preserved_on_downgrade(self) -> None:
        sig = _buy_signal(margin=0.02)
        result = self._enricher(threshold=0.15).enrich(sig)
        assert result.confidence == sig.confidence

    def test_gate_fired_flag_in_metadata(self) -> None:
        sig = _buy_signal(margin=0.02)
        result = self._enricher(threshold=0.15).enrich(sig)
        assert result.metadata.get("enricher_gate_fired") is True

    def test_reason_mentions_margin(self) -> None:
        sig = _buy_signal(margin=0.02)
        result = self._enricher(threshold=0.15).enrich(sig)
        assert "margin" in result.reason

    def test_boundary_equal_to_threshold_passes(self) -> None:
        sig = _buy_signal(margin=0.15)
        result = self._enricher(threshold=0.15).enrich(sig)
        # exactly at threshold: 0.15 < 0.15 is False → should NOT fire
        assert result.direction == SignalDirection.BUY


# ---------------------------------------------------------------------------
# EV gate
# ---------------------------------------------------------------------------


class TestEvGate:
    def _enricher(self, min_r: float = 0.3) -> MlSignalEnricher:
        return MlSignalEnricher(ev_gate_enabled=True, ev_gate_min_r=min_r)

    def test_buy_downgraded_when_ev_negative(self) -> None:
        sig = _buy_signal(ev=-0.4)
        result = self._enricher().enrich(sig)
        assert result.direction == SignalDirection.HOLD

    def test_buy_downgraded_when_ev_too_small(self) -> None:
        sig = _buy_signal(ev=0.10)
        result = self._enricher(min_r=0.3).enrich(sig)
        assert result.direction == SignalDirection.HOLD

    def test_buy_kept_when_ev_aligned(self) -> None:
        sig = _buy_signal(ev=0.50)
        result = self._enricher().enrich(sig)
        assert result.direction == SignalDirection.BUY

    def test_sell_downgraded_when_ev_positive(self) -> None:
        sig = _sell_signal(ev=0.4)
        result = self._enricher().enrich(sig)
        assert result.direction == SignalDirection.HOLD

    def test_sell_downgraded_when_ev_too_small_negative(self) -> None:
        sig = _sell_signal(ev=-0.10)
        result = self._enricher(min_r=0.3).enrich(sig)
        assert result.direction == SignalDirection.HOLD

    def test_sell_kept_when_ev_aligned(self) -> None:
        sig = _sell_signal(ev=-0.50)
        result = self._enricher().enrich(sig)
        assert result.direction == SignalDirection.SELL

    def test_no_fire_when_ev_absent(self) -> None:
        sig = _buy_signal()  # no ev key
        result = self._enricher().enrich(sig)
        assert result is sig

    def test_reason_mentions_ev_on_downgrade(self) -> None:
        sig = _buy_signal(ev=-0.4)
        result = self._enricher().enrich(sig)
        assert "EV" in result.reason or "ev" in result.reason.lower()

    def test_gate_fired_flag_set(self) -> None:
        sig = _buy_signal(ev=-0.4)
        result = self._enricher().enrich(sig)
        assert result.metadata.get("enricher_gate_fired") is True


# ---------------------------------------------------------------------------
# Both gates together
# ---------------------------------------------------------------------------


class TestBothGates:
    def _enricher(self) -> MlSignalEnricher:
        return MlSignalEnricher(
            confidence_margin_gate_enabled=True,
            min_confidence_margin=0.15,
            ev_gate_enabled=True,
            ev_gate_min_r=0.3,
        )

    def test_cm_gate_fires_first_ev_not_evaluated(self) -> None:
        # margin too low AND ev conflicts — cm gate should fire (first)
        sig = _buy_signal(margin=0.05, ev=-0.5)
        result = self._enricher().enrich(sig)
        assert result.direction == SignalDirection.HOLD
        assert "margin" in result.reason

    def test_ev_gate_fires_when_cm_passes(self) -> None:
        # margin sufficient, ev contradicts
        sig = _buy_signal(margin=0.30, ev=-0.5)
        result = self._enricher().enrich(sig)
        assert result.direction == SignalDirection.HOLD
        assert "EV" in result.reason or "ev" in result.reason.lower()

    def test_no_downgrade_when_both_pass(self) -> None:
        sig = _buy_signal(margin=0.30, ev=0.50)
        result = self._enricher().enrich(sig)
        assert result is sig


# ---------------------------------------------------------------------------
# from_config()
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_reads_config_attributes(self) -> None:
        class Cfg:
            ev_gate_enabled = True
            ev_gate_min_r = 0.5
            cm_gate_enabled = True
            min_confidence_margin = 0.20

        enricher = MlSignalEnricher.from_config(Cfg())
        # cm gate fires if margin is 0.10 (below 0.20 threshold)
        sig = _buy_signal(margin=0.10)
        result = enricher.enrich(sig)
        assert result.direction == SignalDirection.HOLD

    def test_defaults_when_attributes_missing(self) -> None:
        class Cfg:
            pass

        enricher = MlSignalEnricher.from_config(Cfg())
        # both gates disabled by default → signal unchanged
        sig = _buy_signal(margin=0.0, ev=-1.0)
        result = enricher.enrich(sig)
        assert result is sig
