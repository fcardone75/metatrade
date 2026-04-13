"""Tests for core/contracts/signal.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from metatrade.core.contracts.signal import SIGNAL_SCHEMA_VERSION, AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion, SchemaVersion

UTC = timezone.utc
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
V100 = ModuleVersion(1, 0, 0)


def make_signal(**kwargs) -> AnalysisSignal:
    defaults = dict(
        direction=SignalDirection.BUY,
        confidence=0.75,
        reason="test reason",
        module_id="test_module",
        module_version=V100,
        timestamp_utc=T0,
    )
    defaults.update(kwargs)
    return AnalysisSignal(**defaults)


class TestAnalysisSignal:
    def test_buy_signal(self, buy_signal: AnalysisSignal) -> None:
        assert buy_signal.direction == SignalDirection.BUY
        assert buy_signal.is_actionable

    def test_sell_signal(self, sell_signal: AnalysisSignal) -> None:
        assert sell_signal.direction == SignalDirection.SELL
        assert sell_signal.is_actionable

    def test_hold_signal_not_actionable(self, hold_signal: AnalysisSignal) -> None:
        assert hold_signal.direction == SignalDirection.HOLD
        assert not hold_signal.is_actionable

    def test_confidence_at_boundaries(self) -> None:
        make_signal(confidence=0.0)
        make_signal(confidence=1.0)

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            make_signal(confidence=-0.01)

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            make_signal(confidence=1.01)

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            make_signal(timestamp_utc=datetime(2024, 1, 1, 10, 0))

    def test_empty_module_id_raises(self) -> None:
        with pytest.raises(ValueError, match="module_id"):
            make_signal(module_id="")

    def test_empty_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="reason"):
            make_signal(reason="")

    def test_incompatible_schema_version_raises(self) -> None:
        future_schema = SIGNAL_SCHEMA_VERSION.next()
        with pytest.raises(ValueError, match="schema"):
            make_signal(schema_version=future_schema)

    def test_metadata_optional(self) -> None:
        s = make_signal()
        assert s.metadata == {}

    def test_metadata_with_data(self) -> None:
        s = make_signal(metadata={"rsi": 72.5, "period": 14})
        assert s.metadata["rsi"] == 72.5

    def test_metadata_excluded_from_equality(self) -> None:
        s1 = make_signal(metadata={"x": 1})
        s2 = make_signal(metadata={"x": 999})
        assert s1 == s2

    def test_metadata_excluded_from_hash(self) -> None:
        s1 = make_signal(metadata={"x": 1})
        s2 = make_signal(metadata={"x": 999})
        assert hash(s1) == hash(s2)

    def test_frozen(self, buy_signal: AnalysisSignal) -> None:
        with pytest.raises(Exception):
            buy_signal.confidence = 0.99  # type: ignore[misc]

    def test_repr_contains_key_info(self, buy_signal: AnalysisSignal) -> None:
        r = repr(buy_signal)
        assert "ema_crossover_h1" in r
        assert "BUY" in r

    def test_signals_with_same_fields_are_equal(self) -> None:
        s1 = make_signal()
        s2 = make_signal()
        assert s1 == s2

    def test_signals_with_different_confidence_not_equal(self) -> None:
        s1 = make_signal(confidence=0.5)
        s2 = make_signal(confidence=0.9)
        assert s1 != s2

    def test_schema_version_defaults_to_current(self) -> None:
        s = make_signal()
        assert s.schema_version == SIGNAL_SCHEMA_VERSION
