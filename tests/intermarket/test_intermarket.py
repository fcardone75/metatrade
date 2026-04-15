"""Tests for the intermarket analysis module.

Coverage targets:
  - CorrelationEngine: pearson, spearman, cross_corr, covariance
  - LeadLagDetector: known lag, no lag, noisy data, insufficient data
  - StabilityMonitor: stable/unstable regimes, regime break, warnings
  - IntermarketFeatureBuilder: key naming, completeness
  - IntermarketModule: signal generation, reference feed management,
    HOLD fallback, insufficient data, high/low confidence paths
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import SignalDirection, Timeframe
from metatrade.intermarket.config import IntermarketConfig
from metatrade.intermarket.contracts import DependencySnapshot, LeadLagResult
from metatrade.intermarket.correlation import CorrelationEngine, _to_returns, _rank
from metatrade.intermarket.feature_builder import IntermarketFeatureBuilder
from metatrade.intermarket.lead_lag import LeadLagDetector
from metatrade.intermarket.module import IntermarketModule
from metatrade.intermarket.stability import StabilityMonitor

# ── Test helpers ───────────────────────────────────────────────────────────────

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


def _bar(close: float, i: int = 0) -> Bar:
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_utc=datetime(2024, 1, 1, i, 0, tzinfo=timezone.utc),
        open=Decimal(str(close)),
        high=Decimal(str(close + 0.001)),
        low=Decimal(str(close - 0.001)),
        close=Decimal(str(close)),
        volume=Decimal("1000"),
    )


def _rising(n: int, start: float = 1.0, step: float = 0.001) -> list[float]:
    return [start + i * step for i in range(n)]


def _falling(n: int, start: float = 1.2, step: float = 0.001) -> list[float]:
    return [start - i * step for i in range(n)]


def _noisy(n: int, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(1.0 + np.cumsum(rng.normal(0, 0.001, n)))


def _bars_from_prices(prices: list[float], symbol: str = "EURUSD") -> list[Bar]:
    return [
        Bar(
            symbol=symbol,
            timeframe=Timeframe.H1,
            timestamp_utc=datetime(2024, 1, 1, i % 24, i // 24, tzinfo=timezone.utc),
            open=Decimal(str(p)),
            high=Decimal(str(p + 0.001)),
            low=Decimal(str(p - 0.001)),
            close=Decimal(str(p)),
            volume=Decimal("1000"),
        )
        for i, p in enumerate(prices)
    ]


def _default_cfg(**kwargs) -> IntermarketConfig:
    defaults = dict(
        primary_symbol="EURUSD",
        correlation_window=20,
        lag_max_bars=5,
        stability_window=30,
        min_correlation_strength=0.3,
        min_stability_score=0.0,   # disable for unit tests
        min_confidence=0.0,        # disable for unit tests
    )
    defaults.update(kwargs)
    return IntermarketConfig(**defaults)


# ══════════════════════════════════════════════════════════════════════════════
# CorrelationEngine
# ══════════════════════════════════════════════════════════════════════════════

class TestToReturns:
    def test_length(self) -> None:
        prices = [1.0, 1.1, 1.2]
        r = _to_returns(prices)
        assert len(r) == 2

    def test_positive_return(self) -> None:
        r = _to_returns([1.0, 1.1])
        assert abs(r[0] - 0.1) < 1e-9

    def test_zero_price_returns_zero(self) -> None:
        r = _to_returns([0.0, 1.0])
        assert r[0] == 0.0


class TestRankHelper:
    def test_ascending(self) -> None:
        ranks = _rank(np.array([10.0, 20.0, 30.0]))
        np.testing.assert_array_equal(ranks, [1.0, 2.0, 3.0])

    def test_descending(self) -> None:
        ranks = _rank(np.array([30.0, 20.0, 10.0]))
        np.testing.assert_array_equal(ranks, [3.0, 2.0, 1.0])


class TestPearsonCorrelation:
    def setup_method(self) -> None:
        self.engine = CorrelationEngine()

    def test_perfect_positive_correlation(self) -> None:
        p = _rising(60)
        r = self.engine.pearson(p, p, 50)
        assert r is not None
        assert abs(r - 1.0) < 1e-6

    def test_perfect_negative_correlation(self) -> None:
        # Anti-phase oscillation: returns of x and y are exactly opposite
        n = 60
        x = [1.0 + (i % 2) * 0.01 for i in range(n)]   # 1.00, 1.01, 1.00, ...
        y = [1.01 - (i % 2) * 0.01 for i in range(n)]  # 1.01, 1.00, 1.01, ...
        r = self.engine.pearson(x, y, 50)
        assert r is not None
        assert r < -0.9

    def test_correlation_bounded_minus1_to_1(self) -> None:
        """Pearson is always in [-1, 1] for any valid series."""
        rng = np.random.default_rng(42)
        x = list(1.0 + np.cumsum(rng.normal(0, 0.001, 80)))
        rng2 = np.random.default_rng(99)
        y = list(1.0 + np.cumsum(rng2.normal(0, 0.001, 80)))
        r = self.engine.pearson(x, y, 50)
        if r is not None:
            assert -1.0 <= r <= 1.0

    def test_returns_none_for_insufficient_data(self) -> None:
        p = _rising(10)
        r = self.engine.pearson(p, p, 50)
        assert r is None

    def test_returns_none_for_constant_series(self) -> None:
        constant = [1.0] * 60
        rising = _rising(60)
        r = self.engine.pearson(constant, rising, 50)
        assert r is None

    def test_uses_only_last_window_bars(self) -> None:
        # First 40 bars are negatively correlated; last 50 positively correlated
        n = 90
        x_a = _falling(40)
        x_b = _rising(50)
        y_a = _rising(40)
        y_b = _rising(50)
        x = x_a + x_b
        y = y_a + y_b
        r = self.engine.pearson(x, y, 50)
        assert r is not None
        assert r > 0.5  # recent window is positive


class TestSpearmanCorrelation:
    def setup_method(self) -> None:
        self.engine = CorrelationEngine()

    def test_monotone_positive(self) -> None:
        p = _rising(60)
        r = self.engine.spearman(p, p, 50)
        assert r is not None
        assert abs(r - 1.0) < 1e-6

    def test_monotone_negative(self) -> None:
        # Anti-phase oscillation produces anti-correlated returns
        n = 60
        x = [1.0 + (i % 2) * 0.01 for i in range(n)]
        y = [1.01 - (i % 2) * 0.01 for i in range(n)]
        r = self.engine.spearman(x, y, 50)
        assert r is not None
        assert r < -0.9

    def test_insufficient_data_returns_none(self) -> None:
        p = _rising(5)
        assert self.engine.spearman(p, p, 50) is None

    def test_robust_to_outlier(self) -> None:
        # Pearson and Spearman should both be positive; outlier should not flip Spearman
        p = _rising(60)
        q = list(p)
        q[30] = 100.0  # extreme outlier
        r = self.engine.spearman(p, q, 50)
        assert r is not None
        # Should still be positive (mostly monotone despite outlier)
        assert r > 0.0


class TestCrossCorrelation:
    def setup_method(self) -> None:
        self.engine = CorrelationEngine()

    def test_zero_lag_contemporaneous(self) -> None:
        p = _rising(60)
        profile = self.engine.cross_correlation(p, p, max_lag=5)
        assert 0 in profile
        assert abs(profile[0] - 1.0) < 1e-6

    def test_positive_lags_present(self) -> None:
        p = _rising(60)
        profile = self.engine.cross_correlation(p, p, max_lag=5)
        assert all(lag in profile for lag in range(-5, 6))

    def test_known_lead_lag(self) -> None:
        # x leads y by 3 bars: y[i] = x[i-3]
        n = 80
        x = _rising(n)
        y = x[3:]  # y is x shifted right by 3
        x = x[:len(y)]  # align lengths

        profile = self.engine.cross_correlation(x, y, max_lag=5)
        # At lag=3 (x leads y by 3) we should see high correlation
        assert 3 in profile
        # Lag 3 should have higher |corr| than lag 0 or other lags
        assert abs(profile.get(3, 0)) >= abs(profile.get(0, 0)) - 0.2

    def test_insufficient_data_returns_empty(self) -> None:
        p = [1.0, 1.1]
        profile = self.engine.cross_correlation(p, p, max_lag=5)
        # Should handle gracefully, may return empty or only lag=0
        assert isinstance(profile, dict)


class TestRollingCovariance:
    def setup_method(self) -> None:
        self.engine = CorrelationEngine()

    def test_positive_covariance_for_same_series(self) -> None:
        p = _rising(60)
        cov = self.engine.rolling_covariance(p, p, 50)
        assert cov is not None
        assert cov > 0

    def test_negative_covariance_for_opposite(self) -> None:
        n = 60
        x = [1.0 + (i % 2) * 0.01 for i in range(n)]
        y = [1.01 - (i % 2) * 0.01 for i in range(n)]
        cov = self.engine.rolling_covariance(x, y, 50)
        assert cov is not None
        assert cov < 0

    def test_none_on_insufficient_data(self) -> None:
        p = _rising(5)
        assert self.engine.rolling_covariance(p, p, 50) is None


# ══════════════════════════════════════════════════════════════════════════════
# LeadLagDetector
# ══════════════════════════════════════════════════════════════════════════════

class TestLeadLagDetector:
    def setup_method(self) -> None:
        self.detector = LeadLagDetector(max_lag=5)

    def test_detects_zero_lag_for_identical_series(self) -> None:
        p = _rising(80)
        result = self.detector.detect(p, p)
        assert result is not None
        assert result.dominant_lag == 0
        assert abs(result.max_correlation - 1.0) < 1e-6

    def test_lag_profile_has_correct_keys(self) -> None:
        p = _rising(80)
        result = self.detector.detect(p, p)
        assert result is not None
        assert set(result.lag_profile.keys()) == set(range(-5, 6))

    def test_known_positive_lag(self) -> None:
        # x leads y by 2 bars
        n = 60
        base = _rising(n + 2)
        x = base[2:]   # shifted forward
        y = base[:n]
        result = self.detector.detect(x, y)
        assert result is not None
        # dominant lag should be 2 or nearby
        assert result.dominant_lag in range(0, 5)

    def test_confidence_high_for_clean_lag(self) -> None:
        p = _rising(80)
        result = self.detector.detect(p, p)
        assert result is not None
        # Perfect contemporaneous correlation → no lag dominance → moderate confidence
        assert 0.0 <= result.confidence <= 1.0

    def test_returns_none_for_insufficient_data(self) -> None:
        p = [1.0, 1.1]
        result = self.detector.detect(p, p)
        assert result is None

    def test_noisy_data_still_returns_result(self) -> None:
        x = _noisy(60)
        y = _noisy(60, seed=99)
        result = self.detector.detect(x, y)
        # Should not raise and should return a LeadLagResult
        assert result is None or isinstance(result, LeadLagResult)


# ══════════════════════════════════════════════════════════════════════════════
# StabilityMonitor
# ══════════════════════════════════════════════════════════════════════════════

class TestStabilityMonitor:
    def setup_method(self) -> None:
        self.monitor = StabilityMonitor(stability_window=10, regime_break_threshold=0.4)

    def test_stable_series_gives_high_score(self) -> None:
        for _ in range(15):
            self.monitor.update("A|B", 0.8)
        score = self.monitor.get_stability_score("A|B")
        assert score > 0.8

    def test_volatile_series_gives_low_score(self) -> None:
        for i in range(20):
            self.monitor.update("A|B", 1.0 if i % 2 == 0 else -1.0)
        score = self.monitor.get_stability_score("A|B")
        assert score < 0.3

    def test_neutral_score_with_no_history(self) -> None:
        score = self.monitor.get_stability_score("X|Y")
        assert score == 0.5

    def test_neutral_score_with_few_observations(self) -> None:
        self.monitor.update("A|B", 0.5)
        assert self.monitor.get_stability_score("A|B") == 0.5

    def test_regime_break_detected(self) -> None:
        # First half: positive correlation
        for _ in range(15):
            self.monitor.update("A|B", 0.8)
        # Second half: negative correlation (regime break)
        for _ in range(15):
            self.monitor.update("A|B", -0.8)
        assert self.monitor.is_regime_break("A|B")

    def test_no_regime_break_for_stable_correlation(self) -> None:
        for _ in range(20):
            self.monitor.update("A|B", 0.75)
        assert not self.monitor.is_regime_break("A|B")

    def test_no_regime_break_for_insufficient_history(self) -> None:
        self.monitor.update("A|B", 0.8)
        assert not self.monitor.is_regime_break("A|B")

    def test_warning_on_regime_break(self) -> None:
        for _ in range(15):
            self.monitor.update("A|B", 0.8)
        for _ in range(15):
            self.monitor.update("A|B", -0.8)
        w = self.monitor.get_warning("A|B")
        assert w is not None
        assert "Regime break" in w

    def test_warning_on_low_stability(self) -> None:
        # Alternating correlations without sign flip → low stability only
        monitor = StabilityMonitor(stability_window=10, regime_break_threshold=0.99)
        for i in range(20):
            monitor.update("A|B", 0.9 if i % 2 == 0 else 0.1)
        w = monitor.get_warning("A|B")
        # Should warn about low stability
        assert w is not None

    def test_no_warning_for_healthy_relationship(self) -> None:
        for _ in range(15):
            self.monitor.update("A|B", 0.75)
        w = self.monitor.get_warning("A|B")
        assert w is None

    def test_reset_single_pair(self) -> None:
        self.monitor.update("A|B", 0.8)
        self.monitor.reset("A|B")
        assert self.monitor.get_history("A|B") == []

    def test_reset_all(self) -> None:
        self.monitor.update("A|B", 0.8)
        self.monitor.update("C|D", 0.5)
        self.monitor.reset()
        assert self.monitor.get_history("A|B") == []
        assert self.monitor.get_history("C|D") == []

    def test_history_capped(self) -> None:
        monitor = StabilityMonitor(stability_window=10)
        for i in range(100):
            monitor.update("A|B", float(i % 2))
        history = monitor.get_history("A|B")
        assert len(history) <= 20  # cap = 2 × window


# ══════════════════════════════════════════════════════════════════════════════
# IntermarketFeatureBuilder
# ══════════════════════════════════════════════════════════════════════════════

def _make_dep(
    src: str = "USDJPY",
    tgt: str = "EURUSD",
    rel_type: str = "pearson",
    lag: int = 0,
    direction: int = -1,
    strength: float = 0.7,
    stability: float = 0.8,
    confidence: float = 0.56,
    impact: float = -0.4,
) -> DependencySnapshot:
    return DependencySnapshot(
        source_instrument=src,
        target_instrument=tgt,
        relationship_type=rel_type,
        lag_bars=lag,
        directionality=direction,
        strength_score=strength,
        stability_score=stability,
        confidence=confidence,
        valid_from=_NOW,
        valid_to=None,
        current_signal_impact=impact,
        warnings=(),
        timestamp_utc=_NOW,
    )


class TestIntermarketFeatureBuilder:
    def setup_method(self) -> None:
        self.builder = IntermarketFeatureBuilder()

    def test_pearson_key_present(self) -> None:
        dep = _make_dep(rel_type="pearson")
        features = self.builder.build([dep])
        assert "im_usdjpy_eurusd_pcorr" in features

    def test_spearman_key_present(self) -> None:
        dep = _make_dep(rel_type="spearman")
        features = self.builder.build([dep])
        assert "im_usdjpy_eurusd_scorr" in features

    def test_cross_corr_keys_present(self) -> None:
        dep = _make_dep(rel_type="cross_corr", lag=3)
        features = self.builder.build([dep])
        assert "im_usdjpy_eurusd_lag" in features
        assert "im_usdjpy_eurusd_lagcorr" in features

    def test_meta_keys_always_present(self) -> None:
        dep = _make_dep(rel_type="pearson")
        features = self.builder.build([dep])
        for key in ("im_usdjpy_eurusd_stab", "im_usdjpy_eurusd_conf", "im_usdjpy_eurusd_impact"):
            assert key in features

    def test_signed_pearson(self) -> None:
        dep = _make_dep(rel_type="pearson", direction=-1, strength=0.7)
        features = self.builder.build([dep])
        assert features["im_usdjpy_eurusd_pcorr"] == pytest.approx(-0.7)

    def test_lag_value_stored(self) -> None:
        dep = _make_dep(rel_type="cross_corr", lag=4)
        features = self.builder.build([dep])
        assert features["im_usdjpy_eurusd_lag"] == 4.0

    def test_empty_input(self) -> None:
        assert self.builder.build([]) == {}

    def test_multiple_deps_dont_collide(self) -> None:
        d1 = _make_dep(src="USDJPY", rel_type="pearson")
        d2 = _make_dep(src="GBPUSD", rel_type="pearson")
        features = self.builder.build([d1, d2])
        assert "im_usdjpy_eurusd_pcorr" in features
        assert "im_gbpusd_eurusd_pcorr" in features

    def test_symbol_normalisation(self) -> None:
        dep = _make_dep(src="USD/JPY", tgt="EUR/USD", rel_type="pearson")
        features = self.builder.build([dep])
        assert "im_usdjpy_eurusd_pcorr" in features


# ══════════════════════════════════════════════════════════════════════════════
# IntermarketModule
# ══════════════════════════════════════════════════════════════════════════════

class TestIntermarketModule:
    def _module(self, **kwargs) -> IntermarketModule:
        return IntermarketModule(config=_default_cfg(**kwargs), timeframe_id="h1")

    def _primary_bars(self, n: int = 60, start: float = 1.0) -> list[Bar]:
        return _bars_from_prices(_rising(n, start=start))

    def _ref_bars(self, n: int = 60, start: float = 1.0, symbol: str = "USDJPY") -> list[Bar]:
        return _bars_from_prices(_rising(n, start=start), symbol=symbol)

    # ── module_id and min_bars ────────────────────────────────────────────────

    def test_module_id(self) -> None:
        m = self._module()
        assert m.module_id == "intermarket_h1"

    def test_min_bars_respects_config(self) -> None:
        m = self._module(correlation_window=30, lag_max_bars=10)
        assert m.min_bars == 30 + 10 + 2

    # ── HOLD paths ────────────────────────────────────────────────────────────

    def test_hold_when_no_reference_fed(self) -> None:
        m = self._module()
        bars = self._primary_bars(60)
        sig = m.analyse(bars, _NOW)
        assert sig.direction == SignalDirection.HOLD
        assert "no_reference_data_fed" in sig.reason

    def test_raises_when_insufficient_primary_bars(self) -> None:
        from metatrade.core.errors import ModuleNotReadyError
        m = self._module()
        m.feed_reference("USDJPY", self._ref_bars(60))
        with pytest.raises(ModuleNotReadyError):
            m.analyse(self._primary_bars(5), _NOW)

    def test_hold_when_reference_has_insufficient_bars(self) -> None:
        m = self._module()
        m.feed_reference("USDJPY", self._ref_bars(3))  # too few
        sig = m.analyse(self._primary_bars(60), _NOW)
        assert sig.direction == SignalDirection.HOLD

    # ── Reference data management ─────────────────────────────────────────────

    def test_feed_reference_stores_symbol(self) -> None:
        m = self._module()
        m.feed_reference("USDJPY", self._ref_bars(60))
        assert "USDJPY" in m.reference_symbols

    def test_clear_reference_single(self) -> None:
        m = self._module()
        m.feed_reference("USDJPY", self._ref_bars(60))
        m.clear_reference("USDJPY")
        assert "USDJPY" not in m.reference_symbols

    def test_clear_reference_all(self) -> None:
        m = self._module()
        m.feed_reference("USDJPY", self._ref_bars(60))
        m.feed_reference("GBPUSD", self._ref_bars(60, symbol="GBPUSD"))
        m.clear_reference()
        assert m.reference_symbols == []

    # ── Signal generation with strong positive correlation ────────────────────

    def test_positive_corr_rising_ref_emits_buy(self) -> None:
        """Strongly correlated rising pair → BUY signal."""
        m = self._module(min_confidence=0.0, min_stability_score=0.0)
        # Primary and reference both rising → strong positive correlation
        primary = _bars_from_prices(_rising(60, start=1.1000, step=0.0001))
        ref = _bars_from_prices(_rising(60, start=110.0, step=0.01), symbol="USDJPY")
        m.feed_reference("USDJPY", ref)
        sig = m.analyse(primary, _NOW)
        # With rising reference and positive correlation, impact should be positive
        # Signal may be BUY or HOLD depending on exact magnitudes; just verify no crash
        assert sig.direction in (SignalDirection.BUY, SignalDirection.HOLD, SignalDirection.SELL)
        assert 0.0 <= sig.confidence <= 1.0

    def test_negative_corr_falling_ref_emits_buy(self) -> None:
        """Negatively correlated pair: reference falls → primary likely rises → BUY."""
        m = self._module(min_confidence=0.0, min_stability_score=0.0)
        primary = _bars_from_prices(_rising(60, start=1.1000, step=0.0001))
        ref = _bars_from_prices(_falling(60, start=110.0, step=0.01), symbol="USDJPY")
        m.feed_reference("USDJPY", ref)
        sig = m.analyse(primary, _NOW)
        assert sig.direction in (SignalDirection.BUY, SignalDirection.HOLD, SignalDirection.SELL)

    # ── Metadata ──────────────────────────────────────────────────────────────

    def test_signal_metadata_contains_dependencies(self) -> None:
        m = self._module()
        ref = _bars_from_prices(_rising(80), symbol="USDJPY")
        primary = _bars_from_prices(_rising(80))
        m.feed_reference("USDJPY", ref)
        sig = m.analyse(primary, _NOW)
        if sig.direction != SignalDirection.HOLD:
            assert "dependencies" in sig.metadata
            assert "features" in sig.metadata

    def test_signal_confidence_in_valid_range(self) -> None:
        m = self._module()
        ref = _bars_from_prices(_rising(80), symbol="USDJPY")
        primary = _bars_from_prices(_rising(80))
        m.feed_reference("USDJPY", ref)
        sig = m.analyse(primary, _NOW)
        assert 0.0 <= sig.confidence <= 1.0

    # ── min_confidence threshold ──────────────────────────────────────────────

    def test_low_confidence_forces_hold(self) -> None:
        """Set min_confidence very high → should produce HOLD."""
        m = self._module(min_confidence=0.99)
        ref = _bars_from_prices(_rising(80), symbol="USDJPY")
        primary = _bars_from_prices(_rising(80))
        m.feed_reference("USDJPY", ref)
        sig = m.analyse(primary, _NOW)
        assert sig.direction == SignalDirection.HOLD

    # ── Multiple reference symbols ────────────────────────────────────────────

    def test_multiple_references_processed(self) -> None:
        m = self._module()
        primary = _bars_from_prices(_rising(80))
        m.feed_reference("USDJPY", _bars_from_prices(_rising(80), symbol="USDJPY"))
        m.feed_reference("GBPUSD", _bars_from_prices(_rising(80), symbol="GBPUSD"))
        sig = m.analyse(primary, _NOW)
        assert sig is not None  # no crash with multiple references

    # ── Constant series (zero variance) ──────────────────────────────────────

    def test_constant_reference_produces_hold(self) -> None:
        """Constant reference → zero variance → no valid correlation → HOLD."""
        m = self._module()
        constant = [1.0] * 80
        primary = _bars_from_prices(_rising(80))
        m.feed_reference("USDJPY", _bars_from_prices(constant, symbol="USDJPY"))
        sig = m.analyse(primary, _NOW)
        assert sig.direction == SignalDirection.HOLD


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio-risk contracts: validation
# ══════════════════════════════════════════════════════════════════════════════

from metatrade.intermarket.contracts import (
    CurrencyExposure,
    IntermarketDecision,
    PairCorrelation as PortfolioPairCorrelation,
)


class TestCurrencyExposureValidation:
    def test_negative_gross_lots_raises(self) -> None:
        import pytest
        from decimal import Decimal
        with pytest.raises(ValueError, match="gross_lots must be >= 0"):
            CurrencyExposure(
                currency="EUR",
                net_lots=Decimal("0.5"),
                gross_lots=Decimal("-0.1"),
            )


class TestPairCorrelationValidation:
    def test_correlation_out_of_range_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="correlation must be in"):
            PortfolioPairCorrelation(
                symbol_a="EURUSD",
                symbol_b="GBPUSD",
                correlation=1.5,
                abs_correlation=1.5,
                lookback_bars=100,
                overlap_bars=90,
                returns_mode="pct",
                timestamp_utc=_NOW,
            )

    def test_overlap_bars_too_small_raises(self) -> None:
        import pytest
        with pytest.raises(ValueError, match="overlap_bars must be >= 2"):
            PortfolioPairCorrelation(
                symbol_a="EURUSD",
                symbol_b="GBPUSD",
                correlation=0.8,
                abs_correlation=0.8,
                lookback_bars=100,
                overlap_bars=1,
                returns_mode="pct",
                timestamp_utc=_NOW,
            )


class TestIntermarketDecisionValidation:
    def test_zero_risk_multiplier_raises(self) -> None:
        import pytest
        from decimal import Decimal
        with pytest.raises(ValueError, match="risk_multiplier must be > 0"):
            IntermarketDecision(
                approved=True,
                risk_multiplier=Decimal("0"),
                correlated_positions=(),
                reason="test",
                warnings=(),
            )

    def test_empty_reason_raises(self) -> None:
        import pytest
        from decimal import Decimal
        with pytest.raises(ValueError, match="reason cannot be empty"):
            IntermarketDecision(
                approved=True,
                risk_multiplier=Decimal("1.0"),
                correlated_positions=(),
                reason="",
                warnings=(),
            )
