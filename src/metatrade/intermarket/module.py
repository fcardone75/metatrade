"""Intermarket analysis module — ITechnicalModule bridge.

This module integrates all intermarket sub-components and exposes
the standard ITechnicalModule interface so it can be plugged into
the consensus engine alongside EMA, RSI, MACD, and ML modules.

Key design decisions:
  - Primary bars are passed via analyse() as usual.
  - Reference bars are fed via feed_reference(), decoupled from the
    main bar feed (they may come from different timeframes or symbols).
  - The module returns HOLD if no reference data has been fed or if
    confidence is too low.
  - Each call to analyse() produces a full set of DependencySnapshots,
    updates the StabilityMonitor, and emits a synthesised AnalysisSignal.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.interface import ITechnicalModule

from metatrade.intermarket.config import IntermarketConfig
from metatrade.intermarket.contracts import DependencySnapshot, IntermarketAnalysis
from metatrade.intermarket.correlation import CorrelationEngine
from metatrade.intermarket.feature_builder import IntermarketFeatureBuilder
from metatrade.intermarket.lead_lag import LeadLagDetector
from metatrade.intermarket.stability import StabilityMonitor

_VERSION = ModuleVersion(1, 0, 0)


class IntermarketModule(ITechnicalModule):
    """Cross-currency correlation and lead-lag analysis module.

    Feeds on the primary symbol bars (via analyse()) plus one or more
    reference symbol bars (via feed_reference()), and emits a signal
    that represents the net directional bias implied by cross-instrument
    statistical dependencies.

    Args:
        config:       IntermarketConfig.  Created with defaults if None.
        timeframe_id: Suffix appended to module_id (default "h1").
    """

    def __init__(
        self,
        config: IntermarketConfig | None = None,
        timeframe_id: str = "h1",
    ) -> None:
        self._config = config or IntermarketConfig()
        self._timeframe_id = timeframe_id
        self._reference_bars: dict[str, list[Bar]] = {}

        self._engine = CorrelationEngine()
        self._lead_lag = LeadLagDetector(max_lag=self._config.lag_max_bars)
        self._stability = StabilityMonitor(
            stability_window=self._config.stability_window,
            regime_break_threshold=self._config.regime_break_threshold,
        )
        self._features = IntermarketFeatureBuilder()

    # ── ITechnicalModule ───────────────────────────────────────────────────────

    @property
    def module_id(self) -> str:
        return f"intermarket_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        return self._config.correlation_window + self._config.lag_max_bars + 2

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        """Run the full intermarket analysis pipeline.

        Steps:
          1. Validate primary bars.
          2. For each reference symbol: compute Pearson, Spearman, cross-corr.
          3. Update stability monitor.
          4. Synthesise a directional signal from weighted dependency impacts.
          5. Return AnalysisSignal with ML features in metadata.
        """
        self._require_min_bars(bars)

        if not self._reference_bars:
            return self._hold(timestamp_utc, "no_reference_data_fed")

        primary_closes = [float(b.close) for b in bars]
        dependencies: list[DependencySnapshot] = []

        cfg = self._config
        primary_sym = cfg.primary_symbol

        for ref_sym, ref_bars in self._reference_bars.items():
            if len(ref_bars) < cfg.correlation_window + 1:
                continue

            ref_closes = [float(b.close) for b in ref_bars]
            valid_from = ref_bars[max(0, len(ref_bars) - cfg.correlation_window)].timestamp_utc

            # ── Pearson ───────────────────────────────────────────────────────
            pearson = self._engine.pearson(primary_closes, ref_closes, cfg.correlation_window)
            if pearson is not None and not np.isnan(pearson):
                pair_key = f"{primary_sym}|{ref_sym}|pearson"
                self._stability.update(pair_key, pearson)
                stability = self._stability.get_stability_score(pair_key)
                strength = abs(pearson)

                if strength >= cfg.min_correlation_strength:
                    directionality = int(np.sign(pearson))
                    confidence = strength * stability
                    impact = self._signal_impact(ref_closes, directionality, confidence)
                    warnings = self._collect_warnings(pair_key)
                    dependencies.append(DependencySnapshot(
                        source_instrument=ref_sym,
                        target_instrument=primary_sym,
                        relationship_type="pearson",
                        lag_bars=0,
                        directionality=directionality,
                        strength_score=round(strength, 4),
                        stability_score=round(stability, 4),
                        confidence=round(confidence, 4),
                        valid_from=valid_from,
                        valid_to=None,
                        current_signal_impact=round(impact, 4),
                        warnings=tuple(warnings),
                        timestamp_utc=timestamp_utc,
                    ))

            # ── Spearman ──────────────────────────────────────────────────────
            spearman = self._engine.spearman(primary_closes, ref_closes, cfg.correlation_window)
            if spearman is not None and not np.isnan(spearman):
                pair_key = f"{primary_sym}|{ref_sym}|spearman"
                self._stability.update(pair_key, spearman)
                stability = self._stability.get_stability_score(pair_key)
                strength = abs(spearman)

                if strength >= cfg.min_correlation_strength:
                    directionality = int(np.sign(spearman))
                    confidence = strength * stability
                    impact = self._signal_impact(ref_closes, directionality, confidence)
                    warnings = self._collect_warnings(pair_key)
                    dependencies.append(DependencySnapshot(
                        source_instrument=ref_sym,
                        target_instrument=primary_sym,
                        relationship_type="spearman",
                        lag_bars=0,
                        directionality=directionality,
                        strength_score=round(strength, 4),
                        stability_score=round(stability, 4),
                        confidence=round(confidence, 4),
                        valid_from=valid_from,
                        valid_to=None,
                        current_signal_impact=round(impact, 4),
                        warnings=tuple(warnings),
                        timestamp_utc=timestamp_utc,
                    ))

            # ── Cross-correlation / lead-lag ───────────────────────────────────
            ll = self._lead_lag.detect(primary_closes, ref_closes)
            if ll is not None and abs(ll.max_correlation) >= cfg.min_correlation_strength:
                pair_key = f"{primary_sym}|{ref_sym}|cross_corr"
                self._stability.update(pair_key, ll.max_correlation)
                stability = self._stability.get_stability_score(pair_key)
                strength = abs(ll.max_correlation)
                directionality = int(np.sign(ll.max_correlation))
                confidence = strength * stability * ll.confidence
                impact = self._lagged_impact(ref_closes, ll.dominant_lag, directionality, confidence)
                warnings = self._collect_warnings(pair_key)

                dependencies.append(DependencySnapshot(
                    source_instrument=ref_sym,
                    target_instrument=primary_sym,
                    relationship_type="cross_corr",
                    lag_bars=ll.dominant_lag,
                    directionality=directionality,
                    strength_score=round(strength, 4),
                    stability_score=round(stability, 4),
                    confidence=round(confidence, 4),
                    valid_from=valid_from,
                    valid_to=None,
                    current_signal_impact=round(impact, 4),
                    warnings=tuple(warnings),
                    timestamp_utc=timestamp_utc,
                ))

        if not dependencies:
            return self._hold(timestamp_utc, "no_significant_dependencies")

        analysis = self._synthesise(dependencies, primary_sym, timestamp_utc)
        features = self._features.build(dependencies)

        direction_map = {1: SignalDirection.BUY, -1: SignalDirection.SELL, 0: SignalDirection.HOLD}
        direction = direction_map[analysis.signal_direction]

        if analysis.overall_confidence < cfg.min_confidence:
            direction = SignalDirection.HOLD
            reason = f"Low intermarket confidence ({analysis.overall_confidence:.2f})"
        else:
            reason = (
                f"Intermarket {direction.value}: "
                f"{len(dependencies)} deps, conf={analysis.overall_confidence:.2f}"
            )

        return AnalysisSignal(
            direction=direction,
            confidence=round(max(0.5, analysis.overall_confidence), 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "n_dependencies": len(dependencies),
                "overall_confidence": analysis.overall_confidence,
                "warnings": analysis.warnings,
                "features": features,
                "dependencies": [
                    {
                        "source": d.source_instrument,
                        "type": d.relationship_type,
                        "lag": d.lag_bars,
                        "direction": d.directionality,
                        "confidence": d.confidence,
                    }
                    for d in dependencies
                ],
            },
        )

    # ── Reference data management ──────────────────────────────────────────────

    def feed_reference(self, symbol: str, bars: list[Bar]) -> None:
        """Supply or update the reference bar series for a symbol.

        Args:
            symbol: Instrument name (e.g. "USDJPY").
            bars:   Historical bars, oldest first, newest last.
        """
        self._reference_bars[symbol] = list(bars)

    def clear_reference(self, symbol: str | None = None) -> None:
        """Remove reference data for one symbol or all symbols."""
        if symbol is None:
            self._reference_bars.clear()
        else:
            self._reference_bars.pop(symbol, None)

    @property
    def reference_symbols(self) -> list[str]:
        """Currently loaded reference symbols."""
        return list(self._reference_bars)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _hold(self, timestamp_utc: datetime, reason: str) -> AnalysisSignal:
        return AnalysisSignal(
            direction=SignalDirection.HOLD,
            confidence=0.5,
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
        )

    def _signal_impact(
        self,
        ref_closes: list[float],
        directionality: int,
        confidence: float,
        lookback: int = 5,
    ) -> float:
        """Estimate the current directional impact from recent reference moves.

        The reference instrument's recent return is normalised via tanh
        and multiplied by directionality and confidence.
        """
        if len(ref_closes) < lookback + 1:
            return 0.0
        prev = ref_closes[-(lookback + 1)]
        if prev == 0.0:
            return 0.0
        recent_return = (ref_closes[-1] - prev) / abs(prev)
        # tanh(100 * r): a 1% move → tanh(1) ≈ 0.76 ; 0.1% move → 0.10
        normalised = float(np.tanh(recent_return * 100))
        return float(np.clip(normalised * directionality * confidence, -1.0, 1.0))

    def _lagged_impact(
        self,
        ref_closes: list[float],
        lag_bars: int,
        directionality: int,
        confidence: float,
    ) -> float:
        """Estimate impact using the lagged reference return.

        For a positive lag (source leads target), we look at the reference
        instrument's return `lag_bars` steps ago to predict the primary now.
        """
        if lag_bars <= 0 or len(ref_closes) < lag_bars + 2:
            return self._signal_impact(ref_closes, directionality, confidence)
        prev = ref_closes[-(lag_bars + 1)]
        if prev == 0.0:
            return 0.0
        lagged_return = (ref_closes[-lag_bars] - prev) / abs(prev)
        normalised = float(np.tanh(lagged_return * 100))
        return float(np.clip(normalised * directionality * confidence, -1.0, 1.0))

    def _collect_warnings(self, pair_key: str) -> list[str]:
        """Collect any stability / regime-break warnings for a pair."""
        w = self._stability.get_warning(pair_key)
        return [w] if w is not None else []

    def _synthesise(
        self,
        dependencies: list[DependencySnapshot],
        primary_sym: str,
        timestamp_utc: datetime,
    ) -> IntermarketAnalysis:
        """Aggregate dependency impacts into a single directional signal."""
        cfg = self._config

        # Filter to dependencies that meet the minimum thresholds
        active = [
            d for d in dependencies
            if d.confidence >= cfg.min_confidence
            and d.stability_score >= cfg.min_stability_score
        ]

        all_warnings: list[str] = []
        for d in dependencies:
            all_warnings.extend(d.warnings)

        if not active:
            return IntermarketAnalysis(
                primary_symbol=primary_sym,
                dependencies=dependencies,
                signal_direction=0,
                overall_confidence=0.0,
                warnings=all_warnings,
                timestamp_utc=timestamp_utc,
            )

        # Weighted mean of impact scores (weight = confidence)
        total_weight = sum(d.confidence for d in active)
        weighted_impact = sum(d.current_signal_impact * d.confidence for d in active)
        net_impact = weighted_impact / total_weight if total_weight > 0 else 0.0

        # Map to discrete direction
        if net_impact > 0.2:
            direction = 1
        elif net_impact < -0.2:
            direction = -1
        else:
            direction = 0

        overall_confidence = total_weight / len(active)  # mean confidence

        return IntermarketAnalysis(
            primary_symbol=primary_sym,
            dependencies=dependencies,
            signal_direction=direction,
            overall_confidence=round(overall_confidence, 4),
            features=self._features.build(active),
            warnings=all_warnings,
            timestamp_utc=timestamp_utc,
        )
