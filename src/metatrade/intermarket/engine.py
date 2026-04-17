"""IntermarketEngine — portfolio-risk orchestrator.

Inserted between ConsensusEngine and RiskManager in the processing pipeline:

    ConsensusEngine → IntermarketDecision → RiskManager

Responsibilities
----------------
1. Rolling correlation matrix across enabled symbols.
2. Currency exposure accounting for open positions + candidate trade.
3. Duplicate-risk detection via risk clusters (union-find on strong correlations).
4. Risk multiplier calculation based on correlation level.
5. Veto when currency exposure limits or cluster limits would be exceeded.
6. Optional persistence of decisions and snapshots.

Usage
-----
    engine = IntermarketEngine(config)

    # Feed updated close prices each bar (per symbol)
    engine.update_bars("EURUSD", list_of_closes)
    engine.update_bars("GBPUSD", list_of_closes)

    # Evaluate a candidate trade (call AFTER consensus, BEFORE risk manager)
    decision = engine.evaluate(
        symbol="EURUSD",
        side=OrderSide.BUY,
        lot_size=Decimal("0.10"),
        open_positions=[...],
        timestamp_utc=datetime.now(timezone.utc),
    )
    if not decision.approved:
        # Return vetoed RiskDecision
    else:
        # Pass decision.risk_multiplier to RiskManager for lot-size scaling

Backward compatibility
----------------------
When ``config.enabled = False`` (the default), ``evaluate()`` always returns
an approved decision with ``risk_multiplier = 1.0`` and no warnings.
Existing single-symbol workflows are completely unaffected.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import structlog

from metatrade.core.contracts.position import Position
from metatrade.core.enums import OrderSide
from metatrade.intermarket.clustering import RiskClusterBuilder
from metatrade.intermarket.config import IntermarketConfig
from metatrade.intermarket.contracts import (
    CurrencyCode,
    IntermarketDecision,
    IntermarketSnapshot,
    PairCorrelation,
)
from metatrade.intermarket.correlation import RollingCorrelationService
from metatrade.intermarket.exposure import CurrencyExposureService

log = structlog.get_logger(__name__)

# Sentinel decision returned when the engine is disabled or has no data.
_APPROVED_PASSTHROUGH = IntermarketDecision(
    approved=True,
    risk_multiplier=Decimal("1.0"),
    correlated_positions=(),
    reason="intermarket_disabled",
    warnings=(),
)


class IntermarketEngine:
    """Portfolio-risk orchestrator for cross-pair correlation and exposure control.

    Args:
        config: ``IntermarketConfig`` (reads ``enabled``, ``symbols``,
                ``corr_*``, ``exposure_*``, ``portfolio_*``, ``risk_*``).
        store:  Optional ``IntermarketStore`` for telemetry persistence.
                Pass ``None`` (default) to disable persistence.
    """

    def __init__(
        self,
        config: IntermarketConfig | None = None,
        store: object | None = None,          # IntermarketStore | None
    ) -> None:
        self._config = config or IntermarketConfig()
        self._store = store

        self._corr_service = RollingCorrelationService(self._config)
        self._exposure_service = CurrencyExposureService()
        self._cluster_builder = RiskClusterBuilder(
            strong_threshold=self._config.corr_strong_threshold,
        )

    # ── Public: data feeding ──────────────────────────────────────────────────

    def update_bars(self, symbol: str, closes: list[float]) -> None:
        """Feed updated close-price history for one symbol.

        Should be called each bar for every symbol in ``config.symbols``
        that the runner has data for.  It is safe to call with symbols not
        in ``config.symbols``; the data is stored but ignored in matrix
        computation unless the symbol appears in the config list.

        Args:
            symbol: Instrument identifier.
            closes: Close prices, oldest first, newest last.
        """
        self._corr_service.update(symbol, closes)

    # ── Public: evaluation ────────────────────────────────────────────────────

    def evaluate(
        self,
        symbol: str,
        side: OrderSide,
        lot_size: Decimal,
        open_positions: list[Position],
        timestamp_utc: datetime,
        session_id: str | None = None,
    ) -> IntermarketDecision:
        """Evaluate a candidate trade in the context of the current portfolio.

        Decision logic (in order):
        1. If ``config.enabled = False`` → approve with multiplier 1.0.
        2. Compute rolling correlation matrix for ``config.symbols``.
        3. Compute exposure delta for the candidate trade.
        4. Compute current net exposure from open positions.
        5. If projected net or gross exposure exceeds limits → veto.
        6. Find correlated open positions and determine risk multiplier.
        7. Build risk clusters; check cluster position limit.
        8. Return ``IntermarketDecision``.

        Args:
            symbol:        Instrument to trade.
            side:          BUY or SELL.
            lot_size:      Proposed lot size (base, before multiplier).
            open_positions: Currently open positions across all symbols.
            timestamp_utc: UTC timestamp of the current bar.
            session_id:    Optional session ID for telemetry.

        Returns:
            ``IntermarketDecision`` — caller must check ``approved`` before
            submitting the order.
        """
        if not self._config.enabled:
            return _APPROVED_PASSTHROUGH

        # ── Step 1: Correlation matrix ────────────────────────────────────────
        correlations = self._corr_service.compute_matrix(
            self._config.symbols, timestamp_utc
        )

        # ── Step 2: Risk clusters ─────────────────────────────────────────────
        clusters = self._cluster_builder.build_clusters(
            correlations, self._config.symbols
        )

        # ── Step 3: Exposure delta for candidate ──────────────────────────────
        exposure_delta = self._exposure_service.exposure_delta(
            symbol, side, lot_size
        )

        # ── Step 4: Current net exposure from open positions ──────────────────
        current_net = self._exposure_service.compute_net_exposure(open_positions)
        current_gross = self._exposure_service.compute_gross_exposure(open_positions)

        # ── Step 5: Exposure limit checks ─────────────────────────────────────
        if self._config.risk_veto_if_exposure_limit_exceeded:
            veto = self._check_exposure_limits(
                current_net, current_gross, exposure_delta
            )
            if veto:
                decision = IntermarketDecision(
                    approved=False,
                    risk_multiplier=Decimal("1.0"),
                    correlated_positions=(),
                    reason=veto,
                    warnings=(),
                    exposure_delta_by_ccy=dict(exposure_delta),
                )
                self._persist(symbol, side, decision, snapshot=None, session_id=session_id)
                return decision

        # ── Step 6: Correlated open positions & risk multiplier ───────────────
        correlated_syms = self._cluster_builder.correlated_symbols(
            symbol,
            correlations,
            threshold=self._config.corr_medium_threshold,
        )
        correlated_open = [
            sym for sym, _ in correlated_syms
            if any(p.is_open and p.symbol == sym for p in open_positions)
        ]

        risk_multiplier, mult_reason, warnings = self._compute_risk_adjustment(
            symbol, correlations, correlated_open
        )

        # ── Step 7: Cluster position limit ────────────────────────────────────
        if self._config.risk_veto_same_idea_same_dir and correlated_open:
            open_syms = [p.symbol for p in open_positions if p.is_open]
            positions_in_cluster = self._cluster_builder.count_open_positions_in_cluster(
                symbol, clusters, open_syms
            )
            max_per_cluster = self._config.portfolio_max_corr_positions_per_cluster
            if positions_in_cluster >= max_per_cluster:
                cluster_members = self._cluster_builder.get_cluster_members(
                    symbol, clusters
                )
                reason = (
                    f"Cluster position limit: {positions_in_cluster} open positions "
                    f"in cluster {sorted(cluster_members)} >= max {max_per_cluster}"
                )
                decision = IntermarketDecision(
                    approved=False,
                    risk_multiplier=Decimal("1.0"),
                    correlated_positions=tuple(correlated_open),
                    reason=reason,
                    warnings=tuple(warnings),
                    exposure_delta_by_ccy=dict(exposure_delta),
                )
                self._persist(symbol, side, decision, snapshot=None, session_id=session_id)
                return decision

        # ── Step 8: Build snapshot and return ────────────────────────────────
        snapshot = self._build_snapshot(
            correlations, current_net, current_gross, clusters, timestamp_utc
        )

        decision = IntermarketDecision(
            approved=True,
            risk_multiplier=risk_multiplier,
            correlated_positions=tuple(correlated_open),
            reason=mult_reason,
            warnings=tuple(warnings),
            exposure_delta_by_ccy=dict(exposure_delta),
            snapshot=snapshot,
        )

        if risk_multiplier < Decimal("1.0"):
            log.info(
                "intermarket_risk_reduced",
                symbol=symbol,
                side=side.value,
                multiplier=str(risk_multiplier),
                reason=mult_reason,
                correlated=correlated_open,
            )

        self._persist(symbol, side, decision, snapshot=snapshot, session_id=session_id)
        return decision

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_exposure_limits(
        self,
        current_net: dict[CurrencyCode, Decimal],
        current_gross: dict[CurrencyCode, Decimal],
        delta: dict[CurrencyCode, Decimal],
    ) -> str | None:
        """Return a veto reason string if exposure limits are exceeded, else None."""
        max_net = Decimal(str(self._config.exposure_max_net))
        max_gross = Decimal(str(self._config.exposure_max_gross))

        for ccy, delta_lots in delta.items():
            projected_net = current_net.get(ccy, Decimal("0")) + delta_lots
            if abs(projected_net) > max_net:
                return (
                    f"Currency exposure limit exceeded: {ccy} projected net "
                    f"{projected_net:+.2f} lots (max ±{max_net})"
                )

        # Gross exposure check (sum absolute values)
        combined_gross: dict[str, Decimal] = dict(current_gross)
        for ccy, delta_lots in delta.items():
            combined_gross[ccy] = combined_gross.get(ccy, Decimal("0")) + abs(delta_lots)

        for ccy, gross in combined_gross.items():
            if gross > max_gross:
                return (
                    f"Currency gross exposure limit exceeded: {ccy} projected gross "
                    f"{gross:.2f} lots (max {max_gross})"
                )

        return None

    def _compute_risk_adjustment(
        self,
        symbol: str,
        correlations: dict[tuple[str, str], PairCorrelation],
        correlated_open: list[str],
    ) -> tuple[Decimal, str, list[str]]:
        """Return (risk_multiplier, reason, warnings) based on correlation level.

        Applies the most conservative multiplier across all correlated open
        positions:
        - ``|corr| >= strong_threshold``  → ``scale_down_high_corr``
        - ``|corr| >= medium_threshold``  → ``scale_down_medium_corr``
        - otherwise                       → 1.0

        Args:
            symbol:         Candidate symbol.
            correlations:   Current correlation matrix.
            correlated_open: Symbols of correlated open positions.

        Returns:
            (multiplier, reason_str, list_of_warnings)
        """
        if not correlated_open:
            return Decimal("1.0"), "no_correlated_positions", []

        strong_thresh = self._config.corr_strong_threshold
        medium_thresh = self._config.corr_medium_threshold
        scale_high = Decimal(str(self._config.risk_scale_down_high_corr))
        scale_medium = Decimal(str(self._config.risk_scale_down_medium_corr))

        min_multiplier = Decimal("1.0")
        reasons: list[str] = []
        warnings: list[str] = []

        for open_sym in correlated_open:
            key_ab = (min(symbol, open_sym), max(symbol, open_sym))
            pc = correlations.get(key_ab)
            if pc is None:
                continue

            if pc.abs_correlation >= strong_thresh:
                min_multiplier = min(min_multiplier, scale_high)
                reasons.append(
                    f"strong_corr({open_sym},{pc.correlation:.2f})"
                )
            elif pc.abs_correlation >= medium_thresh:
                min_multiplier = min(min_multiplier, scale_medium)
                warnings.append(
                    f"medium_corr({open_sym},{pc.correlation:.2f}) → size×{scale_medium}"
                )

        if reasons:
            reason = "risk_reduced: " + "; ".join(reasons)
        elif warnings:
            reason = "risk_reduced: " + "; ".join(warnings)
        else:
            reason = "correlated_but_below_threshold"

        return min_multiplier, reason, warnings

    def _build_snapshot(
        self,
        correlations: dict[tuple[str, str], PairCorrelation],
        current_net: dict[CurrencyCode, Decimal],
        current_gross: dict[CurrencyCode, Decimal],
        clusters: list[frozenset[str]],
        timestamp_utc: datetime,
    ) -> IntermarketSnapshot:
        """Assemble an ``IntermarketSnapshot`` from current engine state."""
        exposure_list = self._exposure_service.to_currency_exposure_list(
            current_net, current_gross
        )
        return IntermarketSnapshot(
            correlations=tuple(correlations.values()),
            exposures=exposure_list,
            risk_clusters=tuple(clusters),
            timestamp_utc=timestamp_utc,
        )

    def _persist(
        self,
        symbol: str,
        side: OrderSide,
        decision: IntermarketDecision,
        snapshot: IntermarketSnapshot | None,
        session_id: str | None,
    ) -> None:
        """Persist decision and optional snapshot; silently skip if no store."""
        if self._store is None:
            return
        try:
            from metatrade.intermarket.persistence import IntermarketStore
            if not isinstance(self._store, IntermarketStore):
                return
            self._store.record_decision(
                symbol=symbol,
                side=side.value,
                decision=decision,
                session_id=session_id,
            )
            if snapshot is not None:
                self._store.record_correlation_snapshot(snapshot, session_id=session_id)
                self._store.record_exposure_snapshot(snapshot, session_id=session_id)
        except Exception as exc:
            log.warning("intermarket_persist_failed", error=str(exc))
