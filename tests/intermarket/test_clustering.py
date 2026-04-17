"""Tests for RiskClusterBuilder.

Coverage targets
----------------
- Single symbol → singleton cluster.
- Two strongly correlated symbols → merged cluster.
- Three symbols: A-B strong, B-C strong → A, B, C all in same cluster (transitivity).
- Strong threshold boundary: exactly at threshold → merged; just below → separate.
- Clusters cover all input symbols.
- find_cluster_for: found and not found cases.
- get_cluster_members: found → full cluster; not found → singleton.
- correlated_symbols: returns correct pairs above threshold, sorted by |corr|.
- count_open_positions_in_cluster: counts correctly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from metatrade.intermarket.clustering import RiskClusterBuilder
from metatrade.intermarket.contracts import PairCorrelation

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pc(sym_a: str, sym_b: str, corr: float) -> PairCorrelation:
    return PairCorrelation(
        symbol_a=sym_a,
        symbol_b=sym_b,
        correlation=corr,
        abs_correlation=abs(corr),
        lookback_bars=50,
        overlap_bars=50,
        returns_mode="pct",
        timestamp_utc=_NOW,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RiskClusterBuilder.build_clusters
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildClusters:
    def setup_method(self) -> None:
        self.builder = RiskClusterBuilder(strong_threshold=0.80)

    def test_single_symbol_produces_singleton(self) -> None:
        clusters = self.builder.build_clusters({}, ["EURUSD"])
        assert len(clusters) == 1
        assert frozenset({"EURUSD"}) in clusters

    def test_no_strong_corr_produces_all_singletons(self) -> None:
        corrs = {("EURUSD", "GBPUSD"): _pc("EURUSD", "GBPUSD", 0.50)}
        clusters = self.builder.build_clusters(corrs, ["EURUSD", "GBPUSD"])
        assert len(clusters) == 2
        assert frozenset({"EURUSD"}) in clusters
        assert frozenset({"GBPUSD"}) in clusters

    def test_strong_corr_merges_two_symbols(self) -> None:
        corrs = {("EURUSD", "GBPUSD"): _pc("EURUSD", "GBPUSD", 0.90)}
        clusters = self.builder.build_clusters(corrs, ["EURUSD", "GBPUSD"])
        assert len(clusters) == 1
        assert frozenset({"EURUSD", "GBPUSD"}) in clusters

    def test_negative_strong_corr_also_merges(self) -> None:
        # abs(-0.90) >= 0.80 → merged into same risk cluster
        corrs = {("EURUSD", "USDCHF"): _pc("EURUSD", "USDCHF", -0.90)}
        clusters = self.builder.build_clusters(corrs, ["EURUSD", "USDCHF"])
        assert len(clusters) == 1

    def test_transitivity_merges_chain(self) -> None:
        # A-B strong, B-C strong → A, B, C in one cluster
        syms = ["A", "B", "C"]
        corrs = {
            ("A", "B"): _pc("A", "B", 0.90),
            ("B", "C"): _pc("B", "C", 0.85),
        }
        clusters = self.builder.build_clusters(corrs, syms)
        assert len(clusters) == 1
        assert frozenset({"A", "B", "C"}) in clusters

    def test_threshold_boundary_at_exact_value(self) -> None:
        corrs = {("EURUSD", "GBPUSD"): _pc("EURUSD", "GBPUSD", 0.80)}
        clusters = self.builder.build_clusters(corrs, ["EURUSD", "GBPUSD"])
        assert len(clusters) == 1

    def test_threshold_boundary_just_below(self) -> None:
        corrs = {("EURUSD", "GBPUSD"): _pc("EURUSD", "GBPUSD", 0.799)}
        clusters = self.builder.build_clusters(corrs, ["EURUSD", "GBPUSD"])
        assert len(clusters) == 2

    def test_all_symbols_in_some_cluster(self) -> None:
        syms = ["A", "B", "C", "D"]
        corrs = {("A", "B"): _pc("A", "B", 0.90)}
        clusters = self.builder.build_clusters(corrs, syms)
        all_in_clusters = set()
        for c in clusters:
            all_in_clusters |= c
        assert all_in_clusters == set(syms)

    def test_empty_symbols_produces_empty_clusters(self) -> None:
        clusters = self.builder.build_clusters({}, [])
        assert clusters == []

    def test_three_isolated_symbols(self) -> None:
        corrs = {
            ("A", "B"): _pc("A", "B", 0.50),
            ("A", "C"): _pc("A", "C", 0.30),
        }
        clusters = self.builder.build_clusters(corrs, ["A", "B", "C"])
        assert len(clusters) == 3


# ══════════════════════════════════════════════════════════════════════════════
# find_cluster_for / get_cluster_members
# ══════════════════════════════════════════════════════════════════════════════

class TestFindCluster:
    def setup_method(self) -> None:
        self.builder = RiskClusterBuilder(strong_threshold=0.80)

    def test_find_existing_symbol(self) -> None:
        corrs = {("A", "B"): _pc("A", "B", 0.90)}
        clusters = self.builder.build_clusters(corrs, ["A", "B", "C"])
        cluster = self.builder.find_cluster_for("A", clusters)
        assert cluster is not None
        assert "A" in cluster
        assert "B" in cluster

    def test_find_missing_symbol_returns_none(self) -> None:
        clusters = self.builder.build_clusters({}, ["A", "B"])
        cluster = self.builder.find_cluster_for("UNKNOWN", clusters)
        assert cluster is None

    def test_get_cluster_members_returns_full_cluster(self) -> None:
        corrs = {("A", "B"): _pc("A", "B", 0.90)}
        clusters = self.builder.build_clusters(corrs, ["A", "B", "C"])
        members = self.builder.get_cluster_members("A", clusters)
        assert "A" in members
        assert "B" in members

    def test_get_cluster_members_unknown_returns_singleton(self) -> None:
        clusters = self.builder.build_clusters({}, ["A"])
        members = self.builder.get_cluster_members("UNKNOWN", clusters)
        assert members == frozenset({"UNKNOWN"})


# ══════════════════════════════════════════════════════════════════════════════
# correlated_symbols
# ══════════════════════════════════════════════════════════════════════════════

class TestCorrelatedSymbols:
    def setup_method(self) -> None:
        self.builder = RiskClusterBuilder(strong_threshold=0.80)

    def test_returns_correlated_pairs_above_threshold(self) -> None:
        corrs = {
            ("EURUSD", "GBPUSD"): _pc("EURUSD", "GBPUSD", 0.85),
            ("EURUSD", "USDCHF"): _pc("EURUSD", "USDCHF", 0.50),
        }
        result = self.builder.correlated_symbols("EURUSD", corrs, threshold=0.80)
        syms = [s for s, _ in result]
        assert "GBPUSD" in syms
        assert "USDCHF" not in syms

    def test_handles_reverse_key_order(self) -> None:
        # Key is (GBPUSD, USDCHF), querying for USDCHF
        corrs = {("GBPUSD", "USDCHF"): _pc("GBPUSD", "USDCHF", 0.90)}
        result = self.builder.correlated_symbols("USDCHF", corrs, threshold=0.80)
        syms = [s for s, _ in result]
        assert "GBPUSD" in syms

    def test_sorted_by_descending_abs_correlation(self) -> None:
        corrs = {
            ("A", "B"): _pc("A", "B", 0.90),
            ("A", "C"): _pc("A", "C", 0.85),
        }
        result = self.builder.correlated_symbols("A", corrs, threshold=0.80)
        abs_corrs = [abs(c) for _, c in result]
        assert abs_corrs == sorted(abs_corrs, reverse=True)

    def test_empty_when_no_corrs_above_threshold(self) -> None:
        corrs = {("A", "B"): _pc("A", "B", 0.30)}
        result = self.builder.correlated_symbols("A", corrs, threshold=0.80)
        assert result == []

    def test_correlation_value_is_signed(self) -> None:
        corrs = {("A", "B"): _pc("A", "B", -0.90)}
        result = self.builder.correlated_symbols("A", corrs, threshold=0.80)
        assert len(result) == 1
        _, corr = result[0]
        assert corr == pytest.approx(-0.90)


# ══════════════════════════════════════════════════════════════════════════════
# count_open_positions_in_cluster
# ══════════════════════════════════════════════════════════════════════════════

class TestCountOpenPositionsInCluster:
    def setup_method(self) -> None:
        self.builder = RiskClusterBuilder(strong_threshold=0.80)

    def test_counts_positions_in_same_cluster(self) -> None:
        corrs = {("EURUSD", "GBPUSD"): _pc("EURUSD", "GBPUSD", 0.90)}
        clusters = self.builder.build_clusters(corrs, ["EURUSD", "GBPUSD", "USDJPY"])
        count = self.builder.count_open_positions_in_cluster(
            "EURUSD", clusters, ["EURUSD", "GBPUSD"]
        )
        # Both EURUSD and GBPUSD are in the same cluster
        assert count == 2

    def test_does_not_count_symbols_in_other_clusters(self) -> None:
        corrs = {("EURUSD", "GBPUSD"): _pc("EURUSD", "GBPUSD", 0.90)}
        clusters = self.builder.build_clusters(corrs, ["EURUSD", "GBPUSD", "USDJPY"])
        count = self.builder.count_open_positions_in_cluster(
            "EURUSD", clusters, ["USDJPY"]
        )
        assert count == 0

    def test_zero_open_positions(self) -> None:
        corrs = {("EURUSD", "GBPUSD"): _pc("EURUSD", "GBPUSD", 0.90)}
        clusters = self.builder.build_clusters(corrs, ["EURUSD", "GBPUSD"])
        count = self.builder.count_open_positions_in_cluster(
            "EURUSD", clusters, []
        )
        assert count == 0
