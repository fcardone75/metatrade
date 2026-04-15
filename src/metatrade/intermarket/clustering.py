"""Risk cluster builder for the intermarket portfolio-risk layer.

Groups forex pairs into temporary risk clusters based on strong rolling
correlations, using a transparent union-find algorithm.

Design principles
-----------------
- No black-box math: every cluster membership is traceable to a specific
  correlation value that exceeded the strong threshold.
- Clusters are rebuilt each evaluation cycle from the current correlation
  matrix — they are not persisted across calls.
- The cluster for a symbol that has no strong correlations is a singleton
  containing only that symbol.
- ``correlated_symbols()`` exposes all pairs above a configurable threshold,
  enabling both the "same idea" duplicate check and risk scaling.
"""

from __future__ import annotations

from metatrade.intermarket.contracts import PairCorrelation


class RiskClusterBuilder:
    """Builds transparent risk clusters from a correlation matrix.

    A risk cluster is a set of symbols that are all strongly correlated
    with at least one other member (transitive via union-find).  Long
    EURUSD, long GBPUSD, and short USDCHF might all land in the same
    "USD weakness" cluster when correlations are high.

    Args:
        strong_threshold: Minimum |correlation| to merge two symbols into
                          the same cluster.  Defaults to 0.80.
    """

    def __init__(self, strong_threshold: float = 0.80) -> None:
        self._strong_threshold = strong_threshold

    # ── Cluster construction ──────────────────────────────────────────────────

    def build_clusters(
        self,
        correlations: dict[tuple[str, str], PairCorrelation],
        symbols: list[str],
    ) -> list[frozenset[str]]:
        """Build risk clusters from a correlation matrix using union-find.

        Symbols connected by at least one strong-correlation edge are merged
        into the same cluster.  Symbols with no strong edges remain singletons.

        Args:
            correlations: Dict mapping ``(sym_a, sym_b)`` → ``PairCorrelation``.
                          Pairs not present are assumed uncorrelated.
            symbols:      Complete list of symbols to cluster.  Symbols that
                          have no entry in ``correlations`` are included as
                          singletons.

        Returns:
            List of ``frozenset[str]`` clusters.  Every symbol in ``symbols``
            appears in exactly one cluster.
        """
        # Union-Find with path compression
        parent: dict[str, str] = {sym: sym for sym in symbols}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]   # path compression
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Merge strongly correlated pairs
        for (sym_a, sym_b), pc in correlations.items():
            if pc.abs_correlation >= self._strong_threshold:
                if sym_a in parent and sym_b in parent:
                    union(sym_a, sym_b)

        # Collect groups by representative root
        groups: dict[str, set[str]] = {}
        for sym in symbols:
            root = find(sym)
            groups.setdefault(root, set()).add(sym)

        return [frozenset(g) for g in groups.values()]

    # ── Cluster queries ───────────────────────────────────────────────────────

    def find_cluster_for(
        self,
        symbol: str,
        clusters: list[frozenset[str]],
    ) -> frozenset[str] | None:
        """Return the cluster that contains the given symbol, or ``None``.

        Args:
            symbol:   Symbol to look up.
            clusters: Output of ``build_clusters()``.
        """
        for cluster in clusters:
            if symbol in cluster:
                return cluster
        return None

    def get_cluster_members(
        self,
        symbol: str,
        clusters: list[frozenset[str]],
    ) -> frozenset[str]:
        """Return all symbols in the same cluster as the given symbol.

        If the symbol is not found in any cluster, returns a singleton set
        containing only the symbol itself.

        Args:
            symbol:   Symbol to query.
            clusters: Output of ``build_clusters()``.
        """
        cluster = self.find_cluster_for(symbol, clusters)
        return cluster if cluster is not None else frozenset({symbol})

    def correlated_symbols(
        self,
        symbol: str,
        correlations: dict[tuple[str, str], PairCorrelation],
        threshold: float,
    ) -> list[tuple[str, float]]:
        """Return all symbols correlated with ``symbol`` above ``threshold``.

        Searches both ``(symbol, other)`` and ``(other, symbol)`` key forms.

        Args:
            symbol:       Reference symbol.
            correlations: Correlation matrix from ``RollingCorrelationService``.
            threshold:    Minimum |correlation| to include.

        Returns:
            List of ``(other_symbol, signed_correlation)`` tuples, sorted
            by descending |correlation|.
        """
        result: list[tuple[str, float]] = []
        for (sym_a, sym_b), pc in correlations.items():
            if pc.abs_correlation < threshold:
                continue
            if sym_a == symbol:
                result.append((sym_b, pc.correlation))
            elif sym_b == symbol:
                result.append((sym_a, pc.correlation))

        result.sort(key=lambda t: abs(t[1]), reverse=True)
        return result

    def count_open_positions_in_cluster(
        self,
        symbol: str,
        clusters: list[frozenset[str]],
        open_symbols: list[str],
    ) -> int:
        """Count how many of the open position symbols are in the same cluster.

        Args:
            symbol:       Symbol of the candidate trade.
            clusters:     Output of ``build_clusters()``.
            open_symbols: List of symbols with currently open positions.
        """
        cluster = self.get_cluster_members(symbol, clusters)
        return sum(1 for s in open_symbols if s in cluster)
