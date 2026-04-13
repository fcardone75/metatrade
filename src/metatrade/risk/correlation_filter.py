"""Correlation filter — detects over-correlated instruments.

Computes the Pearson correlation of log-returns between two bar series.
Designed as a standalone utility for future multi-symbol runners; not
currently wired into the single-symbol pipeline.

Usage:
    cf = CorrelationFilter(max_correlation=0.70, window_bars=20)
    if cf.is_correlated(bars_eurusd, bars_gbpusd):
        # skip second entry — instruments moving together
        ...
    matrix = cf.correlation_matrix({"EURUSD": bars_eu, "GBPUSD": bars_gb})
"""

from __future__ import annotations

import math

from metatrade.core.contracts.market import Bar


class CorrelationFilter:
    """Pearson correlation filter for multi-symbol position management.

    Args:
        max_correlation: Absolute Pearson correlation threshold above which two
                         instruments are considered correlated (default 0.70).
        window_bars:     Number of most-recent bars to use for the computation
                         (default 20).
    """

    def __init__(
        self,
        max_correlation: float = 0.70,
        window_bars: int = 20,
    ) -> None:
        if not 0.0 < max_correlation <= 1.0:
            raise ValueError(
                f"max_correlation must be in (0.0, 1.0], got {max_correlation}"
            )
        if window_bars < 2:
            raise ValueError(f"window_bars must be >= 2, got {window_bars}")
        self._max_corr = max_correlation
        self._window = window_bars

    # ── Public API ────────────────────────────────────────────────────────────

    def pearson_correlation(
        self,
        returns_a: list[float],
        returns_b: list[float],
    ) -> float:
        """Compute Pearson correlation between two return series.

        Args:
            returns_a: Log-return series A (equal length to returns_b).
            returns_b: Log-return series B.

        Returns:
            Pearson r ∈ [-1.0, 1.0].  Returns 0.0 when series are constant
            (std = 0) or have fewer than 2 elements.
        """
        n = min(len(returns_a), len(returns_b))
        if n < 2:
            return 0.0

        a = returns_a[-n:]
        b = returns_b[-n:]

        mean_a = sum(a) / n
        mean_b = sum(b) / n

        cov  = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
        var_a = sum((a[i] - mean_a) ** 2 for i in range(n))
        var_b = sum((b[i] - mean_b) ** 2 for i in range(n))

        denom = math.sqrt(var_a * var_b)
        if denom == 0.0:
            return 0.0
        return cov / denom

    def is_correlated(
        self,
        bars_a: list[Bar],
        bars_b: list[Bar],
    ) -> bool:
        """Return True if the two instruments' returns are highly correlated.

        Args:
            bars_a: Bar series for instrument A (oldest first).
            bars_b: Bar series for instrument B (oldest first).

        Returns:
            True when |Pearson r| >= max_correlation over the most recent
            window_bars bars.
        """
        returns_a = self._log_returns(bars_a)
        returns_b = self._log_returns(bars_b)
        r = self.pearson_correlation(returns_a, returns_b)
        return abs(r) >= self._max_corr

    def correlation_matrix(
        self,
        bars_dict: dict[str, list[Bar]],
    ) -> dict[tuple[str, str], float]:
        """Compute pairwise Pearson correlations for all symbol pairs.

        Args:
            bars_dict: Mapping of symbol → bar series.

        Returns:
            Dict of {(symbol_a, symbol_b): pearson_r} for each unique pair.
        """
        symbols = list(bars_dict.keys())
        result: dict[tuple[str, str], float] = {}
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i + 1 :]:
                ra = self._log_returns(bars_dict[sym_a])
                rb = self._log_returns(bars_dict[sym_b])
                result[(sym_a, sym_b)] = self.pearson_correlation(ra, rb)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_returns(self, bars: list[Bar]) -> list[float]:
        """Compute log-returns from the most recent window_bars closes."""
        window = bars[-(self._window + 1):]
        returns: list[float] = []
        for i in range(1, len(window)):
            p0 = float(window[i - 1].close)
            p1 = float(window[i].close)
            if p0 > 0 and p1 > 0:
                returns.append(math.log(p1 / p0))
            else:
                returns.append(0.0)
        return returns
