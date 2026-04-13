"""Rolling correlation engine for intermarket analysis.

All correlations are computed on **returns** (percent change), not price
levels, to ensure stationarity and avoid spurious correlations from
co-trending pairs.

Supported metrics:
  - Pearson:    standard linear correlation of returns
  - Spearman:   rank-based correlation (robust to outliers)
  - Cross-corr: Pearson at multiple lags to detect lead-lag relationships
  - Covariance: rolling covariance of returns (unscaled)
"""

from __future__ import annotations

import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_returns(prices: list[float]) -> np.ndarray:
    """Convert a price series to percent returns.

    Returns an array of length len(prices)-1.
    Zero-price entries produce a 0.0 return to avoid division by zero.
    """
    arr = np.array(prices, dtype=float)
    prev = arr[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(prev != 0.0, (arr[1:] - prev) / prev, 0.0)
    return returns


def _rank(arr: np.ndarray) -> np.ndarray:
    """Convert values to ranks (1-based) with average-rank tie handling.

    Tied values receive the arithmetic mean of the ranks they would
    occupy if they were distinct, matching the standard Spearman formula.
    """
    n = len(arr)
    order = np.argsort(arr, kind="stable")
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        # Find the run of elements tied with arr[order[i]]
        j = i + 1
        while j < n and arr[order[j]] == arr[order[i]]:
            j += 1
        # All tied elements share the average of ranks i+1 … j
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j

    return ranks


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation with guard for zero-variance inputs."""
    if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return None
    return float(np.corrcoef(x, y)[0, 1])


# ── Public engine ──────────────────────────────────────────────────────────────

class CorrelationEngine:
    """Computes rolling statistical relationships between two price series."""

    # ── Pearson ───────────────────────────────────────────────────────────────

    def pearson(
        self,
        prices_x: list[float],
        prices_y: list[float],
        window: int,
    ) -> float | None:
        """Rolling Pearson correlation of returns over the last ``window`` bars.

        Returns None if insufficient data or zero variance.
        """
        rx = _to_returns(prices_x)
        ry = _to_returns(prices_y)
        n = min(len(rx), len(ry))
        if n < window:
            return None
        return _safe_corrcoef(rx[-window:], ry[-window:])

    # ── Spearman ──────────────────────────────────────────────────────────────

    def spearman(
        self,
        prices_x: list[float],
        prices_y: list[float],
        window: int,
    ) -> float | None:
        """Rolling Spearman rank correlation of returns.

        Robust to outliers; useful for detecting monotonic relationships
        that may not be linear.
        """
        rx = _to_returns(prices_x)
        ry = _to_returns(prices_y)
        n = min(len(rx), len(ry))
        if n < window:
            return None
        x_w = rx[-window:]
        y_w = ry[-window:]
        return _safe_corrcoef(_rank(x_w), _rank(y_w))

    # ── Cross-correlation ─────────────────────────────────────────────────────

    def cross_correlation(
        self,
        prices_x: list[float],
        prices_y: list[float],
        max_lag: int,
    ) -> dict[int, float]:
        """Pearson correlation between x and y at each integer lag.

        Convention:
          lag > 0  →  x leads y  (x at t, y at t+lag)
          lag = 0  →  contemporaneous
          lag < 0  →  y leads x  (y at t, x at t-lag)

        Returns a dict mapping lag → correlation.
        Lags with insufficient data are omitted.
        """
        rx = _to_returns(prices_x)
        ry = _to_returns(prices_y)
        n = min(len(rx), len(ry))
        # Use only the most recent portion to keep computation windowed
        use = min(n, max(2 * max_lag + 10, 30))
        rx_w = rx[-use:]
        ry_w = ry[-use:]

        result: dict[int, float] = {}
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr = _safe_corrcoef(rx_w, ry_w)
            elif lag > 0:
                if use - lag < 5:
                    continue
                corr = _safe_corrcoef(rx_w[:-lag], ry_w[lag:])
            else:  # lag < 0
                abs_lag = -lag
                if use - abs_lag < 5:
                    continue
                corr = _safe_corrcoef(rx_w[abs_lag:], ry_w[:-abs_lag])

            if corr is not None and not np.isnan(corr):
                result[lag] = corr

        return result

    # ── Covariance ────────────────────────────────────────────────────────────

    def rolling_covariance(
        self,
        prices_x: list[float],
        prices_y: list[float],
        window: int,
    ) -> float | None:
        """Rolling covariance of returns over the last ``window`` bars."""
        rx = _to_returns(prices_x)
        ry = _to_returns(prices_y)
        n = min(len(rx), len(ry))
        if n < window:
            return None
        x_w = rx[-window:]
        y_w = ry[-window:]
        if len(x_w) < 2:
            return None
        return float(np.cov(x_w, y_w, ddof=1)[0, 1])
