"""Rolling correlation engine for intermarket analysis.

Two public classes:

``CorrelationEngine``
    Stateless utility: Pearson, Spearman, cross-correlation, covariance.
    Used by ``IntermarketModule`` (the ITechnicalModule bridge).

``RollingCorrelationService``
    Stateful service: maintains per-symbol close-price histories and
    computes ``PairCorrelation`` objects for the portfolio-risk engine.
    Supports both log and pct returns, enforces min-overlap validation,
    and exposes a matrix-computation helper.

All correlations are computed on **returns** (not price levels) to
ensure stationarity and avoid spurious co-trending correlations.
"""

from __future__ import annotations

from datetime import datetime

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


# ── Portfolio-risk layer ───────────────────────────────────────────────────────

def _to_log_returns(prices: list[float]) -> np.ndarray:
    """Convert a price series to log returns: ln(p[i] / p[i-1]).

    Zero or negative prices produce a 0.0 log-return.
    """
    arr = np.array(prices, dtype=float)
    prev = arr[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rets = np.where(
            prev > 0,
            np.log(np.where(prev > 0, arr[1:] / prev, 1.0)),
            0.0,
        )
    return log_rets


class RollingCorrelationService:
    """Maintains per-symbol close-price histories and computes PairCorrelations.

    Design principles
    -----------------
    - State is local per instance — no global mutable state.
    - Returns ``None`` / omits pairs when overlap is below the configured
      minimum; never produces misleading correlations from sparse data.
    - Deterministic: given the same price histories, always produces the
      same correlation values.
    - Supports both ``"log"`` and ``"pct"`` return modes.

    Args:
        config: ``IntermarketConfig`` — reads ``corr_lookback_bars``,
                ``corr_min_overlap_bars``, and ``corr_returns_mode``.
    """

    def __init__(self, config: IntermarketConfig) -> None:  # type: ignore[name-defined]
        self._config = config
        # Keyed by symbol → list of close prices (oldest first)
        self._histories: dict[str, list[float]] = {}

    # ── Write ──────────────────────────────────────────────────────────────────

    def update(self, symbol: str, closes: list[float]) -> None:
        """Store or replace the close-price history for a symbol.

        Only the most recent ``lookback_bars + 10`` prices are retained
        to keep memory bounded.

        Args:
            symbol: Instrument identifier, e.g. ``"EURUSD"``.
            closes: Price series, oldest-first, newest-last.
        """
        cap = self._config.corr_lookback_bars + 10
        self._histories[symbol] = list(closes[-cap:])

    # ── Read ───────────────────────────────────────────────────────────────────

    def compute_pair(
        self,
        sym_a: str,
        sym_b: str,
        timestamp_utc: datetime,
    ) -> PairCorrelation | None:  # type: ignore[name-defined]
        """Compute rolling Pearson correlation between two symbols.

        Returns ``None`` when:
        - Either symbol has no history stored.
        - The aligned overlap is below ``corr_min_overlap_bars``.
        - Either return series has zero variance.

        Args:
            sym_a:         First symbol.
            sym_b:         Second symbol.
            timestamp_utc: Timestamp to stamp the result with.

        Returns:
            ``PairCorrelation`` or ``None``.
        """
        from metatrade.intermarket.contracts import PairCorrelation

        closes_a = self._histories.get(sym_a)
        closes_b = self._histories.get(sym_b)
        if not closes_a or not closes_b:
            return None

        ret_a = self._returns(closes_a)
        ret_b = self._returns(closes_b)

        # Use the aligned overlap (take from the tail of each)
        overlap = min(len(ret_a), len(ret_b))
        min_overlap = self._config.corr_min_overlap_bars
        if overlap < min_overlap:
            return None

        window = min(overlap, self._config.corr_lookback_bars)
        r_a = np.array(ret_a[-window:], dtype=float)
        r_b = np.array(ret_b[-window:], dtype=float)

        corr = _safe_corrcoef(r_a, r_b)
        if corr is None or np.isnan(corr):
            return None

        # Clamp to [-1, 1] to guard against floating-point drift
        corr = float(np.clip(corr, -1.0, 1.0))
        return PairCorrelation(
            symbol_a=sym_a,
            symbol_b=sym_b,
            correlation=corr,
            abs_correlation=abs(corr),
            lookback_bars=window,
            overlap_bars=window,
            returns_mode=self._config.corr_returns_mode,
            timestamp_utc=timestamp_utc,
        )

    def compute_matrix(
        self,
        symbols: list[str],
        timestamp_utc: datetime,
    ) -> dict[tuple[str, str], PairCorrelation]:  # type: ignore[name-defined]
        """Compute correlations for all unique symbol pairs.

        Only pairs for which ``compute_pair`` returns a non-None result
        are included.  Pairs are stored with the lexicographically smaller
        symbol as ``symbol_a``.

        Args:
            symbols:       List of symbols to include.
            timestamp_utc: Timestamp for all produced ``PairCorrelation`` objects.

        Returns:
            Dict mapping ``(symbol_a, symbol_b)`` → ``PairCorrelation``.
        """
        result: dict[tuple[str, str], PairCorrelation] = {}  # type: ignore[type-arg]
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i + 1:]:
                a, b = (sym_a, sym_b) if sym_a <= sym_b else (sym_b, sym_a)
                pc = self.compute_pair(a, b, timestamp_utc)
                if pc is not None:
                    result[(a, b)] = pc
        return result

    def known_symbols(self) -> list[str]:
        """Return symbols that currently have price history loaded."""
        return list(self._histories)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _returns(self, closes: list[float]) -> list[float]:
        """Compute returns according to the configured ``returns_mode``."""
        if self._config.corr_returns_mode == "log":
            return _to_log_returns(closes).tolist()
        return _to_returns(closes).tolist()
