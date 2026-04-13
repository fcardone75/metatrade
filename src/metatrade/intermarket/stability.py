"""Dependency stability monitor for intermarket analysis.

Tracks the rolling history of correlation values for each instrument pair
and detects two types of degradation:

  1. High variance: the correlation fluctuates a lot (unstable relationship)
  2. Regime break: the sign of the correlation reverses (structural change)

The stability score penalises both types: a relationship that was reliably
+0.7 but suddenly flips to -0.6 will have both low stability AND trigger
a regime-break warning.
"""

from __future__ import annotations

import numpy as np


# Maximum rolling std that maps to zero stability.
# std = 0 → stability = 1.0 (perfectly stable)
# std ≥ MAX_STD → stability = 0.0 (maximally unstable)
_MAX_STD: float = 0.5


class StabilityMonitor:
    """Tracks correlation history per instrument pair and computes stability.

    Args:
        stability_window:       Number of recent correlation observations
                                used to measure variance.
        regime_break_threshold: Minimum absolute difference in mean correlation
                                between first/second half of history to flag
                                a regime break. Signs must also differ.
    """

    def __init__(
        self,
        stability_window: int = 100,
        regime_break_threshold: float = 0.4,
    ) -> None:
        self._window = stability_window
        self._regime_threshold = regime_break_threshold
        # pair_key → list of correlation values (chronological)
        self._history: dict[str, list[float]] = {}

    # ── Write ──────────────────────────────────────────────────────────────────

    def update(self, pair_key: str, correlation: float) -> None:
        """Record a new correlation observation for the given pair."""
        if pair_key not in self._history:
            self._history[pair_key] = []
        self._history[pair_key].append(correlation)
        # Keep at most 2× stability_window values (oldest half used for regime detection)
        cap = self._window * 2
        if len(self._history[pair_key]) > cap:
            self._history[pair_key] = self._history[pair_key][-cap:]

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_stability_score(self, pair_key: str) -> float:
        """Return stability in [0, 1].

        0.5 is returned when there is not enough history to assess stability.
        """
        history = self._history.get(pair_key, [])
        if len(history) < 5:
            return 0.5  # neutral / not enough data

        recent = history[-self._window:]
        std = float(np.std(recent))
        return float(max(0.0, 1.0 - std / _MAX_STD))

    def is_regime_break(self, pair_key: str) -> bool:
        """True if the correlation has reversed sign recently.

        Requires at least 20 observations. Compares the mean of the first
        half vs. the second half of the full history buffer.
        """
        history = self._history.get(pair_key, [])
        if len(history) < 20:
            return False

        mid = len(history) // 2
        first_mean = float(np.mean(history[:mid]))
        second_mean = float(np.mean(history[mid:]))

        signs_flipped = np.sign(first_mean) != np.sign(second_mean)
        magnitude_shifted = abs(first_mean - second_mean) > self._regime_threshold
        return bool(signs_flipped and magnitude_shifted)

    def get_warning(self, pair_key: str) -> str | None:
        """Return a human-readable warning string, or None if all is well."""
        if self.is_regime_break(pair_key):
            return f"Regime break detected for {pair_key}: correlation sign has reversed"

        stability = self.get_stability_score(pair_key)
        if stability < 0.3:
            return f"Low stability ({stability:.2f}) for {pair_key}: correlation is highly variable"

        return None

    def get_history(self, pair_key: str) -> list[float]:
        """Return the full correlation history for a pair (read-only copy)."""
        return list(self._history.get(pair_key, []))

    def reset(self, pair_key: str | None = None) -> None:
        """Clear history for one pair or all pairs."""
        if pair_key is None:
            self._history.clear()
        else:
            self._history.pop(pair_key, None)
