"""Lead-lag relationship detector for intermarket analysis.

Uses the cross-correlation profile to identify whether instrument X
systematically leads or lags instrument Y by a certain number of bars.

The confidence score measures the *sharpness* of the dominant peak:
a clear, isolated peak → high confidence; flat profile → low confidence.
"""

from __future__ import annotations

import numpy as np

from metatrade.intermarket.contracts import LeadLagResult
from metatrade.intermarket.correlation import CorrelationEngine


class LeadLagDetector:
    """Detects the dominant lead-lag relationship between two price series.

    Args:
        max_lag:  Maximum number of bars to test in each direction.
    """

    def __init__(self, max_lag: int = 20) -> None:
        self._max_lag = max_lag
        self._engine = CorrelationEngine()

    def detect(
        self,
        prices_x: list[float],
        prices_y: list[float],
    ) -> LeadLagResult | None:
        """Compute the cross-correlation profile and return the dominant lag.

        Args:
            prices_x: Primary price series (source instrument).
            prices_y: Reference price series (target instrument).

        Returns:
            LeadLagResult with dominant lag and sharpness-based confidence,
            or None if there is insufficient data.
        """
        lag_profile = self._engine.cross_correlation(
            prices_x, prices_y, self._max_lag
        )
        if len(lag_profile) < 3:
            return None

        # Find the lag with maximum absolute correlation
        dominant_lag = max(lag_profile, key=lambda k: abs(lag_profile[k]))
        max_corr = lag_profile[dominant_lag]

        # Confidence = how much the peak stands out above the average
        abs_values = np.array([abs(v) for v in lag_profile.values()])
        peak = abs(max_corr)
        mean_abs = float(np.mean(abs_values))

        # Sharpness ratio: 1.0 means no peak, approaches 1.0 as peak dominates
        # confidence = (peak - mean) / (1 - mean + ε)
        eps = 1e-6
        confidence = float((peak - mean_abs) / (1.0 - mean_abs + eps))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        return LeadLagResult(
            dominant_lag=dominant_lag,
            max_correlation=max_corr,
            lag_profile=lag_profile,
            confidence=confidence,
        )
