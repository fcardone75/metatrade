"""Trading session utilities shared across the ML and runner layers."""

from __future__ import annotations

from datetime import datetime


def is_in_session(ts_utc: datetime, start_hour: int, end_hour: int) -> bool:
    """Return True if ``ts_utc`` falls within the trading session window.

    Args:
        ts_utc:     UTC-aware (or UTC-naive) datetime to check.
        start_hour: Session start hour (0–23, inclusive).
        end_hour:   Session end hour (0–23, exclusive).

    The function handles overnight wraps (e.g. start=22, end=6).
    """
    h = ts_utc.hour
    if start_hour < end_hour:
        return start_hour <= h < end_hour
    # overnight: e.g. 22–06 UTC
    return h >= start_hour or h < end_hour
