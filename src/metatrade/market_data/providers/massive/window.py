"""Date window computation for Massive cache keys and API range."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from metatrade.market_data.providers.massive.timeframes import timeframe_minutes


def massive_date_window(
    *,
    timeframe: str,
    bars: int,
    date_from_override: date | None,
    date_to_override: date | None,
    max_history_days: int,
) -> tuple[date, date]:
    """Return inclusive [from, to] in UTC calendar dates for API from/to."""
    if date_from_override is not None and date_to_override is not None:
        if date_from_override > date_to_override:
            msg = f"massive-from ({date_from_override}) > massive-to ({date_to_override})"
            raise ValueError(msg)
        d_to = date_to_override
        d_from = date_from_override
    else:
        now = datetime.now(UTC)
        d_to = now.date()
        bar_min = timeframe_minutes(timeframe)
        minutes = max(1, bars) * bar_min
        start_dt = now - timedelta(minutes=minutes)
        d_from = start_dt.date()

    oldest = d_to - timedelta(days=max_history_days)
    if d_from < oldest:
        d_from = oldest
    return d_from, d_to


def cache_path(
    cache_dir: str | Path,
    symbol: str,
    timeframe: str,
    d_from: date,
    d_to: date,
) -> Path:
    sym = symbol.strip().upper()
    if sym.startswith("C:"):
        sym = sym[2:]
    name = f"{sym}_{timeframe.upper()}_{d_from.strftime('%Y%m%d')}_{d_to.strftime('%Y%m%d')}.csv"
    return Path(cache_dir) / name


# train.py adaptive schedule max bar_scale
_ADAPTIVE_MAX_BAR_SCALE = 2.0


def max_adaptive_bars(base_bars: int) -> int:
    """Upper bound on bars used across adaptive attempts (matches train schedule scale)."""
    return min(int(base_bars * _ADAPTIVE_MAX_BAR_SCALE), 100_000)
