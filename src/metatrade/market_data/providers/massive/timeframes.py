"""Map Metatrade timeframe codes to Massive aggs multiplier + timespan."""

from __future__ import annotations

# minute-only multipliers avoid relying on hour/day support edge cases
_TF_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def timeframe_to_aggs_params(timeframe: str) -> tuple[int, str]:
    """Return (multiplier, timespan) for /v2/aggs/ticker/.../range/..."""
    tf = timeframe.strip().upper()
    if tf not in _TF_MINUTES:
        msg = f"timeframe Massive non supportato: {timeframe!r}"
        raise ValueError(msg)
    minutes = _TF_MINUTES[tf]
    if tf == "D1":
        return 1, "day"
    return minutes, "minute"


def timeframe_minutes(timeframe: str) -> int:
    tf = timeframe.strip().upper()
    if tf not in _TF_MINUTES:
        msg = f"timeframe sconosciuto: {timeframe!r}"
        raise ValueError(msg)
    return _TF_MINUTES[tf]
