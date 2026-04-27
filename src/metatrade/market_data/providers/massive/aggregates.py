"""Fetch aggregate bars from Massive /v2/aggs/ticker/.../range/..."""

from __future__ import annotations

import urllib.parse
from datetime import date, timedelta
from typing import Any

from metatrade.core.log import get_logger
from metatrade.market_data.providers.massive.http_client import ThrottledMassiveClient
from metatrade.market_data.providers.massive.settings import MassiveSettings
from metatrade.market_data.providers.massive.ticker import normalize_forex_ticker
from metatrade.market_data.providers.massive.timeframes import (
    timeframe_minutes,
    timeframe_to_aggs_params,
)

log = get_logger(__name__)


def _date_chunks(d0: date, d1: date, max_days: int) -> list[tuple[date, date]]:
    """Inclusive chunk ranges [a,b] covering [d0,d1]."""
    if d0 > d1:
        return []
    out: list[tuple[date, date]] = []
    cur = d0
    while cur <= d1:
        end = min(cur + timedelta(days=max_days - 1), d1)
        out.append((cur, end))
        cur = end + timedelta(days=1)
    return out


def max_chunk_calendar_days(timeframe: str, settings: MassiveSettings) -> int:
    """Keep each request under max_aggs_limit bars (minute-based TFs)."""
    bar_min = timeframe_minutes(timeframe)
    max_minutes = settings.max_aggs_limit * bar_min
    days = max(1, max_minutes // (24 * 60))
    return min(days, settings.max_history_days)


def _parse_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("results")
    if not raw:
        return []
    return list(raw)


def fetch_aggregates_range(
    *,
    client: ThrottledMassiveClient,
    settings: MassiveSettings,
    symbol: str,
    timeframe: str,
    date_from: date,
    date_to: date,
) -> list[tuple[int, float, float, float, float, float]]:
    """Return sorted (t_ms, o,h,l,c,v) tuples."""
    ticker = normalize_forex_ticker(symbol)
    mult, span = timeframe_to_aggs_params(timeframe)
    chunk_days = max_chunk_calendar_days(timeframe, settings)
    chunks = _date_chunks(date_from, date_to, chunk_days)

    merged: dict[int, tuple[float, float, float, float, float]] = {}
    for a, b in chunks:
        from_s = a.isoformat()
        to_s = b.isoformat()
        path = f"/v2/aggs/ticker/{urllib.parse.quote(ticker, safe=':')}/range/{mult}/{span}/{from_s}/{to_s}"
        payload = client.get_json(
            path,
            {
                "adjusted": "true",
                "sort": "asc",
                "limit": str(settings.max_aggs_limit),
            },
        )
        status = str(payload.get("status") or "")
        if status and status not in ("OK", "DELAYED"):
            log.warning("massive_aggs_status", status=status, ticker=ticker, from_=from_s, to=to_s)
        for row in _parse_results(payload):
            t_ms = int(row["t"])
            merged[t_ms] = (
                float(row["o"]),
                float(row["h"]),
                float(row["l"]),
                float(row["c"]),
                float(row.get("v") or row.get("n") or 0),
            )
        n = int(payload.get("resultsCount") or len(_parse_results(payload)))
        if n >= settings.max_aggs_limit:
            log.warning(
                "massive_aggs_chunk_hit_limit",
                n=n,
                limit=settings.max_aggs_limit,
                from_=from_s,
                to=to_s,
                ticker=ticker,
                msg="Possibile troncamento: ridurre il chunk o verificare i dati.",
            )

    out = [(ts, *merged[ts]) for ts in sorted(merged)]
    log.info(
        "massive_aggs_merged",
        ticker=ticker,
        timeframe=timeframe,
        n_bars=len(out),
        date_from=date_from.isoformat(),
        date_to=date_to.isoformat(),
    )
    return out
