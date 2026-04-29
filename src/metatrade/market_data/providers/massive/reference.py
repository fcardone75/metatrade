"""Reference data helpers for Massive Forex tickers."""

from __future__ import annotations

from metatrade.core.log import get_logger
from metatrade.market_data.providers.massive.http_client import ThrottledMassiveClient
from metatrade.market_data.providers.massive.ticker import strip_massive_prefix

log = get_logger(__name__)


def list_active_forex_symbols(client: ThrottledMassiveClient, *, limit: int = 1000) -> list[str]:
    """Return active Forex symbols available from Massive reference data."""
    symbols: list[str] = []
    next_url: str | None = "/v3/reference/tickers"
    params: dict[str, str] | None = {
        "market": "fx",
        "active": "true",
        "order": "asc",
        "sort": "ticker",
        "limit": str(limit),
    }

    while next_url:
        payload = client.get_json(next_url, params)
        params = None
        for row in payload.get("results") or []:
            ticker = str(row.get("ticker") or "").strip().upper()
            if ticker:
                symbols.append(strip_massive_prefix(ticker))
        next_url = payload.get("next_url")

    out = sorted(set(symbols))
    log.info("massive_forex_symbols_discovered", count=len(out))
    return out
