"""High-level: ensure CSV cache for Massive forex aggregates."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from metatrade.core.log import get_logger
from metatrade.market_data.providers.massive.aggregates import fetch_aggregates_range
from metatrade.market_data.providers.massive.http_client import ThrottledMassiveClient
from metatrade.market_data.providers.massive.postprocess import fill_micro_gaps, write_ohlcv_csv
from metatrade.market_data.providers.massive.settings import MassiveSettings
from metatrade.market_data.providers.massive.window import cache_path, massive_date_window

log = get_logger(__name__)


def ensure_massive_forex_csv(
    *,
    symbol: str,
    timeframe: str,
    bars: int,
    cache_dir: str | Path,
    refresh: bool = False,
    date_from: date | None = None,
    date_to: date | None = None,
    settings: MassiveSettings | None = None,
    client: ThrottledMassiveClient | None = None,
) -> Path:
    """Download if missing (or refresh), return path to CSV compatible with CsvCollector."""
    cfg = settings or MassiveSettings.from_env()
    d_from, d_to = massive_date_window(
        timeframe=timeframe,
        bars=bars,
        date_from_override=date_from,
        date_to_override=date_to,
        max_history_days=cfg.max_history_days,
    )
    out = cache_path(cache_dir, symbol, timeframe, d_from, d_to)
    if out.exists() and not refresh:
        log.info("massive_cache_hit", path=str(out))
        return out

    log.info(
        "massive_cache_fetch",
        path=str(out),
        symbol=symbol,
        timeframe=timeframe,
        date_from=d_from.isoformat(),
        date_to=d_to.isoformat(),
        refresh=refresh,
    )
    http = client or ThrottledMassiveClient(cfg)
    raw_rows = fetch_aggregates_range(
        client=http,
        settings=cfg,
        symbol=symbol,
        timeframe=timeframe,
        date_from=d_from,
        date_to=d_to,
    )
    filled = fill_micro_gaps(raw_rows, timeframe, cfg.gap_fill_max_bars)
    write_ohlcv_csv(out, filled)
    log.info("massive_cache_written", path=str(out), n_rows=len(filled))
    return out


def collect_timeframes_for_training(timeframes: list[str], multires: bool) -> list[str]:
    """Primary timeframes plus M5/M15 when multires is enabled."""
    seen: set[str] = set()
    ordered: list[str] = []
    for tf in timeframes:
        u = tf.strip().upper()
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    if multires:
        for h in ("M5", "M15"):
            if h not in seen:
                seen.add(h)
                ordered.append(h)
    return ordered
