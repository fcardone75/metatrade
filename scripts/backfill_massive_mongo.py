"""Backfill Massive Forex OHLCV bars into MongoDB.

Example:
    python scripts/backfill_massive_mongo.py --symbols EURUSD,GBPUSD --timeframes H1,M15 \
        --from 2024-01-01 --to 2024-12-31
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv  # noqa: E402
from pymongo import MongoClient  # noqa: E402

from metatrade.core.log import configure_logging, get_logger  # noqa: E402
from metatrade.market_data.providers.massive.aggregates import (  # noqa: E402
    fetch_aggregates_range,
    fetch_first_aggregate_date,
)
from metatrade.market_data.providers.massive.http_client import ThrottledMassiveClient  # noqa: E402
from metatrade.market_data.providers.massive.postprocess import fill_micro_gaps  # noqa: E402
from metatrade.market_data.providers.massive.reference import list_active_forex_symbols  # noqa: E402
from metatrade.market_data.providers.massive.settings import MassiveSettings  # noqa: E402
from metatrade.market_data.providers.massive.window import massive_date_window  # noqa: E402
from metatrade.market_data.storage.mongo_bar_store import MongoBarStore  # noqa: E402
from metatrade.ml.distributed.config import MongoTrainConfig  # noqa: E402

log = get_logger("backfill_massive_mongo")

_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")


def _parse_date_or_oldest(raw: str) -> date | str:
    if raw.strip().lower() == "oldest":
        return "oldest"
    try:
        return date.fromisoformat(raw.strip())
    except ValueError as exc:
        msg = f"Data non valida (usa YYYY-MM-DD): {raw!r}"
        raise argparse.ArgumentTypeError(msg) from exc


def _split_csv(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.replace(" ", ",").split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scarica storico Massive e lo salva in MongoDB")
    p.add_argument("--env-file", default=".env.worker", help="File env da caricare prima di .env")
    p.add_argument(
        "--symbols",
        default="EURUSD",
        help="Lista simboli separati da virgola/spazio, oppure 'all' per tutti i Forex attivi Massive",
    )
    p.add_argument("--timeframes", default="H1", help="Lista timeframe separati da virgola/spazio")
    p.add_argument(
        "--from",
        dest="date_from",
        type=_parse_date_or_oldest,
        default=None,
        help="Inizio YYYY-MM-DD oppure 'oldest' per scoprire la prima barra Massive",
    )
    p.add_argument("--to", dest="date_to", type=_parse_date_or_oldest, default=None, help="Fine YYYY-MM-DD")
    p.add_argument("--bars", type=int, default=30_000, help="Fallback finestra ultimi N bars")
    p.add_argument(
        "--oldest-search-from",
        type=_parse_date_or_oldest,
        default=date(1970, 1, 1),
        help="Data iniziale della sonda oldest (default: 1970-01-01)",
    )
    p.add_argument("--collection", default="market_bars", help="Collection Mongo per le barre")
    p.add_argument("--dry-run", action="store_true", help="Scarica ma non scrive su Mongo")
    p.add_argument(
        "--discover-only",
        action="store_true",
        help="Con --symbols all stampa i simboli scoperti e termina",
    )
    p.add_argument("--symbol-limit", type=int, default=0, help="Limita i primi N simboli dopo discovery")
    p.add_argument("--symbol-offset", type=int, default=0, help="Salta i primi N simboli dopo discovery")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.env_file, override=True)
    load_dotenv(override=False)
    configure_logging()

    timeframes = _split_csv(args.timeframes)
    invalid = [tf for tf in timeframes if tf not in _TIMEFRAMES]
    if invalid:
        raise SystemExit(f"Timeframe non supportati: {', '.join(invalid)}")
    if args.date_to == "oldest":
        raise SystemExit("--to non può essere 'oldest'")
    if args.oldest_search_from == "oldest":
        raise SystemExit("--oldest-search-from deve essere una data YYYY-MM-DD")
    if args.date_from == "oldest" and args.date_to is None:
        args.date_to = datetime.now(UTC).date() - timedelta(days=1)
    if (args.date_from is None) ^ (args.date_to is None):
        raise SystemExit("Specificare entrambi --from e --to, oppure nessuno")

    massive_cfg = MassiveSettings.from_env()
    http = ThrottledMassiveClient(massive_cfg)
    symbols = (
        list_active_forex_symbols(http)
        if args.symbols.strip().lower() == "all"
        else _split_csv(args.symbols)
    )
    if not symbols:
        raise SystemExit("Nessun simbolo da scaricare")
    if args.symbol_offset > 0 or args.symbol_limit > 0:
        start = max(0, args.symbol_offset)
        end = start + args.symbol_limit if args.symbol_limit > 0 else None
        symbols = symbols[start:end]
        if not symbols:
            raise SystemExit("Nessun simbolo dopo offset/limit")
    if args.discover_only:
        print(",".join(symbols))
        print(f"discovered={len(symbols)}")
        return

    store: MongoBarStore | None = None
    client: MongoClient | None = None
    if not args.dry_run:
        mongo_cfg = MongoTrainConfig()
        if not mongo_cfg.mongo_uri:
            raise SystemExit("MONGO_URI non impostata")
        client = MongoClient(mongo_cfg.mongo_uri)
        db = client[mongo_cfg.mongo_db]
        store = MongoBarStore(db, args.collection)
        store.ensure_indexes()

    total_rows = 0
    total_changed = 0
    try:
        for symbol in symbols:
            for timeframe in timeframes:
                if args.date_from == "oldest":
                    d_to = args.date_to
                    d_from = fetch_first_aggregate_date(
                        client=http,
                        settings=massive_cfg,
                        symbol=symbol,
                        timeframe=timeframe,
                        search_from=args.oldest_search_from,
                        search_to=d_to,
                    )
                    if d_from is None:
                        print(f"{symbol} {timeframe}: no Massive bars found")
                        continue
                else:
                    d_from, d_to = massive_date_window(
                        timeframe=timeframe,
                        bars=args.bars,
                        date_from_override=args.date_from,
                        date_to_override=args.date_to,
                        max_history_days=massive_cfg.max_history_days,
                    )
                rows = fetch_aggregates_range(
                    client=http,
                    settings=massive_cfg,
                    symbol=symbol,
                    timeframe=timeframe,
                    date_from=d_from,
                    date_to=d_to,
                )
                rows = fill_micro_gaps(rows, timeframe, massive_cfg.gap_fill_max_bars)
                total_rows += len(rows)

                changed = 0
                count_after: int | None = None
                if store is not None:
                    changed = store.upsert_massive_rows(
                        symbol=symbol,
                        timeframe=timeframe,
                        rows=rows,
                        source_from=d_from.isoformat(),
                        source_to=d_to.isoformat(),
                    )
                    total_changed += changed
                    count_after = store.count(
                        provider="massive",
                        symbol=symbol,
                        timeframe=timeframe,
                    )

                log.info(
                    "massive_mongo_backfill_pair_done",
                    symbol=symbol,
                    timeframe=timeframe,
                    date_from=d_from.isoformat(),
                    date_to=d_to.isoformat(),
                    fetched=len(rows),
                    changed=changed,
                    count_after=count_after,
                    dry_run=args.dry_run,
                )
                print(
                    f"{symbol} {timeframe}: fetched={len(rows)} changed={changed}"
                    + (f" total_in_mongo={count_after}" if count_after is not None else "")
                )
    finally:
        if client is not None:
            client.close()

    print(f"done: fetched={total_rows} changed={total_changed}")


if __name__ == "__main__":
    main()
