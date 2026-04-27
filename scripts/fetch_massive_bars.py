"""Download Forex OHLCV from Massive.com into CSV (cache for training).

Requires MASSIVE_API_KEY in the environment (.env supported via python-dotenv).

Examples:

    python scripts/fetch_massive_bars.py --symbol EURUSD --timeframe M1 --bars 30000

    python scripts/fetch_massive_bars.py --symbol EURUSD --timeframe H1 \\
        --massive-from 2024-01-01 --massive-to 2025-01-01

    python scripts/fetch_massive_bars.py --symbol EURUSD --timeframe M1 --bars 10000 --refresh
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv  # noqa: E402

from metatrade.core.log import configure_logging, get_logger  # noqa: E402
from metatrade.market_data.providers.massive import (  # noqa: E402
    ensure_massive_forex_csv,
)

log = get_logger("fetch_massive_bars")

_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")


def _parse_date(s: str) -> date:
    parts = s.strip().split("-")
    if len(parts) != 3:
        msg = f"Data non valida (usa YYYY-MM-DD): {s!r}"
        raise argparse.ArgumentTypeError(msg)
    y, m, d = (int(parts[0]), int(parts[1]), int(parts[2]))
    return date(y, m, d)


def main() -> None:
    load_dotenv()
    configure_logging()
    p = argparse.ArgumentParser(description="Scarica barre Forex da Massive.com in CSV")
    p.add_argument("--symbol", default="EURUSD", help="Simbolo es. EURUSD")
    p.add_argument("--timeframe", default="H1", choices=_TIMEFRAMES)
    p.add_argument(
        "--bars",
        type=int,
        default=30_000,
        help="Finestra = ultimi N barre (se non usi --massive-from/--massive-to)",
    )
    p.add_argument(
        "--massive-from", type=_parse_date, default=None, help="Data inizio YYYY-MM-DD (inclusa)"
    )
    p.add_argument(
        "--massive-to", type=_parse_date, default=None, help="Data fine YYYY-MM-DD (inclusa)"
    )
    p.add_argument("--cache-dir", type=Path, default=Path("data/massive"))
    p.add_argument("--refresh", action="store_true", help="Forza riscarico anche se il CSV esiste")
    args = p.parse_args()

    if (args.massive_from is None) ^ (args.massive_to is None):
        log.error(
            "massive_date_pair",
            msg="Specificare entrambi --massive-from e --massive-to, oppure nessuno",
        )
        sys.exit(1)

    try:
        path = ensure_massive_forex_csv(
            symbol=args.symbol,
            timeframe=args.timeframe,
            bars=args.bars,
            cache_dir=args.cache_dir,
            refresh=args.refresh,
            date_from=args.massive_from,
            date_to=args.massive_to,
        )
    except OSError as e:
        log.error("massive_fetch_failed", error=str(e))
        sys.exit(1)

    print(path)


if __name__ == "__main__":
    main()
