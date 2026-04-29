"""Run Massive-to-Mongo backfill in resumable symbol batches.

This wrapper calls scripts/backfill_massive_mongo.py repeatedly with
--symbols all, --symbol-offset, and --symbol-limit. It is intended for long
backfills where running all discovered symbols in a single process is awkward.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dotenv import load_dotenv  # noqa: E402
from pymongo import MongoClient  # noqa: E402

from metatrade.market_data.providers.massive.http_client import ThrottledMassiveClient  # noqa: E402
from metatrade.market_data.providers.massive.reference import list_active_forex_symbols  # noqa: E402
from metatrade.market_data.providers.massive.settings import MassiveSettings  # noqa: E402
from metatrade.ml.distributed.config import MongoTrainConfig  # noqa: E402

_ALL_TIMEFRAMES = "M1,M5,M15,M30,H1,H4,D1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Esegue backfill Massive->Mongo a batch")
    p.add_argument("--env-file", default=".env.worker")
    p.add_argument("--timeframes", default=_ALL_TIMEFRAMES)
    p.add_argument("--from", dest="date_from", default="oldest")
    p.add_argument("--to", dest="date_to", default="")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--start-offset", type=int, default=0)
    p.add_argument("--run-id", default="", help="Checkpoint id Mongo. Default: tf+from+to")
    p.add_argument("--resume", action="store_true", help="Riprende da next_offset del checkpoint")
    p.add_argument("--max-batches", type=int, default=0, help="0 = tutti i batch rimanenti")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _checkpoint_id(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    to_part = args.date_to or "yesterday"
    tf_part = args.timeframes.replace(",", "_").replace(" ", "_")
    return f"massive_all_{tf_part}_{args.date_from}_{to_part}"


def _open_checkpoint() -> tuple[MongoClient[Any], Any]:
    cfg = MongoTrainConfig()
    if not cfg.mongo_uri:
        raise SystemExit("MONGO_URI non impostata")
    client: MongoClient[Any] = MongoClient(cfg.mongo_uri)
    return client, client[cfg.mongo_db]["massive_backfill_runs"]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size deve essere > 0")

    load_dotenv(args.env_file, override=True)
    cfg = MassiveSettings.from_env()
    symbols = list_active_forex_symbols(ThrottledMassiveClient(cfg))
    total = len(symbols)

    run_id = _checkpoint_id(args)
    mongo_client: MongoClient[Any] | None = None
    checkpoint_col: Any | None = None
    if not args.dry_run:
        mongo_client, checkpoint_col = _open_checkpoint()
        checkpoint_col.update_one(
            {"_id": run_id},
            {
                "$setOnInsert": {
                    "created_at": __import__("datetime").datetime.datetime.now(
                        __import__("datetime").datetime.UTC
                    ),
                },
                "$set": {
                    "timeframes": args.timeframes,
                    "batch_size": args.batch_size,
                    "total_symbols": total,
                    "status": "running",
                },
            },
            upsert=True,
        )

    offset = args.start_offset
    if args.resume:
        if checkpoint_col is None:
            raise SystemExit("--resume richiede Mongo, non usare --dry-run")
        doc = checkpoint_col.find_one({"_id": run_id}) or {}
        offset = int(doc.get("next_offset") or args.start_offset)

    if offset >= total:
        raise SystemExit(f"start-offset {args.start_offset} >= simboli disponibili {total}")

    script = Path(__file__).resolve().parent / "backfill_massive_mongo.py"
    completed = 0
    try:
        while offset < total:
            if args.max_batches and completed >= args.max_batches:
                break
            limit = min(args.batch_size, total - offset)
            if checkpoint_col is not None:
                checkpoint_col.update_one(
                    {"_id": run_id},
                    {"$set": {"current_offset": offset, "current_limit": limit}},
                )
            cmd = [
                sys.executable,
                str(script),
                "--env-file",
                args.env_file,
                "--symbols",
                "all",
                "--symbol-offset",
                str(offset),
                "--symbol-limit",
                str(limit),
                "--timeframes",
                args.timeframes,
                "--from",
                args.date_from,
            ]
            if args.date_to:
                cmd.extend(["--to", args.date_to])
            if args.dry_run:
                cmd.append("--dry-run")

            print(f"batch offset={offset} limit={limit} timeframes={args.timeframes}", flush=True)
            subprocess.run(cmd, check=True)
            offset += limit
            completed += 1
            if checkpoint_col is not None:
                checkpoint_col.update_one(
                    {"_id": run_id},
                    {"$set": {"next_offset": offset, "completed_batches": completed}},
                )
    finally:
        if checkpoint_col is not None:
            status = "completed" if offset >= total else "paused"
            checkpoint_col.update_one(
                {"_id": run_id},
                {"$set": {"status": status, "next_offset": offset}},
            )
        if mongo_client is not None:
            mongo_client.close()

    print(f"run_id={run_id} done batches={completed} next_offset={offset} total_symbols={total}", flush=True)


if __name__ == "__main__":
    main()
