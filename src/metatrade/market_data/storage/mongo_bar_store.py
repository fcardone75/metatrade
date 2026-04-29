"""MongoDB storage for normalized market bars."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, UpdateOne

from metatrade.core.log import get_logger

log = get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class MongoBarStore:
    """Idempotent OHLCV bar storage backed by MongoDB."""

    def __init__(self, db: Any, collection_name: str = "market_bars") -> None:
        self._col = db[collection_name]

    def ensure_indexes(self) -> None:
        self._col.create_index(
            [
                ("provider", ASCENDING),
                ("symbol", ASCENDING),
                ("timeframe", ASCENDING),
                ("ts", ASCENDING),
            ],
            unique=True,
            name="uniq_provider_symbol_timeframe_ts",
        )

    def latest_ts(self, *, provider: str, symbol: str, timeframe: str) -> datetime | None:
        doc = self._col.find_one(
            {
                "provider": provider,
                "symbol": symbol.upper(),
                "timeframe": timeframe.upper(),
            },
            {"ts": 1},
            sort=[("ts", -1)],
        )
        if doc is None:
            return None
        ts = doc.get("ts")
        return ts if isinstance(ts, datetime) else None

    def count(self, *, provider: str, symbol: str, timeframe: str) -> int:
        return int(
            self._col.count_documents(
                {
                    "provider": provider,
                    "symbol": symbol.upper(),
                    "timeframe": timeframe.upper(),
                }
            )
        )

    def upsert_massive_rows(
        self,
        *,
        symbol: str,
        timeframe: str,
        rows: list[tuple[int, float, float, float, float, float]],
        source_from: str,
        source_to: str,
    ) -> int:
        """Upsert Massive aggregate tuples and return modified/upserted count."""
        if not rows:
            return 0

        now = _utcnow()
        sym = symbol.upper()
        tf = timeframe.upper()
        ops: list[UpdateOne] = []
        for t_ms, open_, high, low, close, volume in rows:
            ts = datetime.fromtimestamp(t_ms / 1000.0, tz=UTC)
            filt = {
                "provider": "massive",
                "symbol": sym,
                "timeframe": tf,
                "ts": ts,
            }
            doc = {
                "provider": "massive",
                "symbol": sym,
                "timeframe": tf,
                "ts": ts,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "is_closed": True,
                "source_from": source_from,
                "source_to": source_to,
                "updated_at": now,
            }
            ops.append(
                UpdateOne(
                    filt,
                    {
                        "$set": doc,
                        "$setOnInsert": {"created_at": now},
                    },
                    upsert=True,
                )
            )

        result = self._col.bulk_write(ops, ordered=False)
        changed = int(result.upserted_count + result.modified_count)
        log.info(
            "mongo_bar_store_upserted",
            provider="massive",
            symbol=sym,
            timeframe=tf,
            rows=len(rows),
            changed=changed,
        )
        return changed
