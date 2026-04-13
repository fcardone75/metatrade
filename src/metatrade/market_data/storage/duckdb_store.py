"""DuckDB bar store — historical OHLCV persistence.

Responsibilities:
- Persist bars fetched from MT5 or CSV sources.
- Serve bar queries to the analysis pipeline and ML training.
- Upsert semantics: saving an existing bar overwrites it (idempotent).
- All timestamps stored as Unix seconds (BIGINT) to avoid timezone issues.

Thread-safety: DuckDB file connections are NOT thread-safe.
This store is designed for single-threaded / single-asyncio-loop use.
If multi-threading is added later, use a connection pool or serialize access.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

import duckdb

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.core.log import get_logger
from metatrade.market_data.errors import BarStoreError
from metatrade.market_data.interface import IBarStore
from metatrade.market_data.models import BarQuery, SymbolCoverage
from metatrade.market_data.storage.schema import (
    DUCKDB_COUNT_BARS,
    DUCKDB_CREATE_BARS_IDX,
    DUCKDB_CREATE_BARS_TABLE,
    DUCKDB_DELETE_BARS,
    DUCKDB_SELECT_BARS,
    DUCKDB_SELECT_LATEST_TS,
    DUCKDB_SELECT_OLDEST_TS,
    DUCKDB_UPSERT_BARS,
)

log = get_logger(__name__)

# Decimal quantizer matching our price precision
_PRICE_QUANT = Decimal("0.00001")  # 5 decimal places — covers all FX pairs
_VOLUME_QUANT = Decimal("1")


class DuckDBBarStore(IBarStore):
    """Persistent DuckDB store for historical OHLCV bars.

    Args:
        db_path: Path to the DuckDB file. Use ":memory:" for tests.
        read_only: Open in read-only mode (useful for parallel readers during backtest).
    """

    def __init__(self, db_path: str = ":memory:", read_only: bool = False) -> None:
        self._db_path = db_path
        self._read_only = read_only
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._connect()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            if self._db_path != ":memory:":
                os.makedirs(os.path.dirname(self._db_path), exist_ok=True) if os.path.dirname(self._db_path) else None
            self._conn = duckdb.connect(self._db_path, read_only=self._read_only)
            if not self._read_only:
                self._create_schema()
            log.info("duckdb_connected", path=self._db_path, read_only=self._read_only)
        except Exception as exc:
            raise BarStoreError(
                message=f"Cannot connect to DuckDB at {self._db_path!r}: {exc}",
                code="DUCKDB_CONNECT_FAILED",
                context={"path": self._db_path},
            ) from exc

    def _create_schema(self) -> None:
        assert self._conn is not None
        self._conn.execute(DUCKDB_CREATE_BARS_TABLE)
        self._conn.execute(DUCKDB_CREATE_BARS_IDX)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            log.info("duckdb_closed", path=self._db_path)

    def __enter__(self) -> DuckDBBarStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Write ─────────────────────────────────────────────────────────────────

    def save_bars(self, bars: list[Bar]) -> int:
        """Upsert bars. Returns the number of rows written."""
        if not bars:
            return 0
        self._assert_writable()
        assert self._conn is not None

        rows = [_bar_to_row(b) for b in bars]
        try:
            self._conn.executemany(DUCKDB_UPSERT_BARS, rows)
            log.debug(
                "bars_saved",
                count=len(rows),
                symbol=bars[0].symbol,
                timeframe=bars[0].timeframe.value,
            )
            return len(rows)
        except Exception as exc:
            raise BarStoreError(
                message=f"Failed to save {len(bars)} bars: {exc}",
                code="DUCKDB_WRITE_FAILED",
            ) from exc

    def delete_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        date_from: datetime,
        date_to: datetime,
    ) -> int:
        self._assert_writable()
        assert self._conn is not None
        try:
            result = self._conn.execute(
                DUCKDB_DELETE_BARS,
                [symbol, timeframe.value, _to_unix(date_from), _to_unix(date_to)],
            )
            deleted = result.fetchone()[0] if result else 0
            log.info(
                "bars_deleted",
                symbol=symbol,
                timeframe=timeframe.value,
                deleted=deleted,
            )
            return deleted
        except Exception as exc:
            raise BarStoreError(
                message=f"Failed to delete bars for {symbol}/{timeframe.value}: {exc}",
                code="DUCKDB_DELETE_FAILED",
            ) from exc

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_bars(self, query: BarQuery) -> list[Bar]:
        assert self._conn is not None
        try:
            rows = self._conn.execute(
                DUCKDB_SELECT_BARS,
                [
                    query.symbol,
                    query.timeframe.value,
                    _to_unix(query.date_from),
                    _to_unix(query.date_to),
                ],
            ).fetchall()
            return [_row_to_bar(r) for r in rows]
        except Exception as exc:
            raise BarStoreError(
                message=f"Failed to query bars for {query.symbol}: {exc}",
                code="DUCKDB_READ_FAILED",
            ) from exc

    def get_latest_bar_time(self, symbol: str, timeframe: Timeframe) -> datetime | None:
        assert self._conn is not None
        row = self._conn.execute(
            DUCKDB_SELECT_LATEST_TS, [symbol, timeframe.value]
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return _from_unix(row[0])

    def get_oldest_bar_time(self, symbol: str, timeframe: Timeframe) -> datetime | None:
        assert self._conn is not None
        row = self._conn.execute(
            DUCKDB_SELECT_OLDEST_TS, [symbol, timeframe.value]
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return _from_unix(row[0])

    def count_bars(self, symbol: str, timeframe: Timeframe) -> int:
        assert self._conn is not None
        row = self._conn.execute(
            DUCKDB_COUNT_BARS, [symbol, timeframe.value]
        ).fetchone()
        return int(row[0]) if row else 0

    def get_coverage(self, symbol: str, timeframe: Timeframe) -> SymbolCoverage:
        return SymbolCoverage(
            symbol=symbol,
            timeframe=timeframe,
            oldest_bar_utc=self.get_oldest_bar_time(symbol, timeframe),
            newest_bar_utc=self.get_latest_bar_time(symbol, timeframe),
            total_bars=self.count_bars(symbol, timeframe),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _assert_writable(self) -> None:
        if self._read_only:
            raise BarStoreError(
                message="Cannot write to a read-only DuckDB connection",
                code="DUCKDB_READ_ONLY",
            )


# ── Row conversion helpers ────────────────────────────────────────────────────

def _to_unix(dt: datetime) -> int:
    """Convert UTC-aware datetime to Unix seconds integer."""
    return int(dt.timestamp())


def _from_unix(ts: int) -> datetime:
    """Convert Unix seconds integer to UTC-aware datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _bar_to_row(bar: Bar) -> tuple:
    """Convert a Bar to a DuckDB row tuple."""
    return (
        bar.symbol,
        bar.timeframe.value,
        _to_unix(bar.timestamp_utc),
        float(bar.open),
        float(bar.high),
        float(bar.low),
        float(bar.close),
        float(bar.volume),
        float(bar.spread) if bar.spread is not None else None,
    )


def _row_to_bar(row: tuple) -> Bar:
    """Convert a DuckDB row tuple back to a Bar contract."""
    symbol, timeframe_str, ts, open_, high, low, close, volume, spread = row
    return Bar(
        symbol=symbol,
        timeframe=Timeframe(timeframe_str),
        timestamp_utc=_from_unix(ts),
        open=Decimal(str(open_)).quantize(_PRICE_QUANT, rounding=ROUND_HALF_UP),
        high=Decimal(str(high)).quantize(_PRICE_QUANT, rounding=ROUND_HALF_UP),
        low=Decimal(str(low)).quantize(_PRICE_QUANT, rounding=ROUND_HALF_UP),
        close=Decimal(str(close)).quantize(_PRICE_QUANT, rounding=ROUND_HALF_UP),
        volume=Decimal(str(volume)).quantize(_VOLUME_QUANT, rounding=ROUND_HALF_UP),
        spread=Decimal(str(spread)).quantize(_PRICE_QUANT, rounding=ROUND_HALF_UP)
        if spread is not None
        else None,
    )
