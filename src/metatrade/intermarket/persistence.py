"""Persistence adapter for intermarket portfolio-risk events.

Stores correlation snapshots, exposure snapshots, veto decisions, and
risk-multiplier events in the same SQLite database as TelemetryStore,
under dedicated tables.

Design principles
-----------------
- Optional: ``IntermarketEngine`` works without a store.  Pass ``store=None``
  to disable persistence entirely.
- Low coupling: the store is a thin SQLite wrapper; no trading logic here.
- WAL mode is assumed to be enabled by the TelemetryStore connection
  (shared database), so we do not re-enable it.
- Errors in persistence never propagate to the trading pipeline — they are
  logged and swallowed.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import structlog

from metatrade.intermarket.contracts import IntermarketDecision, IntermarketSnapshot

log = structlog.get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS intermarket_decisions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT,
    symbol           TEXT    NOT NULL,
    side             TEXT    NOT NULL,
    approved         INTEGER NOT NULL,
    risk_multiplier  REAL    NOT NULL,
    reason           TEXT    NOT NULL,
    warnings_json    TEXT    NOT NULL DEFAULT '[]',
    correlated_json  TEXT    NOT NULL DEFAULT '[]',
    exposure_json    TEXT    NOT NULL DEFAULT '{}',
    timestamp_utc    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS intermarket_correlation_snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT,
    symbol_a       TEXT    NOT NULL,
    symbol_b       TEXT    NOT NULL,
    correlation    REAL    NOT NULL,
    overlap_bars   INTEGER NOT NULL,
    returns_mode   TEXT    NOT NULL,
    timestamp_utc  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS intermarket_exposure_snapshots (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT,
    currency       TEXT    NOT NULL,
    net_lots       REAL    NOT NULL,
    gross_lots     REAL    NOT NULL,
    timestamp_utc  INTEGER NOT NULL
);
"""


def _ts(dt: datetime) -> int:
    """Convert a UTC-aware datetime to a Unix timestamp integer."""
    return int(dt.timestamp())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class IntermarketStore:
    """SQLite adapter for intermarket portfolio-risk events.

    Can share a database file with ``TelemetryStore``; the tables are
    namespaced with the ``intermarket_`` prefix.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._connect()

    @classmethod
    def from_env(cls) -> "IntermarketStore":
        """Create an ``IntermarketStore`` reading ``TELEMETRY_DB_PATH`` from env.

        Falls back to ``data/telemetry.db``.
        """
        import os
        db_path = os.environ.get("TELEMETRY_DB_PATH", "data/telemetry.db")
        return cls(db_path)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        if self._conn is None:
            return
        self._conn.executescript(_DDL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Write operations ───────────────────────────────────────────────────────

    def record_decision(
        self,
        symbol: str,
        side: str,
        decision: IntermarketDecision,
        session_id: str | None = None,
    ) -> None:
        """Persist one ``IntermarketDecision`` (veto or approval with multiplier).

        Errors are logged and swallowed so the trading pipeline is never
        interrupted by a storage failure.

        Args:
            symbol:     Instrument being evaluated.
            side:       "BUY" or "SELL".
            decision:   The decision to persist.
            session_id: Optional session identifier for cross-referencing.
        """
        if self._conn is None:
            return
        try:
            exposure_serializable: dict[str, str] = {
                k: str(v) for k, v in decision.exposure_delta_by_ccy.items()
            }
            self._conn.execute(
                """
                INSERT INTO intermarket_decisions
                    (session_id, symbol, side, approved, risk_multiplier,
                     reason, warnings_json, correlated_json, exposure_json,
                     timestamp_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    symbol,
                    side,
                    int(decision.approved),
                    float(decision.risk_multiplier),
                    decision.reason,
                    json.dumps(list(decision.warnings)),
                    json.dumps(list(decision.correlated_positions)),
                    json.dumps(exposure_serializable),
                    _ts(_utc_now()),
                ),
            )
            self._conn.commit()
        except Exception as exc:
            log.warning(
                "intermarket_store_decision_failed",
                symbol=symbol,
                error=str(exc),
            )

    def record_correlation_snapshot(
        self,
        snapshot: IntermarketSnapshot,
        session_id: str | None = None,
    ) -> None:
        """Persist all ``PairCorrelation`` objects from a snapshot.

        Args:
            snapshot:   Produced by ``IntermarketEngine``.
            session_id: Optional session identifier.
        """
        if self._conn is None or not snapshot.correlations:
            return
        try:
            ts = _ts(snapshot.timestamp_utc)
            rows: list[tuple[Any, ...]] = [
                (
                    session_id,
                    pc.symbol_a,
                    pc.symbol_b,
                    pc.correlation,
                    pc.overlap_bars,
                    pc.returns_mode,
                    ts,
                )
                for pc in snapshot.correlations
            ]
            self._conn.executemany(
                """
                INSERT INTO intermarket_correlation_snapshots
                    (session_id, symbol_a, symbol_b, correlation,
                     overlap_bars, returns_mode, timestamp_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
        except Exception as exc:
            log.warning(
                "intermarket_store_correlation_snapshot_failed",
                error=str(exc),
            )

    def record_exposure_snapshot(
        self,
        snapshot: IntermarketSnapshot,
        session_id: str | None = None,
    ) -> None:
        """Persist all ``CurrencyExposure`` objects from a snapshot.

        Args:
            snapshot:   Produced by ``IntermarketEngine``.
            session_id: Optional session identifier.
        """
        if self._conn is None or not snapshot.exposures:
            return
        try:
            ts = _ts(snapshot.timestamp_utc)
            rows: list[tuple[Any, ...]] = [
                (
                    session_id,
                    exp.currency,
                    float(exp.net_lots),
                    float(exp.gross_lots),
                    ts,
                )
                for exp in snapshot.exposures
            ]
            self._conn.executemany(
                """
                INSERT INTO intermarket_exposure_snapshots
                    (session_id, currency, net_lots, gross_lots, timestamp_utc)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
        except Exception as exc:
            log.warning(
                "intermarket_store_exposure_snapshot_failed",
                error=str(exc),
            )

    # ── Read operations ────────────────────────────────────────────────────────

    def list_recent_decisions(
        self,
        *,
        limit: int = 50,
        session_id: str | None = None,
        vetoes_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Return recent intermarket decisions, newest first.

        Args:
            limit:       Maximum number of rows to return.
            session_id:  If set, filter to this session.
            vetoes_only: If True, return only vetoed decisions.
        """
        if self._conn is None:
            return []
        try:
            clauses = []
            params: list[Any] = []
            if session_id is not None:
                clauses.append("session_id = ?")
                params.append(session_id)
            if vetoes_only:
                clauses.append("approved = 0")
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params.append(limit)
            rows = self._conn.execute(
                f"SELECT * FROM intermarket_decisions {where} "
                f"ORDER BY timestamp_utc DESC LIMIT ?",
                params,
            ).fetchall()
            cols = [
                "id", "session_id", "symbol", "side", "approved",
                "risk_multiplier", "reason", "warnings_json",
                "correlated_json", "exposure_json", "timestamp_utc",
            ]
            return [dict(zip(cols, row)) for row in rows]
        except Exception as exc:
            log.warning("intermarket_store_list_decisions_failed", error=str(exc))
            return []
