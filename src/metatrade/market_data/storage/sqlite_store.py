"""SQLite store — operational data (audit log, kill switch history).

Responsibilities:
- Append-only audit trail of every trading decision.
- Kill switch activation/reset log.
- Lightweight, zero-server, always available.

WAL mode is enabled on connection to allow concurrent reads while writing.
The audit log is append-only — no updates or deletes (immutable audit trail).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from metatrade.core.log import get_logger
from metatrade.market_data.errors import AuditStoreError
from metatrade.market_data.storage.schema import (
    SQLITE_CREATE_AUDIT_IDX,
    SQLITE_CREATE_AUDIT_TABLE,
    SQLITE_CREATE_KILL_SWITCH_TABLE,
    SQLITE_INSERT_AUDIT,
    SQLITE_INSERT_KILL_SWITCH,
    SQLITE_SELECT_AUDIT_RECENT,
)

log = get_logger(__name__)


class AuditEntry:
    """Represents one row in the audit log."""

    __slots__ = (
        "ts", "event_type", "symbol", "run_mode", "direction",
        "consensus_pct", "approved", "veto_code", "lot_size",
        "module_id", "details",
    )

    def __init__(
        self,
        ts: datetime,
        event_type: str,
        run_mode: str,
        symbol: str | None = None,
        direction: str | None = None,
        consensus_pct: float | None = None,
        approved: bool | None = None,
        veto_code: str | None = None,
        lot_size: float | None = None,
        module_id: str | None = None,
        details: dict | None = None,
    ) -> None:
        if ts.tzinfo is None:
            raise ValueError("AuditEntry.ts must be timezone-aware (UTC)")
        self.ts = ts
        self.event_type = event_type
        self.run_mode = run_mode
        self.symbol = symbol
        self.direction = direction
        self.consensus_pct = consensus_pct
        self.approved = approved
        self.veto_code = veto_code
        self.lot_size = lot_size
        self.module_id = module_id
        self.details = details or {}


class SQLiteAuditStore:
    """Append-only SQLite store for audit trail and kill switch log.

    Args:
        db_path: Path to the SQLite file. Use ":memory:" for tests.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._connect()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            if self._db_path != ":memory:":
                Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

            self._conn = sqlite3.connect(
                self._db_path,
                check_same_thread=False,  # We manage thread safety at app level
            )
            self._conn.row_factory = sqlite3.Row

            # WAL mode: concurrent reads while writing
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")

            self._create_schema()
            log.info("sqlite_connected", path=self._db_path)

        except sqlite3.Error as exc:
            raise AuditStoreError(
                message=f"Cannot connect to SQLite at {self._db_path!r}: {exc}",
                code="SQLITE_CONNECT_FAILED",
                context={"path": self._db_path},
            ) from exc

    def _create_schema(self) -> None:
        assert self._conn is not None
        self._conn.execute(SQLITE_CREATE_AUDIT_TABLE)
        self._conn.execute(SQLITE_CREATE_AUDIT_IDX)
        self._conn.execute(SQLITE_CREATE_KILL_SWITCH_TABLE)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            log.info("sqlite_closed", path=self._db_path)

    def __enter__(self) -> SQLiteAuditStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Audit log ─────────────────────────────────────────────────────────────

    def log_decision(self, entry: AuditEntry) -> None:
        """Append one decision record to the audit log."""
        assert self._conn is not None
        try:
            self._conn.execute(
                SQLITE_INSERT_AUDIT,
                (
                    int(entry.ts.timestamp()),
                    entry.event_type,
                    entry.symbol,
                    entry.run_mode,
                    entry.direction,
                    entry.consensus_pct,
                    int(entry.approved) if entry.approved is not None else None,
                    entry.veto_code,
                    entry.lot_size,
                    entry.module_id,
                    json.dumps(entry.details) if entry.details else None,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            raise AuditStoreError(
                message=f"Failed to write audit entry: {exc}",
                code="SQLITE_AUDIT_WRITE_FAILED",
            ) from exc

    def get_recent_decisions(self, limit: int = 100) -> list[dict]:
        """Return the most recent audit entries as plain dicts."""
        assert self._conn is not None
        try:
            rows = self._conn.execute(SQLITE_SELECT_AUDIT_RECENT, (limit,)).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise AuditStoreError(
                message=f"Failed to read audit log: {exc}",
                code="SQLITE_AUDIT_READ_FAILED",
            ) from exc

    # ── Kill switch log ───────────────────────────────────────────────────────

    def log_kill_switch(
        self,
        level: int,
        reason: str,
        activated_by: str,
        ts: datetime,
    ) -> None:
        """Record a kill switch activation or reset."""
        assert self._conn is not None
        try:
            self._conn.execute(
                SQLITE_INSERT_KILL_SWITCH,
                (int(ts.timestamp()), level, reason, activated_by),
            )
            self._conn.commit()
            log.warning(
                "kill_switch_logged",
                level=level,
                reason=reason,
                activated_by=activated_by,
            )
        except sqlite3.Error as exc:
            raise AuditStoreError(
                message=f"Failed to write kill switch log: {exc}",
                code="SQLITE_KS_WRITE_FAILED",
            ) from exc
