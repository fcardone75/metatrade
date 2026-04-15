"""Observability storage for dashboard/API telemetry."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from metatrade.core.contracts.account import AccountState
from metatrade.core.contracts.consensus import ConsensusResult
from metatrade.core.contracts.risk import RiskDecision
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.log import get_logger
from metatrade.market_data.config import MarketDataConfig

log = get_logger(__name__)


CREATE_SESSIONS_TABLE = """
CREATE TABLE IF NOT EXISTS dashboard_sessions (
    session_id       TEXT PRIMARY KEY,
    started_at       BIGINT NOT NULL,
    ended_at         BIGINT,
    run_mode         TEXT NOT NULL,
    symbol           TEXT,
    timeframe        TEXT,
    ml_enabled       INTEGER NOT NULL DEFAULT 0,
    model_version    TEXT,
    status           TEXT NOT NULL DEFAULT 'running',
    details          TEXT
)
"""

CREATE_ACCOUNT_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS dashboard_account_snapshots (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id           TEXT NOT NULL,
    ts                   BIGINT NOT NULL,
    symbol               TEXT,
    timeframe            TEXT,
    run_mode             TEXT NOT NULL,
    balance              REAL NOT NULL,
    equity               REAL NOT NULL,
    free_margin          REAL NOT NULL,
    used_margin          REAL NOT NULL,
    unrealized_pnl       REAL NOT NULL,
    currency             TEXT NOT NULL,
    open_positions_count INTEGER NOT NULL DEFAULT 0,
    details              TEXT
)
"""

CREATE_DECISIONS_TABLE = """
CREATE TABLE IF NOT EXISTS dashboard_decisions (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id           TEXT NOT NULL,
    ts                   BIGINT NOT NULL,
    symbol               TEXT NOT NULL,
    timeframe            TEXT,
    run_mode             TEXT NOT NULL,
    direction            TEXT NOT NULL,
    actionable           INTEGER NOT NULL,
    approved             INTEGER,
    veto_code            TEXT,
    veto_reason          TEXT,
    buy_pct              REAL NOT NULL,
    sell_pct             REAL NOT NULL,
    hold_pct             REAL NOT NULL,
    aggregate_confidence REAL NOT NULL,
    threshold_used       REAL NOT NULL,
    lot_size             REAL,
    risk_pct             REAL,
    model_version        TEXT,
    explanation          TEXT,
    details              TEXT
)
"""

CREATE_ORDER_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS dashboard_order_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT,
    ts              BIGINT NOT NULL,
    run_mode        TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    symbol          TEXT,
    side            TEXT,
    order_id        TEXT,
    broker_order_id TEXT,
    status          TEXT,
    lot_size        REAL,
    price           REAL,
    details         TEXT
)
"""

CREATE_TRAINING_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS dashboard_training_runs (
    training_id         TEXT PRIMARY KEY,
    started_at          BIGINT NOT NULL,
    completed_at        BIGINT,
    status              TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    timeframe           TEXT NOT NULL,
    source              TEXT NOT NULL,
    bars_requested      INTEGER,
    bars_fetched        INTEGER,
    train_window        INTEGER,
    test_window         INTEGER,
    step                INTEGER,
    forward_bars        INTEGER,
    max_iter            INTEGER,
    max_depth           INTEGER,
    min_accuracy        REAL,
    mean_test_accuracy  REAL,
    best_test_accuracy  REAL,
    best_fold           INTEGER,
    model_version       TEXT,
    details             TEXT
)
"""

CREATE_MODEL_ARTIFACTS_TABLE = """
CREATE TABLE IF NOT EXISTS dashboard_model_artifacts (
    version             TEXT PRIMARY KEY,
    symbol              TEXT NOT NULL,
    timeframe           TEXT,
    created_at          BIGINT NOT NULL,
    source              TEXT,
    train_bars          INTEGER,
    n_folds             INTEGER,
    mean_test_accuracy  REAL,
    best_test_accuracy  REAL,
    is_active           INTEGER NOT NULL DEFAULT 0,
    artifact_path       TEXT NOT NULL,
    details             TEXT
)
"""

CREATE_MODULE_THRESHOLDS_TABLE = """
CREATE TABLE IF NOT EXISTS module_thresholds (
    module_id       TEXT PRIMARY KEY,
    threshold       REAL NOT NULL,
    eval_count      INTEGER NOT NULL DEFAULT 0,
    correct_count   INTEGER NOT NULL DEFAULT 0,
    mean_score      REAL NOT NULL DEFAULT 0.5,
    updated_at      BIGINT NOT NULL
)
"""

CREATE_SESSION_IDX = "CREATE INDEX IF NOT EXISTS idx_dashboard_sessions_started_at ON dashboard_sessions(started_at DESC)"
CREATE_ACCOUNT_IDX = "CREATE INDEX IF NOT EXISTS idx_dashboard_account_ts ON dashboard_account_snapshots(ts DESC)"
CREATE_DECISION_IDX = "CREATE INDEX IF NOT EXISTS idx_dashboard_decisions_ts ON dashboard_decisions(ts DESC)"
CREATE_ORDER_IDX = "CREATE INDEX IF NOT EXISTS idx_dashboard_order_events_ts ON dashboard_order_events(ts DESC)"
CREATE_TRAINING_IDX = "CREATE INDEX IF NOT EXISTS idx_dashboard_training_completed_at ON dashboard_training_runs(completed_at DESC)"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts(value: datetime) -> int:
    return int(value.timestamp())


def _serialize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {k: _serialize(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(v) for v in value]
    return value


class TelemetryStore:
    """SQLite-backed telemetry storage for the dashboard."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._connect()

    @classmethod
    def from_env(cls) -> "TelemetryStore":
        cfg = MarketDataConfig.load()
        return cls(cfg.sqlite_path)

    def _connect(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()
        log.info("telemetry_connected", path=self._db_path)

    def _create_schema(self) -> None:
        assert self._conn is not None
        statements = [
            CREATE_SESSIONS_TABLE,
            CREATE_ACCOUNT_SNAPSHOTS_TABLE,
            CREATE_DECISIONS_TABLE,
            CREATE_ORDER_EVENTS_TABLE,
            CREATE_TRAINING_RUNS_TABLE,
            CREATE_MODEL_ARTIFACTS_TABLE,
            CREATE_MODULE_THRESHOLDS_TABLE,
            CREATE_SESSION_IDX,
            CREATE_ACCOUNT_IDX,
            CREATE_DECISION_IDX,
            CREATE_ORDER_IDX,
            CREATE_TRAINING_IDX,
        ]
        for statement in statements:
            self._conn.execute(statement)
        self._conn.commit()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def start_session(
        self,
        *,
        run_mode: str,
        symbol: str,
        timeframe: str,
        ml_enabled: bool,
        model_version: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> str:
        session_id = f"{run_mode.lower()}-{uuid.uuid4().hex[:12]}"
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO dashboard_sessions
                (session_id, started_at, run_mode, symbol, timeframe, ml_enabled, model_version, status, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                _ts(_utc_now()),
                run_mode,
                symbol,
                timeframe,
                int(ml_enabled),
                model_version,
                "running",
                json.dumps(_serialize(details or {})),
            ),
        )
        self._conn.commit()
        return session_id

    def finish_session(
        self,
        session_id: str,
        *,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            UPDATE dashboard_sessions
            SET ended_at = ?, status = ?, details = COALESCE(?, details)
            WHERE session_id = ?
            """,
            (
                _ts(_utc_now()),
                status,
                json.dumps(_serialize(details)) if details is not None else None,
                session_id,
            ),
        )
        self._conn.commit()

    def record_account_snapshot(
        self,
        *,
        session_id: str,
        symbol: str,
        timeframe: str | None,
        account: AccountState,
        details: dict[str, Any] | None = None,
    ) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO dashboard_account_snapshots
                (session_id, ts, symbol, timeframe, run_mode, balance, equity, free_margin, used_margin,
                 unrealized_pnl, currency, open_positions_count, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                _ts(account.timestamp_utc),
                symbol,
                timeframe,
                account.run_mode.value,
                float(account.balance),
                float(account.equity),
                float(account.free_margin),
                float(account.used_margin),
                float(account.unrealized_pnl),
                account.currency,
                account.open_positions_count,
                json.dumps(_serialize(details or {})),
            ),
        )
        self._conn.commit()

    def record_decision(
        self,
        *,
        session_id: str,
        symbol: str,
        timeframe: str | None,
        run_mode: str,
        signals: list[AnalysisSignal],
        consensus_result: ConsensusResult,
        decision: RiskDecision | None,
    ) -> None:
        assert self._conn is not None
        model_version = self._extract_model_version(signals)
        lot_size = None
        risk_pct = None
        veto_code = None
        veto_reason = None
        approved = None
        if decision is not None:
            approved = int(decision.approved)
            if decision.approved and decision.position_size is not None:
                lot_size = float(decision.position_size.lot_size)
                risk_pct = decision.position_size.risk_pct
            elif decision.veto is not None:
                veto_code = decision.veto.veto_code
                veto_reason = decision.veto.reason

        details = {
            "signals": [
                {
                    "module_id": sig.module_id,
                    "direction": sig.direction.value,
                    "confidence": sig.confidence,
                    "reason": sig.reason,
                    "metadata": _serialize(sig.metadata),
                }
                for sig in signals
            ],
            "vote_records": [
                {
                    "module_id": vr.signal.module_id,
                    "direction": vr.signal.direction.value,
                    "confidence": vr.signal.confidence,
                    "weight": vr.weight,
                    "weighted_score": vr.weighted_score,
                }
                for vr in consensus_result.vote_records
            ],
        }

        self._conn.execute(
            """
            INSERT INTO dashboard_decisions
                (session_id, ts, symbol, timeframe, run_mode, direction, actionable, approved, veto_code, veto_reason,
                 buy_pct, sell_pct, hold_pct, aggregate_confidence, threshold_used, lot_size, risk_pct, model_version,
                 explanation, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                _ts(consensus_result.timestamp_utc),
                symbol,
                timeframe,
                run_mode,
                consensus_result.final_direction.value,
                int(consensus_result.is_actionable),
                approved,
                veto_code,
                veto_reason,
                consensus_result.buy_pct,
                consensus_result.sell_pct,
                consensus_result.hold_pct,
                consensus_result.aggregate_confidence,
                consensus_result.threshold_used,
                lot_size,
                risk_pct,
                model_version,
                consensus_result.explanation,
                json.dumps(_serialize(details)),
            ),
        )
        self._conn.commit()

    def record_order_event(
        self,
        *,
        session_id: str | None,
        run_mode: str,
        event_type: str,
        ts: datetime,
        symbol: str | None = None,
        side: str | None = None,
        order_id: str | None = None,
        broker_order_id: str | None = None,
        status: str | None = None,
        lot_size: float | None = None,
        price: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO dashboard_order_events
                (session_id, ts, run_mode, event_type, symbol, side, order_id, broker_order_id, status, lot_size, price, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                _ts(ts),
                run_mode,
                event_type,
                symbol,
                side,
                order_id,
                broker_order_id,
                status,
                lot_size,
                price,
                json.dumps(_serialize(details or {})),
            ),
        )
        self._conn.commit()

    def record_training_run(
        self,
        *,
        symbol: str,
        timeframe: str,
        source: str,
        bars_requested: int,
        bars_fetched: int,
        train_window: int,
        test_window: int,
        step: int,
        forward_bars: int,
        max_iter: int,
        max_depth: int,
        min_accuracy: float,
        mean_test_accuracy: float,
        best_test_accuracy: float,
        best_fold: int,
        model_version: str | None,
        status: str,
        details: dict[str, Any] | None = None,
    ) -> str:
        training_id = f"train-{uuid.uuid4().hex[:12]}"
        now = _utc_now()
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO dashboard_training_runs
                (training_id, started_at, completed_at, status, symbol, timeframe, source, bars_requested, bars_fetched,
                 train_window, test_window, step, forward_bars, max_iter, max_depth, min_accuracy,
                 mean_test_accuracy, best_test_accuracy, best_fold, model_version, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                training_id,
                _ts(now),
                _ts(now),
                status,
                symbol,
                timeframe,
                source,
                bars_requested,
                bars_fetched,
                train_window,
                test_window,
                step,
                forward_bars,
                max_iter,
                max_depth,
                min_accuracy,
                mean_test_accuracy,
                best_test_accuracy,
                best_fold,
                model_version,
                json.dumps(_serialize(details or {})),
            ),
        )
        self._conn.commit()
        return training_id

    def record_model_artifact(
        self,
        *,
        version: str,
        symbol: str,
        timeframe: str,
        source: str,
        train_bars: int,
        n_folds: int,
        mean_test_accuracy: float,
        best_test_accuracy: float,
        artifact_path: str,
        is_active: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        assert self._conn is not None
        if is_active:
            self._conn.execute("UPDATE dashboard_model_artifacts SET is_active = 0 WHERE symbol = ?", (symbol,))
        self._conn.execute(
            """
            INSERT OR REPLACE INTO dashboard_model_artifacts
                (version, symbol, timeframe, created_at, source, train_bars, n_folds, mean_test_accuracy,
                 best_test_accuracy, is_active, artifact_path, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version,
                symbol,
                timeframe,
                _ts(_utc_now()),
                source,
                train_bars,
                n_folds,
                mean_test_accuracy,
                best_test_accuracy,
                int(is_active),
                artifact_path,
                json.dumps(_serialize(details or {})),
            ),
        )
        self._conn.commit()

    def list_sessions(self, *, limit: int = 20, active_only: bool = False) -> list[dict[str, Any]]:
        assert self._conn is not None
        query = "SELECT * FROM dashboard_sessions"
        if active_only:
            query += " WHERE ended_at IS NULL"
        query += " ORDER BY started_at DESC LIMIT ?"
        return [dict(row) for row in self._conn.execute(query, (limit,)).fetchall()]

    def get_latest_account_snapshot(self, *, session_id: str | None = None) -> dict[str, Any] | None:
        rows = self.list_account_snapshots(limit=1, session_id=session_id)
        return rows[0] if rows else None

    def list_account_snapshots(
        self,
        *,
        limit: int = 200,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        assert self._conn is not None
        query = "SELECT * FROM dashboard_account_snapshots"
        params: list[Any] = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self._conn.execute(query, tuple(params)).fetchall()]

    def list_recent_decisions(
        self,
        *,
        limit: int = 100,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        assert self._conn is not None
        query = "SELECT * FROM dashboard_decisions"
        params: list[Any] = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self._conn.execute(query, tuple(params)).fetchall()]

    def list_order_events(
        self,
        *,
        limit: int = 100,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        assert self._conn is not None
        query = "SELECT * FROM dashboard_order_events"
        params: list[Any] = []
        if session_id:
            query += " WHERE session_id = ?"
            params.append(session_id)
        query += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self._conn.execute(query, tuple(params)).fetchall()]

    def list_training_runs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM dashboard_training_runs ORDER BY completed_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def latest_training_run(self) -> dict[str, Any] | None:
        runs = self.list_training_runs(limit=1)
        return runs[0] if runs else None

    def list_model_artifacts(self, *, limit: int = 20) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM dashboard_model_artifacts ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    # ── Adaptive thresholds ───────────────────────────────────────────────────

    def upsert_module_threshold(
        self,
        module_id: str,
        threshold: float,
        eval_count: int,
        correct_count: int,
        mean_score: float,
    ) -> None:
        """Insert or update the adaptive threshold record for a module."""
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO module_thresholds
                (module_id, threshold, eval_count, correct_count, mean_score, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(module_id) DO UPDATE SET
                threshold     = excluded.threshold,
                eval_count    = excluded.eval_count,
                correct_count = excluded.correct_count,
                mean_score    = excluded.mean_score,
                updated_at    = excluded.updated_at
            """,
            (
                module_id,
                threshold,
                eval_count,
                correct_count,
                mean_score,
                _ts(_utc_now()),
            ),
        )
        self._conn.commit()

    def list_module_thresholds(self) -> list[dict[str, Any]]:
        """Return all module threshold records ordered by module_id."""
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT * FROM module_thresholds ORDER BY module_id ASC"
        ).fetchall()
        return [dict(row) for row in rows]

    def _extract_model_version(self, signals: list[AnalysisSignal]) -> str | None:
        for signal in signals:
            if signal.module_id.startswith("ml_"):
                version = signal.metadata.get("model_version")
                if isinstance(version, str) and version:
                    return version
        return None
