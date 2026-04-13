"""Database schema definitions for both DuckDB and SQLite stores.

Design choices:
- Timestamps stored as BIGINT (Unix seconds UTC) — no timezone ambiguity,
  faster than VARCHAR, and DuckDB/SQLite both handle it natively.
- Prices stored as DOUBLE in DuckDB. float64 has ~15 significant digits,
  sufficient for FX prices (max 8 decimal places). Converted to Decimal on read.
- DuckDB uses a composite PRIMARY KEY to prevent duplicates on upsert.
- SQLite uses INTEGER PRIMARY KEY (rowid alias) for the audit log — append-only.
"""

# ── DuckDB ─────────────────────────────────────────────────────────────────────

DUCKDB_CREATE_BARS_TABLE = """
CREATE TABLE IF NOT EXISTS bars (
    symbol      VARCHAR     NOT NULL,
    timeframe   VARCHAR     NOT NULL,
    ts          BIGINT      NOT NULL,
    open        DOUBLE      NOT NULL,
    high        DOUBLE      NOT NULL,
    low         DOUBLE      NOT NULL,
    close       DOUBLE      NOT NULL,
    volume      DOUBLE      NOT NULL,
    spread      DOUBLE,
    PRIMARY KEY (symbol, timeframe, ts)
)
"""

DUCKDB_CREATE_BARS_IDX = """
CREATE INDEX IF NOT EXISTS idx_bars_symbol_tf_ts
ON bars (symbol, timeframe, ts DESC)
"""

DUCKDB_UPSERT_BARS = """
INSERT OR REPLACE INTO bars
    (symbol, timeframe, ts, open, high, low, close, volume, spread)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

DUCKDB_SELECT_BARS = """
SELECT symbol, timeframe, ts, open, high, low, close, volume, spread
FROM   bars
WHERE  symbol = ?
  AND  timeframe = ?
  AND  ts >= ?
  AND  ts <= ?
ORDER  BY ts ASC
"""

DUCKDB_SELECT_LATEST_TS = """
SELECT MAX(ts) FROM bars WHERE symbol = ? AND timeframe = ?
"""

DUCKDB_SELECT_OLDEST_TS = """
SELECT MIN(ts) FROM bars WHERE symbol = ? AND timeframe = ?
"""

DUCKDB_COUNT_BARS = """
SELECT COUNT(*) FROM bars WHERE symbol = ? AND timeframe = ?
"""

DUCKDB_DELETE_BARS = """
DELETE FROM bars
WHERE  symbol = ?
  AND  timeframe = ?
  AND  ts >= ?
  AND  ts <= ?
"""

# ── SQLite ─────────────────────────────────────────────────────────────────────

SQLITE_CREATE_AUDIT_TABLE = """
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              BIGINT      NOT NULL,
    event_type      VARCHAR     NOT NULL,
    symbol          VARCHAR,
    run_mode        VARCHAR     NOT NULL,
    direction       VARCHAR,
    consensus_pct   REAL,
    approved        INTEGER,
    veto_code       VARCHAR,
    lot_size        REAL,
    module_id       VARCHAR,
    details         TEXT        -- JSON blob for extra context
)
"""

SQLITE_CREATE_AUDIT_IDX = """
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts DESC)
"""

SQLITE_INSERT_AUDIT = """
INSERT INTO audit_log
    (ts, event_type, symbol, run_mode, direction,
     consensus_pct, approved, veto_code, lot_size, module_id, details)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

SQLITE_SELECT_AUDIT_RECENT = """
SELECT * FROM audit_log ORDER BY ts DESC LIMIT ?
"""

SQLITE_CREATE_KILL_SWITCH_TABLE = """
CREATE TABLE IF NOT EXISTS kill_switch_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              BIGINT  NOT NULL,
    level           INTEGER NOT NULL,
    reason          TEXT    NOT NULL,
    activated_by    VARCHAR NOT NULL
)
"""

SQLITE_INSERT_KILL_SWITCH = """
INSERT INTO kill_switch_log (ts, level, reason, activated_by)
VALUES (?, ?, ?, ?)
"""
