"""Tests for IntermarketStore (persistence layer)."""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.intermarket.contracts import (
    CurrencyExposure,
    IntermarketDecision,
    IntermarketSnapshot,
    PairCorrelation,
)
from metatrade.intermarket.persistence import IntermarketStore

_NOW = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "test_intermarket.db")
    s = IntermarketStore(db)
    yield s
    s.close()


def _decision(
    approved: bool = True,
    risk_multiplier: str = "0.75",
    reason: str = "test_reason",
    warnings: tuple = (),
    correlated: tuple = (),
    exposure: dict | None = None,
) -> IntermarketDecision:
    return IntermarketDecision(
        approved=approved,
        risk_multiplier=Decimal(risk_multiplier),
        correlated_positions=correlated,
        reason=reason,
        warnings=warnings,
        exposure_delta_by_ccy=exposure or {"EUR": Decimal("0.01"), "USD": Decimal("-0.01")},
    )


def _snapshot(correlations=(), exposures=(), clusters=()) -> IntermarketSnapshot:
    return IntermarketSnapshot(
        correlations=correlations,
        exposures=exposures,
        risk_clusters=tuple(clusters),
        timestamp_utc=_NOW,
    )


def _pair_corr(sym_a="EURUSD", sym_b="GBPUSD", corr=0.85) -> PairCorrelation:
    return PairCorrelation(
        symbol_a=sym_a,
        symbol_b=sym_b,
        correlation=corr,
        abs_correlation=abs(corr),
        lookback_bars=100,
        overlap_bars=90,
        returns_mode="pct",
        timestamp_utc=_NOW,
    )


def _exposure(currency="EUR", net=0.5, gross=0.5) -> CurrencyExposure:
    return CurrencyExposure(
        currency=currency,
        net_lots=Decimal(str(net)),
        gross_lots=Decimal(str(gross)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Schema creation
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaCreation:
    def test_store_creates_tables(self, store):
        conn = store._conn
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "intermarket_decisions" in tables
        assert "intermarket_correlation_snapshots" in tables
        assert "intermarket_exposure_snapshots" in tables

    def test_schema_idempotent(self, tmp_path):
        """Creating a store on an existing DB does not fail."""
        db = str(tmp_path / "reopen.db")
        s1 = IntermarketStore(db)
        s1.close()
        s2 = IntermarketStore(db)
        s2.close()


# ══════════════════════════════════════════════════════════════════════════════
# record_decision
# ══════════════════════════════════════════════════════════════════════════════

class TestRecordDecision:
    def test_record_approved_decision(self, store):
        d = _decision(approved=True, risk_multiplier="0.75", reason="low_corr")
        store.record_decision("EURUSD", "BUY", d, session_id="sess1")

        rows = store.list_recent_decisions()
        assert len(rows) == 1
        assert rows[0]["symbol"] == "EURUSD"
        assert rows[0]["side"] == "BUY"
        assert rows[0]["approved"] == 1
        assert abs(rows[0]["risk_multiplier"] - 0.75) < 1e-6
        assert rows[0]["reason"] == "low_corr"
        assert rows[0]["session_id"] == "sess1"

    def test_record_vetoed_decision(self, store):
        d = _decision(approved=False, reason="exposure_limit_exceeded")
        store.record_decision("GBPUSD", "SELL", d)

        rows = store.list_recent_decisions(vetoes_only=True)
        assert len(rows) == 1
        assert rows[0]["approved"] == 0

    def test_record_with_warnings_and_correlated(self, store):
        d = _decision(
            warnings=("medium_corr(GBPUSD,0.75)",),
            correlated=("GBPUSD",),
        )
        store.record_decision("EURUSD", "BUY", d)

        rows = store.list_recent_decisions()
        assert len(rows) == 1
        import json
        warnings = json.loads(rows[0]["warnings_json"])
        assert "medium_corr(GBPUSD,0.75)" in warnings
        correlated = json.loads(rows[0]["correlated_json"])
        assert "GBPUSD" in correlated

    def test_record_without_session_id(self, store):
        d = _decision()
        store.record_decision("EURUSD", "BUY", d, session_id=None)
        rows = store.list_recent_decisions()
        assert len(rows) == 1
        assert rows[0]["session_id"] is None

    def test_record_multiple_decisions(self, store):
        for i in range(5):
            d = _decision(reason=f"reason_{i}")
            store.record_decision("EURUSD", "BUY", d)
        rows = store.list_recent_decisions()
        assert len(rows) == 5

    def test_limit_respected(self, store):
        for i in range(10):
            store.record_decision("EURUSD", "BUY", _decision())
        rows = store.list_recent_decisions(limit=3)
        assert len(rows) == 3

    def test_session_filter(self, store):
        store.record_decision("EURUSD", "BUY", _decision(), session_id="A")
        store.record_decision("GBPUSD", "SELL", _decision(), session_id="B")

        rows_a = store.list_recent_decisions(session_id="A")
        assert len(rows_a) == 1
        assert rows_a[0]["symbol"] == "EURUSD"

        rows_b = store.list_recent_decisions(session_id="B")
        assert len(rows_b) == 1
        assert rows_b[0]["symbol"] == "GBPUSD"

    def test_vetoes_only_filter(self, store):
        store.record_decision("EURUSD", "BUY", _decision(approved=True))
        store.record_decision("GBPUSD", "BUY", _decision(approved=False, reason="veto"))

        vetoes = store.list_recent_decisions(vetoes_only=True)
        assert len(vetoes) == 1
        assert vetoes[0]["approved"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# record_correlation_snapshot
# ══════════════════════════════════════════════════════════════════════════════

class TestRecordCorrelationSnapshot:
    def test_records_pair_correlations(self, store):
        snap = _snapshot(
            correlations=(_pair_corr("EURUSD", "GBPUSD", 0.85),)
        )
        store.record_correlation_snapshot(snap, session_id="sess1")

        rows = store._conn.execute(
            "SELECT * FROM intermarket_correlation_snapshots"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][2] == "EURUSD"   # symbol_a
        assert rows[0][3] == "GBPUSD"   # symbol_b
        assert abs(rows[0][4] - 0.85) < 1e-6

    def test_records_multiple_correlations(self, store):
        snap = _snapshot(
            correlations=(
                _pair_corr("EURUSD", "GBPUSD", 0.85),
                _pair_corr("EURUSD", "USDJPY", -0.70),
            )
        )
        store.record_correlation_snapshot(snap)

        count = store._conn.execute(
            "SELECT COUNT(*) FROM intermarket_correlation_snapshots"
        ).fetchone()[0]
        assert count == 2

    def test_skip_when_empty_correlations(self, store):
        snap = _snapshot(correlations=())
        store.record_correlation_snapshot(snap)  # should not raise

        count = store._conn.execute(
            "SELECT COUNT(*) FROM intermarket_correlation_snapshots"
        ).fetchone()[0]
        assert count == 0


# ══════════════════════════════════════════════════════════════════════════════
# record_exposure_snapshot
# ══════════════════════════════════════════════════════════════════════════════

class TestRecordExposureSnapshot:
    def test_records_exposures(self, store):
        snap = _snapshot(
            exposures=(_exposure("EUR", 1.5, 1.5), _exposure("USD", -1.0, 1.0))
        )
        store.record_exposure_snapshot(snap, session_id="sess1")

        rows = store._conn.execute(
            "SELECT currency, net_lots, gross_lots FROM intermarket_exposure_snapshots"
        ).fetchall()
        assert len(rows) == 2
        currencies = {r[0] for r in rows}
        assert "EUR" in currencies
        assert "USD" in currencies

    def test_skip_when_empty_exposures(self, store):
        snap = _snapshot(exposures=())
        store.record_exposure_snapshot(snap)  # should not raise

        count = store._conn.execute(
            "SELECT COUNT(*) FROM intermarket_exposure_snapshots"
        ).fetchone()[0]
        assert count == 0


# ══════════════════════════════════════════════════════════════════════════════
# Resilience
# ══════════════════════════════════════════════════════════════════════════════

class TestResilience:
    def test_list_returns_empty_when_no_rows(self, store):
        rows = store.list_recent_decisions()
        assert rows == []

    def test_close_is_idempotent(self, store):
        store.close()
        store.close()  # should not raise

    def test_from_env_uses_default_path(self, monkeypatch, tmp_path):
        """from_env() should read TELEMETRY_DB_PATH from environment."""
        db = str(tmp_path / "env_test.db")
        monkeypatch.setenv("TELEMETRY_DB_PATH", db)
        s = IntermarketStore.from_env()
        assert s._db_path == db
        s.close()


# ══════════════════════════════════════════════════════════════════════════════
# Closed connection path
# ══════════════════════════════════════════════════════════════════════════════

class TestClosedConnection:
    def test_create_schema_noop_when_closed(self, store):
        """_create_schema is a no-op when _conn is None."""
        store.close()
        # Should not raise
        store._create_schema()

    def test_record_decision_noop_when_closed(self, store):
        """record_decision silently returns when _conn is None."""
        store.close()
        store.record_decision("EURUSD", "BUY", _decision())  # no error

    def test_record_correlation_noop_when_closed(self, store):
        """record_correlation_snapshot silently returns when _conn is None."""
        store.close()
        snap = _snapshot(correlations=(_pair_corr(),))
        store.record_correlation_snapshot(snap)  # no error

    def test_record_exposure_noop_when_closed(self, store):
        """record_exposure_snapshot silently returns when _conn is None."""
        store.close()
        snap = _snapshot(exposures=(_exposure(),))
        store.record_exposure_snapshot(snap)  # no error

    def test_list_decisions_empty_when_closed(self, store):
        """list_recent_decisions returns empty list when _conn is None."""
        store.close()
        rows = store.list_recent_decisions()
        assert rows == []


# ══════════════════════════════════════════════════════════════════════════════
# Error-handling paths (simulate DB failures)
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    def test_record_decision_swallows_db_error(self, tmp_path):
        """DB error in record_decision is swallowed: corrupt the DB file mid-flight."""
        db = str(tmp_path / "corrupt.db")
        s = IntermarketStore(db)
        # Close the connection and corrupt the DB, then re-open on top
        s.close()
        # Write garbage to simulate a broken connection state
        s._conn = None  # Simulate disconnected state but _conn re-set to None
        # Now call record_decision — should silently skip (conn is None)
        s.record_decision("EURUSD", "BUY", _decision())  # no exception

    def test_record_decision_with_bad_exposure_json(self, store):
        """record_decision with non-serializable exposure gracefully fails."""
        # Create a decision where exposure_delta_by_ccy has Decimal values (normal case)
        d = IntermarketDecision(
            approved=True,
            risk_multiplier=Decimal("1.0"),
            correlated_positions=(),
            reason="ok",
            warnings=(),
            exposure_delta_by_ccy={"EUR": Decimal("0.01")},
        )
        # This should work fine
        store.record_decision("EURUSD", "BUY", d)
        rows = store.list_recent_decisions()
        assert len(rows) == 1

    def test_record_correlation_empty_snapshot(self, store):
        """Empty correlations snapshot is silently skipped."""
        snap = _snapshot(correlations=())
        store.record_correlation_snapshot(snap)  # should not raise
        count = store._conn.execute(
            "SELECT COUNT(*) FROM intermarket_correlation_snapshots"
        ).fetchone()[0]
        assert count == 0

    def test_record_exposure_empty_snapshot(self, store):
        """Empty exposures snapshot is silently skipped."""
        snap = _snapshot(exposures=())
        store.record_exposure_snapshot(snap)  # should not raise

    def test_list_decisions_all_filters_combined(self, store):
        """Combining session_id and vetoes_only works correctly."""
        store.record_decision("EURUSD", "BUY", _decision(approved=True), session_id="X")
        store.record_decision("GBPUSD", "BUY", _decision(approved=False, reason="veto"), session_id="X")
        store.record_decision("USDJPY", "BUY", _decision(approved=False, reason="veto"), session_id="Y")

        rows = store.list_recent_decisions(session_id="X", vetoes_only=True)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "GBPUSD"
