"""Tests for SQLite audit store."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from metatrade.market_data.storage.sqlite_store import AuditEntry, SQLiteAuditStore

UTC = timezone.utc
T0 = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)


@pytest.fixture
def store() -> SQLiteAuditStore:
    s = SQLiteAuditStore(":memory:")
    yield s
    s.close()


class TestAuditEntry:
    def test_valid_construction(self) -> None:
        entry = AuditEntry(ts=T0, event_type="SIGNAL", run_mode="PAPER")
        assert entry.event_type == "SIGNAL"

    def test_naive_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            AuditEntry(ts=datetime(2024, 1, 1), event_type="X", run_mode="PAPER")


class TestSQLiteAuditStore:
    def test_opens_in_memory(self, store: SQLiteAuditStore) -> None:
        assert store is not None

    def test_context_manager(self) -> None:
        with SQLiteAuditStore(":memory:") as s:
            s.log_decision(AuditEntry(ts=T0, event_type="TEST", run_mode="PAPER"))

    def test_close_idempotent(self) -> None:
        s = SQLiteAuditStore(":memory:")
        s.close()
        s.close()

    def test_log_and_retrieve_decision(self, store: SQLiteAuditStore) -> None:
        entry = AuditEntry(
            ts=T0,
            event_type="ORDER_SUBMITTED",
            run_mode="PAPER",
            symbol="EURUSD",
            direction="BUY",
            consensus_pct=0.70,
            approved=True,
            lot_size=0.10,
        )
        store.log_decision(entry)
        rows = store.get_recent_decisions(10)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "ORDER_SUBMITTED"
        assert rows[0]["symbol"] == "EURUSD"
        assert rows[0]["direction"] == "BUY"
        assert rows[0]["approved"] == 1

    def test_log_multiple_decisions(self, store: SQLiteAuditStore) -> None:
        for i in range(5):
            store.log_decision(AuditEntry(ts=T0, event_type=f"EVT_{i}", run_mode="PAPER"))
        rows = store.get_recent_decisions(10)
        assert len(rows) == 5

    def test_get_recent_respects_limit(self, store: SQLiteAuditStore) -> None:
        for _ in range(10):
            store.log_decision(AuditEntry(ts=T0, event_type="X", run_mode="PAPER"))
        rows = store.get_recent_decisions(3)
        assert len(rows) == 3

    def test_log_decision_with_details(self, store: SQLiteAuditStore) -> None:
        entry = AuditEntry(
            ts=T0,
            event_type="RISK_VETO",
            run_mode="LIVE",
            details={"veto_code": "MAX_POSITIONS", "current": 5},
        )
        store.log_decision(entry)
        rows = store.get_recent_decisions(1)
        assert rows[0]["details"] is not None
        assert "MAX_POSITIONS" in rows[0]["details"]

    def test_log_veto_decision(self, store: SQLiteAuditStore) -> None:
        entry = AuditEntry(
            ts=T0,
            event_type="RISK_VETO",
            run_mode="PAPER",
            symbol="GBPUSD",
            approved=False,
            veto_code="SPREAD_TOO_WIDE",
        )
        store.log_decision(entry)
        rows = store.get_recent_decisions(1)
        assert rows[0]["veto_code"] == "SPREAD_TOO_WIDE"
        assert rows[0]["approved"] == 0

    def test_log_kill_switch(self, store: SQLiteAuditStore) -> None:
        store.log_kill_switch(
            level=3,
            reason="daily loss limit hit",
            activated_by="risk_manager",
            ts=T0,
        )

    def test_empty_store_returns_empty_list(self, store: SQLiteAuditStore) -> None:
        assert store.get_recent_decisions(10) == []
