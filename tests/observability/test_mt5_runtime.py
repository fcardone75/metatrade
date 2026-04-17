"""Tests for MT5RuntimeReader.get_closed_deals and close_position."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_reader() -> MT5RuntimeReader:
    from metatrade.observability.mt5_runtime import MT5RuntimeReader

    reader = MT5RuntimeReader.__new__(MT5RuntimeReader)
    reader._cfg = SimpleNamespace(mt5_timeout_ms=5000, mt5_path=None)
    return reader


def _patch_init(reader, mt5_mock):
    """Patch _initialize() to return mt5_mock directly."""
    return patch.object(reader, "_initialize", return_value=mt5_mock)


def _make_deal(
    ticket: int = 1,
    entry: int = 1,
    symbol: str = "EURUSD",
    type_: int = 0,
    volume: float = 0.10,
    price: float = 1.1050,
    profit: float = 25.0,
    swap: float = -0.5,
    commission: float = -3.0,
    comment: str = "tp",
    time: int = 1_700_000_000,
    order: int = 100,
    position_id: int = 200,
) -> SimpleNamespace:
    return SimpleNamespace(
        ticket=ticket,
        order=order,
        position_id=position_id,
        symbol=symbol,
        type=type_,
        volume=volume,
        price=price,
        profit=profit,
        swap=swap,
        commission=commission,
        comment=comment,
        time=time,
        entry=entry,
    )


def _make_position(
    ticket: int = 42,
    symbol: str = "EURUSD",
    type_: int = 0,  # 0=BUY
    volume: float = 0.10,
) -> SimpleNamespace:
    return SimpleNamespace(
        ticket=ticket, symbol=symbol, type=type_, volume=volume
    )


# ── get_closed_deals ──────────────────────────────────────────────────────────

class TestGetClosedDeals:
    def test_returns_empty_when_mt5_unavailable(self) -> None:
        reader = _make_reader()
        with patch.object(reader, "_initialize", return_value=None):
            assert reader.get_closed_deals() == []

    def test_returns_empty_when_history_is_none(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.history_deals_get.return_value = None
        with _patch_init(reader, mt5):
            assert reader.get_closed_deals() == []

    def test_returns_empty_when_no_deals(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.history_deals_get.return_value = []
        with _patch_init(reader, mt5):
            assert reader.get_closed_deals() == []

    def test_filters_out_entry_in_deals(self) -> None:
        """Only entry=1 (OUT) and entry=2 (INOUT) deals are included."""
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.history_deals_get.return_value = [
            _make_deal(ticket=1, entry=0, time=100),  # ENTRY_IN → excluded
            _make_deal(ticket=2, entry=1, time=200),  # ENTRY_OUT → included
            _make_deal(ticket=3, entry=2, time=300),  # ENTRY_INOUT → included
        ]
        with _patch_init(reader, mt5):
            result = reader.get_closed_deals()
        tickets = [r["ticket"] for r in result]
        assert 1 not in tickets
        assert 2 in tickets
        assert 3 in tickets

    def test_sorted_most_recent_first(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.history_deals_get.return_value = [
            _make_deal(ticket=10, entry=1, time=1000),
            _make_deal(ticket=20, entry=1, time=3000),
            _make_deal(ticket=30, entry=1, time=2000),
        ]
        with _patch_init(reader, mt5):
            result = reader.get_closed_deals()
        assert [r["ticket"] for r in result] == [20, 30, 10]

    def test_limit_respected(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.history_deals_get.return_value = [
            _make_deal(ticket=i, entry=1, time=i) for i in range(10)
        ]
        with _patch_init(reader, mt5):
            result = reader.get_closed_deals(limit=3)
        assert len(result) == 3

    def test_deal_fields_mapped_correctly(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        deal = _make_deal(
            ticket=99, entry=1, symbol="GBPUSD", type_=1,
            volume=0.20, price=1.2700, profit=-15.0,
            swap=-1.0, commission=-2.0, comment="sl", time=1_700_000_500,
            order=555, position_id=777,
        )
        mt5.history_deals_get.return_value = [deal]
        with _patch_init(reader, mt5):
            result = reader.get_closed_deals()
        assert len(result) == 1
        r = result[0]
        assert r["ticket"] == 99
        assert r["symbol"] == "GBPUSD"
        assert r["type"] == 1
        assert r["volume"] == pytest.approx(0.20)
        assert r["price"] == pytest.approx(1.2700)
        assert r["profit"] == pytest.approx(-15.0)
        assert r["swap"] == pytest.approx(-1.0)
        assert r["commission"] == pytest.approx(-2.0)
        assert r["comment"] == "sl"
        assert r["time"] == 1_700_000_500
        assert r["order"] == 555
        assert r["position_id"] == 777

    def test_shutdown_called_even_on_empty_result(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.history_deals_get.return_value = []
        with _patch_init(reader, mt5):
            reader.get_closed_deals()
        mt5.shutdown.assert_called_once()


# ── close_position ────────────────────────────────────────────────────────────

class TestClosePosition:
    def test_returns_error_when_mt5_unavailable(self) -> None:
        reader = _make_reader()
        with patch.object(reader, "_initialize", return_value=None):
            result = reader.close_position(42)
        assert result["ok"] is False
        assert "not available" in result["comment"]

    def test_returns_error_when_position_not_found(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.positions_get.return_value = []
        with _patch_init(reader, mt5):
            result = reader.close_position(99)
        assert result["ok"] is False
        assert "not found" in result["comment"]

    def test_returns_error_when_no_tick(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        pos = _make_position(ticket=1, type_=0)
        mt5.positions_get.return_value = [pos]
        mt5.symbol_info_tick.return_value = None
        with _patch_init(reader, mt5):
            result = reader.close_position(1)
        assert result["ok"] is False
        assert "No tick" in result["comment"]

    def test_closes_buy_with_sell_order(self) -> None:
        """BUY position (type=0) must be closed with a SELL order (type=1)."""
        reader = _make_reader()
        mt5 = MagicMock()
        pos = _make_position(ticket=10, type_=0)  # BUY
        mt5.positions_get.return_value = [pos]
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.1000, ask=1.1002)
        mt5.symbol_info.return_value = SimpleNamespace(filling_mode=0)
        mt5.order_send.return_value = SimpleNamespace(retcode=10009, comment="done")
        with _patch_init(reader, mt5):
            result = reader.close_position(10)
        assert result["ok"] is True
        sent = mt5.order_send.call_args[0][0]
        assert sent["type"] == 1        # SELL
        assert sent["price"] == pytest.approx(1.1000)  # bid
        assert sent["position"] == 10

    def test_closes_sell_with_buy_order(self) -> None:
        """SELL position (type=1) must be closed with a BUY order (type=0)."""
        reader = _make_reader()
        mt5 = MagicMock()
        pos = _make_position(ticket=20, type_=1)  # SELL
        mt5.positions_get.return_value = [pos]
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.0990, ask=1.1005)
        mt5.symbol_info.return_value = SimpleNamespace(filling_mode=0)
        mt5.order_send.return_value = SimpleNamespace(retcode=10009, comment="done")
        with _patch_init(reader, mt5):
            result = reader.close_position(20)
        assert result["ok"] is True
        sent = mt5.order_send.call_args[0][0]
        assert sent["type"] == 0        # BUY
        assert sent["price"] == pytest.approx(1.1005)  # ask

    def test_filling_mode_fok_when_flag_set(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.positions_get.return_value = [_make_position(ticket=1)]
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1)
        mt5.symbol_info.return_value = SimpleNamespace(filling_mode=1)  # FOK flag
        mt5.order_send.return_value = SimpleNamespace(retcode=10009, comment="ok")
        with _patch_init(reader, mt5):
            reader.close_position(1)
        sent = mt5.order_send.call_args[0][0]
        assert sent["type_filling"] == 0  # FOK

    def test_filling_mode_ioc_when_only_ioc_flag(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.positions_get.return_value = [_make_position(ticket=1)]
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1)
        mt5.symbol_info.return_value = SimpleNamespace(filling_mode=2)  # IOC flag only
        mt5.order_send.return_value = SimpleNamespace(retcode=10009, comment="ok")
        with _patch_init(reader, mt5):
            reader.close_position(1)
        sent = mt5.order_send.call_args[0][0]
        assert sent["type_filling"] == 1  # IOC

    def test_filling_mode_return_when_no_flags(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.positions_get.return_value = [_make_position(ticket=1)]
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1)
        mt5.symbol_info.return_value = SimpleNamespace(filling_mode=0)  # no flags
        mt5.order_send.return_value = SimpleNamespace(retcode=10009, comment="ok")
        with _patch_init(reader, mt5):
            reader.close_position(1)
        sent = mt5.order_send.call_args[0][0]
        assert sent["type_filling"] == 2  # RETURN

    def test_filling_mode_return_when_symbol_info_none(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.positions_get.return_value = [_make_position(ticket=1)]
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1)
        mt5.symbol_info.return_value = None
        mt5.order_send.return_value = SimpleNamespace(retcode=10009, comment="ok")
        with _patch_init(reader, mt5):
            reader.close_position(1)
        sent = mt5.order_send.call_args[0][0]
        assert sent["type_filling"] == 2  # RETURN fallback

    def test_returns_ok_false_on_bad_retcode(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.positions_get.return_value = [_make_position(ticket=1)]
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1)
        mt5.symbol_info.return_value = SimpleNamespace(filling_mode=0)
        mt5.order_send.return_value = SimpleNamespace(retcode=10019, comment="No money")
        with _patch_init(reader, mt5):
            result = reader.close_position(1)
        assert result["ok"] is False
        assert result["retcode"] == 10019

    def test_partial_fill_retcode_is_ok(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.positions_get.return_value = [_make_position(ticket=1)]
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1)
        mt5.symbol_info.return_value = SimpleNamespace(filling_mode=0)
        mt5.order_send.return_value = SimpleNamespace(retcode=10010, comment="partial")
        with _patch_init(reader, mt5):
            result = reader.close_position(1)
        assert result["ok"] is True

    def test_returns_error_when_order_send_returns_none(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.positions_get.return_value = [_make_position(ticket=1)]
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1)
        mt5.symbol_info.return_value = SimpleNamespace(filling_mode=0)
        mt5.order_send.return_value = None
        mt5.last_error.return_value = "connection lost"
        with _patch_init(reader, mt5):
            result = reader.close_position(1)
        assert result["ok"] is False
        assert "connection lost" in result["comment"]

    def test_request_contains_required_fields(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        pos = _make_position(ticket=55, symbol="GBPUSD", type_=0, volume=0.30)
        mt5.positions_get.return_value = [pos]
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.2700, ask=1.2703)
        mt5.symbol_info.return_value = SimpleNamespace(filling_mode=0)
        mt5.order_send.return_value = SimpleNamespace(retcode=10009, comment="ok")
        with _patch_init(reader, mt5):
            reader.close_position(55)
        sent = mt5.order_send.call_args[0][0]
        assert sent["action"] == 1         # TRADE_ACTION_DEAL
        assert sent["symbol"] == "GBPUSD"
        assert sent["volume"] == pytest.approx(0.30)
        assert sent["position"] == 55
        assert sent["deviation"] == 20
        assert sent["comment"] == "dashboard close"

    def test_shutdown_called_on_success(self) -> None:
        reader = _make_reader()
        mt5 = MagicMock()
        mt5.positions_get.return_value = [_make_position(ticket=1)]
        mt5.symbol_info_tick.return_value = SimpleNamespace(bid=1.1, ask=1.1)
        mt5.symbol_info.return_value = SimpleNamespace(filling_mode=0)
        mt5.order_send.return_value = SimpleNamespace(retcode=10009, comment="ok")
        with _patch_init(reader, mt5):
            reader.close_position(1)
        mt5.shutdown.assert_called_once()
