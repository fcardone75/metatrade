"""Tests for MT5BrokerAdapter position management methods.

All tests run offline (no real MT5 needed) via unittest.mock.patch.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from metatrade.broker.mt5_adapter import (
    MT5BrokerAdapter,
    _MT5_ORDER_TYPE_BUY,
    _MT5_ORDER_TYPE_SELL,
    _MT5_RETCODE_DONE,
    _MT5_RETCODE_CHECK_OK,
    _MT5_TRADE_ACTION_DEAL,
    _MT5_TRADE_ACTION_SLTP,
)
from metatrade.core.contracts.order import Order
from metatrade.core.enums import OrderSide, OrderType


def _adapter() -> MT5BrokerAdapter:
    a = MT5BrokerAdapter(login=1, password="x", server="Demo")
    a._connected = True
    return a


def _mt5_position(
    ticket: int = 101,
    symbol: str = "EURUSD",
    volume: float = 0.10,
    pos_type: int = 0,   # 0=buy, 1=sell
    price_open: float = 1.10000,
    sl: float = 1.09800,
    tp: float = 1.10200,
    profit: float = 5.0,
    time: int = 1700000000,
) -> SimpleNamespace:
    return SimpleNamespace(
        ticket=ticket, symbol=symbol, volume=volume, type=pos_type,
        price_open=price_open, sl=sl, tp=tp, profit=profit,
        magic=20240601, comment="metatrade", time=time,
    )


def _send_result(retcode: int = _MT5_RETCODE_DONE) -> SimpleNamespace:
    return SimpleNamespace(retcode=retcode, comment="ok")


# ---------------------------------------------------------------------------
# get_open_positions
# ---------------------------------------------------------------------------

class TestGetOpenPositions:

    def test_returns_empty_when_not_connected(self) -> None:
        a = MT5BrokerAdapter(login=1, password="x", server="Demo")
        assert a.get_open_positions() == []

    def test_returns_empty_when_no_positions(self) -> None:
        a = _adapter()
        mt5 = MagicMock()
        mt5.positions_get.return_value = []
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5):
            assert a.get_open_positions() == []

    def test_maps_position_fields(self) -> None:
        a = _adapter()
        pos = _mt5_position(ticket=42, symbol="GBPUSD", volume=0.05,
                             price_open=1.27000, sl=1.26800, tp=1.27400)
        mt5 = MagicMock()
        mt5.positions_get.return_value = [pos]
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5):
            result = a.get_open_positions()

        assert len(result) == 1
        p = result[0]
        assert p["ticket"] == 42
        assert p["symbol"] == "GBPUSD"
        assert p["volume"] == 0.05
        assert p["type"] == 0
        assert p["price_open"] == 1.27000
        assert p["sl"] == 1.26800
        assert p["tp"] == 1.27400
        assert isinstance(p["time_utc"], datetime)

    def test_filters_by_symbol(self) -> None:
        a = _adapter()
        mt5 = MagicMock()
        mt5.positions_get.return_value = [_mt5_position()]
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5):
            a.get_open_positions(symbol="EURUSD")
        mt5.positions_get.assert_called_once_with(symbol="EURUSD")

    def test_returns_empty_when_mt5_returns_none(self) -> None:
        a = _adapter()
        mt5 = MagicMock()
        mt5.positions_get.return_value = None
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5):
            assert a.get_open_positions() == []


# ---------------------------------------------------------------------------
# modify_sl_tp
# ---------------------------------------------------------------------------

class TestModifySlTp:

    def _call(
        self,
        *,
        ticket: int = 101,
        new_sl: float | None = 1.09700,
        new_tp: float | None = 1.10300,
        retcode: int = _MT5_RETCODE_DONE,
        pos_sl: float = 1.09800,
        pos_tp: float = 1.10200,
    ) -> tuple[bool, MagicMock]:
        a = _adapter()
        mt5 = MagicMock()
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5, point=0.00001, stops_level=0,
        )
        mt5.symbol_info_tick.return_value = SimpleNamespace(
            ask=1.10010, bid=1.10000,
        )
        mt5.positions_get.return_value = [
            _mt5_position(ticket=ticket, sl=pos_sl, tp=pos_tp)
        ]
        mt5.order_send.return_value = _send_result(retcode)
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5):
            ok = a.modify_sl_tp(ticket, "EURUSD", new_sl, new_tp)
        return ok, mt5

    def test_returns_true_on_success(self) -> None:
        ok, _ = self._call()
        assert ok is True

    def test_returns_false_on_failure(self) -> None:
        ok, _ = self._call(retcode=10006)
        assert ok is False

    def test_sends_sltp_action(self) -> None:
        _, mt5 = self._call(new_sl=1.097, new_tp=1.103)
        sent = mt5.order_send.call_args[0][0]
        assert sent["action"] == _MT5_TRADE_ACTION_SLTP
        assert sent["position"] == 101   # SLTP requests use "position", not "ticket"
        assert sent["sl"] == pytest.approx(1.097, abs=1e-5)
        assert sent["tp"] == pytest.approx(1.103, abs=1e-5)

    def test_keeps_existing_tp_when_new_tp_is_none(self) -> None:
        _, mt5 = self._call(new_sl=1.097, new_tp=None, pos_tp=1.10200)
        sent = mt5.order_send.call_args[0][0]
        assert sent["tp"] == pytest.approx(1.10200, abs=1e-5)

    def test_keeps_existing_sl_when_new_sl_is_none(self) -> None:
        _, mt5 = self._call(new_sl=None, new_tp=1.103, pos_sl=1.09800)
        sent = mt5.order_send.call_args[0][0]
        assert sent["sl"] == pytest.approx(1.09800, abs=1e-5)

    def test_returns_false_when_not_connected(self) -> None:
        a = MT5BrokerAdapter(login=1, password="x", server="Demo")
        assert a.modify_sl_tp(101, "EURUSD", 1.097, 1.103) is False

    def test_returns_false_when_position_not_found(self) -> None:
        a = _adapter()
        mt5 = MagicMock()
        mt5.symbol_info.return_value = SimpleNamespace(digits=5)
        mt5.positions_get.return_value = []   # position not found
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5):
            assert a.modify_sl_tp(999, "EURUSD", 1.097, 1.103) is False


# ---------------------------------------------------------------------------
# close_position_by_ticket
# ---------------------------------------------------------------------------

class TestClosePositionByTicket:

    def _call(
        self,
        *,
        ticket: int = 101,
        volume: float = 0.10,
        is_buy: bool = True,
        retcode: int = _MT5_RETCODE_DONE,
    ) -> tuple[bool, MagicMock]:
        a = _adapter()
        mt5 = MagicMock()
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5, volume_step=0.01, volume_min=0.01, volume_max=100.0,
            filling_mode=1,
        )
        mt5.order_send.return_value = _send_result(retcode)
        mt5.symbol_info_tick.return_value = SimpleNamespace(ask=1.10010, bid=1.10000)
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5):
            ok = a.close_position_by_ticket(ticket, "EURUSD", volume, is_buy)
        return ok, mt5

    def test_returns_true_on_success(self) -> None:
        ok, _ = self._call()
        assert ok is True

    def test_returns_false_on_failure(self) -> None:
        ok, _ = self._call(retcode=10006)
        assert ok is False

    def test_buy_position_closed_with_sell_order(self) -> None:
        _, mt5 = self._call(is_buy=True)
        sent = mt5.order_send.call_args[0][0]
        assert sent["type"] == _MT5_ORDER_TYPE_SELL
        assert sent["action"] == _MT5_TRADE_ACTION_DEAL

    def test_sell_position_closed_with_buy_order(self) -> None:
        _, mt5 = self._call(is_buy=False)
        sent = mt5.order_send.call_args[0][0]
        assert sent["type"] == _MT5_ORDER_TYPE_BUY

    def test_position_ticket_included_in_request(self) -> None:
        _, mt5 = self._call(ticket=555)
        sent = mt5.order_send.call_args[0][0]
        assert sent["position"] == 555

    def test_volume_normalized_to_step(self) -> None:
        """Volume is always rounded to the broker's volume_step before sending."""
        _, mt5 = self._call(volume=0.123)
        sent = mt5.order_send.call_args[0][0]
        # step=0.01 → ROUND_DOWN → 0.12
        assert sent["volume"] == pytest.approx(0.12, abs=1e-9)

    def test_returns_false_when_not_connected(self) -> None:
        a = MT5BrokerAdapter(login=1, password="x", server="Demo")
        assert a.close_position_by_ticket(101, "EURUSD", 0.1, True) is False


# ---------------------------------------------------------------------------
# get_latest_tick
# ---------------------------------------------------------------------------

class TestGetLatestTick:

    def test_returns_none_when_not_connected(self) -> None:
        a = MT5BrokerAdapter(login=1, password="x", server="Demo")
        assert a.get_latest_tick("EURUSD") is None

    def test_returns_tuple_with_time_bid_ask(self) -> None:
        a = _adapter()
        mt5 = MagicMock()
        mt5.symbol_info_tick.return_value = SimpleNamespace(
            time=1700000000, time_msc=1700000000123, bid=1.09990, ask=1.10010
        )
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5):
            result = a.get_latest_tick("EURUSD")

        assert result is not None
        time_msc, bid, ask = result
        assert time_msc == 1700000000123
        assert bid == pytest.approx(1.09990)
        assert ask == pytest.approx(1.10010)

    def test_returns_none_when_tick_unavailable(self) -> None:
        a = _adapter()
        mt5 = MagicMock()
        mt5.symbol_info_tick.return_value = None
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5):
            assert a.get_latest_tick("EURUSD") is None

    def test_falls_back_to_time_times_1000_when_no_time_msc(self) -> None:
        a = _adapter()
        mt5 = MagicMock()
        tick = SimpleNamespace(time=1700000000, bid=1.09990, ask=1.10010)
        # no time_msc attribute
        mt5.symbol_info_tick.return_value = tick
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5):
            result = a.get_latest_tick("EURUSD")
        assert result is not None
        assert result[0] == 1700000000 * 1000


# ---------------------------------------------------------------------------
# order_check pre-validation
# ---------------------------------------------------------------------------

class TestOrderCheckPreValidation:
    """order_check() is called before order_send() and can block/fix bad volumes."""

    def _send_order(
        self,
        *,
        check_retcode: int = _MT5_RETCODE_CHECK_OK,   # order_check() success = 0
        send_retcode: int = _MT5_RETCODE_DONE,         # order_send()  success = 10009
        volume_step: float = 0.01,
    ):
        from metatrade.core.errors import OrderRejectedError

        a = _adapter()
        order = Order(
            order_id=Order.new_id(),
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            lot_size=Decimal("0.10"),
            timestamp_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        mt5 = MagicMock()
        mt5.account_info.return_value = SimpleNamespace(
            margin_free=5000.0, login=1, server="Demo",
            balance=10000.0, equity=10000.0, margin=0.0, currency="USD",
        )
        mt5.order_calc_margin.return_value = 100.0
        mt5.symbol_info.return_value = SimpleNamespace(
            digits=5, volume_step=volume_step, volume_min=0.01,
            volume_max=100.0, filling_mode=1, stops_level=0, point=0.00001,
        )
        mt5.symbol_info_tick.return_value = SimpleNamespace(
            ask=1.10010, bid=1.10000
        )
        mt5.order_check.return_value = SimpleNamespace(
            retcode=check_retcode, comment="ok",
            margin=100.0,
        )
        mt5.order_send.return_value = SimpleNamespace(
            retcode=send_retcode, comment="ok", order=999, price=1.10010,
        )
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5):
            try:
                result = a.send_order(order)
                return result, mt5, None
            except OrderRejectedError as exc:
                return None, mt5, exc

    def test_order_check_called_before_order_send(self) -> None:
        result, mt5, exc = self._send_order()
        assert exc is None
        assert mt5.order_check.called
        # order_check must be called before order_send
        check_idx = [c[0] for c in mt5.method_calls].index("order_check")
        send_idx  = [c[0] for c in mt5.method_calls].index("order_send")
        assert check_idx < send_idx

    def test_raises_when_order_check_fails(self) -> None:
        from metatrade.core.errors import OrderRejectedError
        _, _, exc = self._send_order(check_retcode=10014, send_retcode=10014)
        assert exc is not None
        assert isinstance(exc, OrderRejectedError)
        assert "pre-check" in str(exc).lower() or "10014" in str(exc)

    def test_passes_when_order_check_succeeds(self) -> None:
        result, mt5, exc = self._send_order(
            check_retcode=_MT5_RETCODE_CHECK_OK,   # 0 = order_check OK
            send_retcode=_MT5_RETCODE_DONE,        # 10009 = order_send OK
        )
        assert exc is None
        assert result is not None
