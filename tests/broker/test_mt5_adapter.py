"""Tests for MT5BrokerAdapter (offline — verifies behaviour without MT5)."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from metatrade.broker.mt5_adapter import MT5BrokerAdapter
from metatrade.core.contracts.order import Order
from metatrade.core.enums import OrderSide, OrderType
from metatrade.core.errors import BrokerConnectionError


def _make_order(
    side: OrderSide = OrderSide.BUY,
    lots: str = "0.50",
    price: str | None = None,
) -> Order:
    return Order(
        order_id=Order.new_id(),
        symbol="EURUSD",
        side=side,
        order_type=OrderType.MARKET,
        lot_size=Decimal(lots),
        price=Decimal(price) if price else None,
        timestamp_utc=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
    )


def _make_adapter() -> MT5BrokerAdapter:
    adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
    adapter._connected = True  # bypass connection guard for unit tests
    return adapter


class TestMT5AdapterOffline:
    """Tests that run without MetaTrader5 installed.

    The adapter uses lazy import — operations that don't require the library
    can be tested normally on any OS.
    """

    def test_not_connected_initially(self) -> None:
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        assert not adapter.is_connected()

    def test_connect_raises_on_missing_mt5(self) -> None:
        """On macOS/Linux (no MT5), connect() raises BrokerConnectionError."""
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        try:
            import MetaTrader5  # noqa: F401
            pytest.skip("MetaTrader5 is installed — skipping offline test")
        except ImportError:
            pass
        with pytest.raises(BrokerConnectionError, match="not available"):
            adapter.connect()

    def test_send_order_raises_when_not_connected(self) -> None:
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        order = _make_order()
        with pytest.raises(BrokerConnectionError):
            adapter.send_order(order)

    def test_get_account_raises_when_not_connected(self) -> None:
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        with pytest.raises(BrokerConnectionError):
            adapter.get_account()

    def test_get_current_price_raises_when_not_connected(self) -> None:
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        with pytest.raises(BrokerConnectionError):
            adapter.get_current_price("EURUSD")

    def test_cancel_order_returns_false_when_not_connected(self) -> None:
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        # cancel_order doesn't check connection — just logs and returns False
        result = adapter.cancel_order("some-order-id")
        assert result is False


# ── _get_filling_mode ─────────────────────────────────────────────────────────

class TestGetFillingMode:
    """_get_filling_mode selects the best type_filling supported by the broker."""

    def _call(self, filling_mode_flags: int | None) -> int:
        adapter = _make_adapter()
        mt5_mock = MagicMock()
        if filling_mode_flags is None:
            mt5_mock.symbol_info.return_value = None
        else:
            mt5_mock.symbol_info.return_value = SimpleNamespace(filling_mode=filling_mode_flags)
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5_mock):
            return adapter._get_filling_mode("EURUSD")

    def test_fok_preferred_when_supported(self) -> None:
        # bit0 set → FOK (value 0)
        assert self._call(filling_mode_flags=1) == 0

    def test_fok_preferred_over_ioc_when_both_set(self) -> None:
        # bits 0 and 1 both set → still prefers FOK
        assert self._call(filling_mode_flags=3) == 0

    def test_ioc_used_when_only_ioc_supported(self) -> None:
        # bit1 only → IOC (value 1)
        assert self._call(filling_mode_flags=2) == 1

    def test_return_fallback_when_no_flags(self) -> None:
        # filling_mode = 0 → RETURN (value 2)
        assert self._call(filling_mode_flags=0) == 2

    def test_return_fallback_when_symbol_info_none(self) -> None:
        # symbol_info() returns None → RETURN (value 2)
        assert self._call(filling_mode_flags=None) == 2


# ── _clamp_lot_to_margin ──────────────────────────────────────────────────────

class TestClampLotToMargin:
    """_clamp_lot_to_margin scales lots down when broker margin would be exceeded."""

    def _call(
        self,
        order: Order,
        *,
        free_margin: float,
        required_margin: float,
        tick_ask: float = 1.10000,
        tick_bid: float = 1.09990,
    ) -> Order:
        adapter = _make_adapter()
        mt5_mock = MagicMock()
        mt5_mock.account_info.return_value = SimpleNamespace(margin_free=free_margin)
        mt5_mock.order_calc_margin.return_value = required_margin
        mt5_mock.symbol_info_tick.return_value = SimpleNamespace(
            ask=tick_ask, bid=tick_bid
        )
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5_mock):
            return adapter._clamp_lot_to_margin(order)

    def test_order_unchanged_when_margin_fits(self) -> None:
        order = _make_order(lots="0.50")
        # required 100, usable = 1000 * 0.9 = 900 → fits
        result = self._call(order, free_margin=1000.0, required_margin=100.0)
        assert result.lot_size == Decimal("0.50")

    def test_lot_scaled_down_when_margin_exceeded(self) -> None:
        order = _make_order(lots="1.00")
        # free=200, usable=180; required=500 → scale = 180/500 = 0.36 → 0.36 lots → rounds to 0.36
        result = self._call(order, free_margin=200.0, required_margin=500.0)
        assert result.lot_size < Decimal("1.00")
        assert result.lot_size >= Decimal("0.01")

    def test_lot_floored_at_min_lot(self) -> None:
        order = _make_order(lots="0.50")
        # extreme case: free=1, usable=0.9; required=1000 → scale tiny → must be at least 0.01
        result = self._call(order, free_margin=1.0, required_margin=1000.0)
        assert result.lot_size == Decimal("0.01")

    def test_order_unchanged_when_account_info_none(self) -> None:
        adapter = _make_adapter()
        mt5_mock = MagicMock()
        mt5_mock.account_info.return_value = None
        order = _make_order(lots="2.00")
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5_mock):
            result = adapter._clamp_lot_to_margin(order)
        assert result.lot_size == Decimal("2.00")

    def test_order_unchanged_when_calc_margin_returns_none(self) -> None:
        order = _make_order(lots="1.00")
        adapter = _make_adapter()
        mt5_mock = MagicMock()
        mt5_mock.account_info.return_value = SimpleNamespace(margin_free=500.0)
        mt5_mock.order_calc_margin.return_value = None
        mt5_mock.symbol_info_tick.return_value = SimpleNamespace(ask=1.1, bid=1.0)
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5_mock):
            result = adapter._clamp_lot_to_margin(order)
        assert result.lot_size == Decimal("1.00")

    def test_order_unchanged_when_calc_margin_zero(self) -> None:
        order = _make_order(lots="1.00")
        adapter = _make_adapter()
        mt5_mock = MagicMock()
        mt5_mock.account_info.return_value = SimpleNamespace(margin_free=500.0)
        mt5_mock.order_calc_margin.return_value = 0.0
        mt5_mock.symbol_info_tick.return_value = SimpleNamespace(ask=1.1, bid=1.0)
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5_mock):
            result = adapter._clamp_lot_to_margin(order)
        assert result.lot_size == Decimal("1.00")

    def test_sell_order_uses_bid_price(self) -> None:
        """For SELL orders, _clamp_lot_to_margin uses tick.bid as price."""
        order = _make_order(side=OrderSide.SELL, lots="0.10")
        adapter = _make_adapter()
        mt5_mock = MagicMock()
        mt5_mock.account_info.return_value = SimpleNamespace(margin_free=10_000.0)
        mt5_mock.order_calc_margin.return_value = 50.0
        tick = SimpleNamespace(ask=1.10100, bid=1.10000)
        mt5_mock.symbol_info_tick.return_value = tick
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5_mock):
            adapter._clamp_lot_to_margin(order)
        # verify order_calc_margin was called with bid price (1.10000) for SELL
        _args = mt5_mock.order_calc_margin.call_args[0]
        assert _args[3] == pytest.approx(1.10000)

    def test_lot_step_respected_after_scaling(self) -> None:
        """Scaled lot must be a multiple of 0.01."""
        order = _make_order(lots="0.77")
        result = self._call(order, free_margin=300.0, required_margin=700.0)
        # result.lot_size should be a multiple of 0.01
        remainder = result.lot_size % Decimal("0.01")
        assert remainder == Decimal("0")


# ── _adjust_stops_for_min_distance ───────────────────────────────────────────

def _make_symbol_info(*, point: float = 0.00001, stops_level: int = 0, digits: int = 5) -> SimpleNamespace:
    return SimpleNamespace(point=point, stops_level=stops_level, digits=digits)


class TestAdjustStopsForMinDistance:
    """_adjust_stops_for_min_distance widens SL/TP that are too close to price."""

    def _call(
        self,
        order: Order,
        request: dict,
        *,
        ask: float = 1.10010,
        bid: float = 1.10000,
        point: float = 0.00001,
        stops_level: int = 0,
        min_stop_pips: float = 10.0,
    ) -> dict:
        adapter = MT5BrokerAdapter(
            login=12345, password="pw", server="Demo",
            min_stop_pips=min_stop_pips,
        )
        adapter._connected = True
        mt5_mock = MagicMock()
        mt5_mock.symbol_info_tick.return_value = SimpleNamespace(ask=ask, bid=bid)
        mt5_mock.symbol_info.return_value = _make_symbol_info(point=point, stops_level=stops_level)
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5_mock):
            return adapter._adjust_stops_for_min_distance(order, request, digits=5)

    def test_noop_when_no_sl_tp(self) -> None:
        order = _make_order()
        request: dict = {"action": 1, "symbol": "EURUSD", "volume": 0.5}
        result = self._call(order, request)
        assert result == request

    def test_noop_when_tick_unavailable(self) -> None:
        order = _make_order(side=OrderSide.SELL)
        request = {"sl": 1.10100}
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo")
        adapter._connected = True
        mt5_mock = MagicMock()
        mt5_mock.symbol_info_tick.return_value = None
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5_mock):
            result = adapter._adjust_stops_for_min_distance(order, request, digits=5)
        assert result == request

    # ── BUY ──────────────────────────────────────────────────────────────────

    def test_buy_sl_unchanged_when_far_enough(self) -> None:
        """BUY SL already 15 pips below ask — no adjustment needed (min is 10 pips)."""
        order = _make_order(side=OrderSide.BUY)
        # ask=1.10010, min_stop=10 pips=0.00100, max_valid_sl=1.10010-0.00100=1.09910
        request = {"sl": 1.09895}  # 11.5 pips below ask — fine
        result = self._call(order, request, ask=1.10010, min_stop_pips=10.0)
        assert result["sl"] == pytest.approx(1.09895)

    def test_buy_sl_pushed_down_when_too_close(self) -> None:
        """BUY SL 3 pips below ask is within 10-pip minimum — pushed to 10 pips."""
        order = _make_order(side=OrderSide.BUY)
        # ask=1.10000; sl=1.09970 is only 3 pips below ask
        # min_stop=10 pips → max_valid_sl = 1.10000 - 0.00100 = 1.09900
        request = {"sl": 1.09970}
        result = self._call(order, request, ask=1.10000, min_stop_pips=10.0)
        assert result["sl"] == pytest.approx(1.09900)

    def test_buy_tp_unchanged_when_far_enough(self) -> None:
        order = _make_order(side=OrderSide.BUY)
        request = {"tp": 1.10200}  # 19 pips above ask=1.10010
        result = self._call(order, request, ask=1.10010, min_stop_pips=10.0)
        assert result["tp"] == pytest.approx(1.10200)

    def test_buy_tp_pushed_up_when_too_close(self) -> None:
        """BUY TP 5 pips above ask is within 10-pip minimum — pushed to 10 pips."""
        order = _make_order(side=OrderSide.BUY)
        # ask=1.10000; tp=1.10050 is only 5 pips above
        # min_valid_tp = 1.10000 + 0.00100 = 1.10100
        request = {"tp": 1.10050}
        result = self._call(order, request, ask=1.10000, min_stop_pips=10.0)
        assert result["tp"] == pytest.approx(1.10100)

    # ── SELL ─────────────────────────────────────────────────────────────────

    def test_sell_sl_unchanged_when_far_enough(self) -> None:
        """SELL SL already 15 pips above bid — no adjustment needed."""
        order = _make_order(side=OrderSide.SELL)
        # bid=1.10000, min_stop=10 pips → min_valid_sl=1.10100
        request = {"sl": 1.10150}  # 15 pips above bid — fine
        result = self._call(order, request, bid=1.10000, min_stop_pips=10.0)
        assert result["sl"] == pytest.approx(1.10150)

    def test_sell_sl_pushed_up_when_too_close(self) -> None:
        """SELL SL 3 pips above bid is within 10-pip minimum — pushed up."""
        order = _make_order(side=OrderSide.SELL)
        # bid=1.10000; sl=1.10030 is only 3 pips above
        # min_valid_sl = 1.10000 + 0.00100 = 1.10100
        request = {"sl": 1.10030}
        result = self._call(order, request, bid=1.10000, min_stop_pips=10.0)
        assert result["sl"] == pytest.approx(1.10100)

    def test_sell_tp_unchanged_when_far_enough(self) -> None:
        order = _make_order(side=OrderSide.SELL)
        request = {"tp": 1.09800}  # 20 pips below bid=1.10000
        result = self._call(order, request, bid=1.10000, min_stop_pips=10.0)
        assert result["tp"] == pytest.approx(1.09800)

    def test_sell_tp_pushed_down_when_too_close(self) -> None:
        """SELL TP 5 pips below bid is within 10-pip minimum — pushed down."""
        order = _make_order(side=OrderSide.SELL)
        # bid=1.10000; tp=1.09950 is only 5 pips below
        # max_valid_tp = 1.10000 - 0.00100 = 1.09900
        request = {"tp": 1.09950}
        result = self._call(order, request, bid=1.10000, min_stop_pips=10.0)
        assert result["tp"] == pytest.approx(1.09900)

    # ── broker stops_level ────────────────────────────────────────────────────

    def test_broker_stops_level_used_when_larger_than_min_pips(self) -> None:
        """When broker stops_level (50 points = 5 pips) > min_stop_pips (3), broker wins."""
        order = _make_order(side=OrderSide.BUY)
        # ask=1.10000, stops_level=50 points, point=0.00001 → min_distance=0.00050 (5 pips)
        # min_stop_pips=3 pips → 30 points — broker's 50 points wins
        # max_valid_sl = 1.10000 - 0.00050 = 1.09950
        request = {"sl": 1.09980}  # only 2 pips below ask — too close
        result = self._call(
            order, request, ask=1.10000, point=0.00001,
            stops_level=50, min_stop_pips=3.0,
        )
        assert result["sl"] == pytest.approx(1.09950)

    # ── JPY pairs ─────────────────────────────────────────────────────────────

    def test_jpy_pair_pip_calculation(self) -> None:
        """For JPY pairs (point=0.001), 1 pip = 0.01 price units."""
        order = Order(
            order_id=Order.new_id(),
            symbol="USDJPY",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            lot_size=Decimal("0.10"),
            timestamp_utc=datetime(2024, 1, 15, 10, 0, tzinfo=timezone.utc),
        )
        # bid=150.000, min_stop_pips=10 → min_distance = 10 * 0.01 = 0.10
        # min_valid_sl = 150.000 + 0.10 = 150.100
        request = {"sl": 150.050}  # only 5 pips above bid — too close
        adapter = MT5BrokerAdapter(login=12345, password="pw", server="Demo", min_stop_pips=10.0)
        adapter._connected = True
        mt5_mock = MagicMock()
        mt5_mock.symbol_info_tick.return_value = SimpleNamespace(ask=150.010, bid=150.000)
        mt5_mock.symbol_info.return_value = SimpleNamespace(point=0.001, stops_level=0, digits=3)
        with patch("metatrade.broker.mt5_adapter._get_mt5", return_value=mt5_mock):
            result = adapter._adjust_stops_for_min_distance(order, request, digits=3)
        assert result["sl"] == pytest.approx(150.100)
