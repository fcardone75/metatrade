"""MetaTrader 5 broker adapter.

Wraps the MetaTrader5 Python library for live and paper trading.
MT5 is Windows-only — the library import is deferred until connect() is called,
so the module can be imported cleanly on macOS/Linux for CI and testing.

Order flow:
    send_order()
        → mt5.order_send(request)
        → parse retcode
        → return Order with SUBMITTED status and broker_order_id

Account state:
    get_account() → mt5.account_info()

Price feeds:
    get_current_price() → mt5.symbol_info_tick()

Trade requests use the MQL5 trade request structure. The constants below
map our internal enums to MT5 constants.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal

from metatrade.core.contracts.account import AccountState
from metatrade.core.contracts.order import Order
from metatrade.core.enums import OrderSide, OrderStatus, OrderType, RunMode
from metatrade.core.errors import (
    BrokerConnectionError,
    BrokerTimeoutError,
    OrderRejectedError,
)
from metatrade.core.log import get_logger
from metatrade.broker.interface import IBrokerAdapter

log = get_logger(__name__)

# MT5 trade action constants
_MT5_TRADE_ACTION_DEAL = 1         # Immediate execution (market order)
_MT5_TRADE_ACTION_PENDING = 5      # Place a pending order
_MT5_ORDER_TYPE_BUY = 0
_MT5_ORDER_TYPE_SELL = 1
_MT5_ORDER_TYPE_BUY_LIMIT = 2
_MT5_ORDER_TYPE_SELL_LIMIT = 3
_MT5_ORDER_TYPE_BUY_STOP = 4
_MT5_ORDER_TYPE_SELL_STOP = 5
# type_filling values for mt5.order_send()
_MT5_FILLING_FOK = 0               # Fill or Kill  (ORDER_FILLING_FOK)
_MT5_FILLING_IOC = 1               # Immediate or Cancel (ORDER_FILLING_IOC)
_MT5_FILLING_RETURN = 2            # Return remainder (ORDER_FILLING_RETURN)
# symbol_info().filling_mode bitmask
_MT5_FILL_FLAG_FOK = 1             # broker supports FOK
_MT5_FILL_FLAG_IOC = 2             # broker supports IOC
_MT5_RETCODE_DONE = 10009          # Success
_MT5_RETCODE_DONE_PARTIAL = 10010  # Partial fill


def _get_mt5():  # type: ignore[return]
    """Lazy import of the MetaTrader5 library.

    Raises:
        BrokerConnectionError: On non-Windows systems where MT5 is unavailable.
    """
    try:
        import MetaTrader5 as mt5  # type: ignore[import]
        return mt5
    except ImportError as exc:
        raise BrokerConnectionError(
            message=(
                "MetaTrader5 library is not available. "
                "MT5 trading requires Windows and the MetaTrader5 Python package. "
                "Install it with: pip install MetaTrader5"
            ),
            code="MT5_NOT_AVAILABLE",
        ) from exc


class MT5BrokerAdapter(IBrokerAdapter):
    """Live/paper broker adapter for MetaTrader 5.

    Args:
        login:    MT5 account number.
        password: MT5 account password.
        server:   MT5 broker server name (e.g. "ICMarkets-Demo").
        path:     Path to MetaTrader5 terminal executable (optional).
        timeout:  Connection timeout in milliseconds (default 30000).
        run_mode: RunMode.LIVE or RunMode.PAPER.
                  PAPER mode uses the demo/paper account; order execution is real
                  but positions are not real money.
        deviation: Maximum price deviation in points (default 20).
    """

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        path: str | None = None,
        timeout: int = 30_000,
        run_mode: RunMode = RunMode.PAPER,
        deviation: int = 20,
        magic_number: int = 12345,
    ) -> None:
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._timeout = timeout
        self._run_mode = run_mode
        self._deviation = deviation
        self._magic_number = magic_number
        self._connected = False

    # ── IBrokerAdapter ────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Initialize MT5 and log in.

        Raises:
            BrokerConnectionError: If MT5 init or login fails.
        """
        mt5 = _get_mt5()

        init_kwargs: dict = {"timeout": self._timeout}
        if self._path:
            init_kwargs["path"] = self._path

        if not mt5.initialize(**init_kwargs):
            err = mt5.last_error()
            raise BrokerConnectionError(
                message=f"MT5 initialize() failed: {err}",
                code="MT5_INIT_FAILED",
                context={"error": str(err)},
            )

        if self._login and self._password and self._server:
            authorized = mt5.login(
                login=self._login,
                password=self._password,
                server=self._server,
            )
            if not authorized:
                mt5.shutdown()
                err = mt5.last_error()
                raise BrokerConnectionError(
                    message=f"MT5 login() failed for account {self._login}: {err}",
                    code="MT5_LOGIN_FAILED",
                    context={"login": self._login, "server": self._server, "error": str(err)},
                )

        info = mt5.account_info()
        if info is None:
            mt5.shutdown()
            raise BrokerConnectionError(
                message="MT5 connected but account_info() returned None - not logged in",
                code="MT5_NOT_LOGGED_IN",
            )

        self._connected = True
        log.info("mt5_connected", login=info.login, server=info.server)

    def disconnect(self) -> None:
        if self._connected:
            _get_mt5().shutdown()
            self._connected = False
            log.info("mt5_disconnected")

    def is_connected(self) -> bool:
        return self._connected

    def send_order(self, order: Order) -> Order:
        self._require_connected()
        mt5 = _get_mt5()

        order = self._clamp_lot_to_margin(order)
        request = self._build_mt5_request(order)
        log.debug("mt5_order_send", request=request)

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            raise OrderRejectedError(
                message=f"MT5 order_send returned None: {err}",
                code="MT5_ORDER_SEND_NULL",
                context={"error": str(err), "symbol": order.symbol},
            )

        if result.retcode not in (_MT5_RETCODE_DONE, _MT5_RETCODE_DONE_PARTIAL):
            raise OrderRejectedError(
                message=(
                    f"MT5 order rejected: retcode={result.retcode} "
                    f"comment={result.comment}"
                ),
                code="MT5_ORDER_REJECTED",
                context={
                    "retcode": result.retcode,
                    "comment": result.comment,
                    "order": str(order.order_id),
                },
            )

        broker_id = str(result.order)
        log.info(
            "mt5_order_filled",
            order_id=order.order_id,
            broker_id=broker_id,
            symbol=order.symbol,
            price=str(result.price),
        )
        return order.with_status(OrderStatus.SUBMITTED, broker_order_id=broker_id)

    def cancel_order(self, order_id: str) -> bool:
        # MT5 uses position ticket numbers; this is a no-op placeholder
        log.warning("mt5_cancel_order_not_implemented", order_id=order_id)
        return False

    def get_account(self) -> AccountState:
        self._require_connected()
        mt5 = _get_mt5()
        info = mt5.account_info()
        if info is None:
            raise BrokerConnectionError(
                message="MT5 account_info() returned None",
                code="MT5_ACCOUNT_INFO_FAILED",
            )
        return AccountState(
            balance=Decimal(str(info.balance)),
            equity=Decimal(str(info.equity)),
            free_margin=Decimal(str(info.margin_free)),
            used_margin=Decimal(str(info.margin)),
            timestamp_utc=datetime.now(timezone.utc),
            currency=info.currency,
            run_mode=self._run_mode,
        )

    def get_current_price(self, symbol: str) -> tuple[Decimal, Decimal]:
        self._require_connected()
        mt5 = _get_mt5()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise KeyError(f"MT5: no tick data for {symbol}")
        return Decimal(str(tick.bid)), Decimal(str(tick.ask))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if not self._connected:
            raise BrokerConnectionError(
                message="MT5BrokerAdapter: not connected — call connect() first",
                code="MT5_NOT_CONNECTED",
            )

    def _clamp_lot_to_margin(self, order: Order) -> Order:
        """Scale down lot size if the broker would reject it for insufficient margin.

        Uses mt5.order_calc_margin() to get the exact margin requirement and
        mt5.account_info().margin_free to get available margin.  When available
        margin is less than required, the lot size is reduced proportionally and
        a warning is logged.  The order is returned unchanged if mt5 cannot
        compute the margin (e.g. symbol not available).
        """
        mt5 = _get_mt5()
        account = mt5.account_info()
        if account is None:
            return order

        action = _MT5_ORDER_TYPE_BUY if order.side == OrderSide.BUY else _MT5_ORDER_TYPE_SELL
        price = float(order.price) if order.price is not None else 0.0
        if price == 0.0:
            tick = mt5.symbol_info_tick(order.symbol)
            if tick is not None:
                price = tick.ask if order.side == OrderSide.BUY else tick.bid

        required = mt5.order_calc_margin(action, order.symbol, float(order.lot_size), price)
        if required is None or required <= 0.0:
            return order

        free = account.margin_free
        # Keep a 10 % safety buffer on free margin
        usable = free * 0.90
        if required <= usable:
            return order

        # Scale lot size down proportionally
        scale = usable / required
        lot_step = Decimal("0.01")
        new_lots = (
            (order.lot_size * Decimal(str(scale))) / lot_step
        ).to_integral_value(rounding=ROUND_DOWN) * lot_step
        min_lot = Decimal("0.01")
        new_lots = max(new_lots, min_lot)

        log.warning(
            "mt5_lot_clamped_for_margin",
            symbol=order.symbol,
            original_lots=str(order.lot_size),
            clamped_lots=str(new_lots),
            required_margin=round(required, 2),
            free_margin=round(free, 2),
        )
        return dc_replace(order, lot_size=new_lots)

    def _get_filling_mode(self, symbol: str) -> int:
        """Return the best supported type_filling value for *symbol*.

        MT5 brokers advertise supported filling modes via
        ``symbol_info().filling_mode`` (bitmask: bit0=FOK, bit1=IOC).
        Preference order: FOK → IOC → RETURN (always accepted as fallback).
        """
        mt5 = _get_mt5()
        info = mt5.symbol_info(symbol)
        if info is not None:
            filling_flags: int = info.filling_mode
            if filling_flags & _MT5_FILL_FLAG_FOK:
                return _MT5_FILLING_FOK
            if filling_flags & _MT5_FILL_FLAG_IOC:
                return _MT5_FILLING_IOC
        # RETURN is always supported by MT5 for market orders
        return _MT5_FILLING_RETURN

    def _symbol_digits(self, symbol: str) -> int:
        """Return the number of decimal places for *symbol* prices (default 5)."""
        mt5 = _get_mt5()
        info = mt5.symbol_info(symbol)
        if info is not None:
            return int(info.digits)
        return 5

    def _build_mt5_request(self, order: Order) -> dict:
        """Build an MQL5 trade request dict from an Order."""
        is_buy = order.side == OrderSide.BUY
        action = _MT5_TRADE_ACTION_DEAL
        order_type = _MT5_ORDER_TYPE_BUY if is_buy else _MT5_ORDER_TYPE_SELL

        # MT5 requires prices normalised to the symbol's tick size (digits).
        # Passing raw Python float arithmetic results (e.g. 1.1798970657790784)
        # causes retcode=10016 (TRADE_RETCODE_INVALID_STOPS).
        digits = self._symbol_digits(order.symbol)

        def _norm(price: Decimal) -> float:
            return round(float(price), digits)

        request: dict = {
            "action": action,
            "symbol": order.symbol,
            "volume": float(order.lot_size),
            "type": order_type,
            "deviation": self._deviation,
            "magic": self._magic_number,
            "comment": order.comment or "metatrade",
            "type_filling": self._get_filling_mode(order.symbol),
        }
        if order.price is not None:
            request["price"] = _norm(order.price)
        if order.stop_loss is not None:
            request["sl"] = _norm(order.stop_loss)
        if order.take_profit is not None:
            request["tp"] = _norm(order.take_profit)

        return request
