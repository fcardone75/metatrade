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

from datetime import datetime, timezone
from decimal import Decimal

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
_MT5_FILLING_FOK = 1               # Fill or Kill (most common for FX)
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
    ) -> None:
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._timeout = timeout
        self._run_mode = run_mode
        self._deviation = deviation
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

        self._connected = True
        log.info("mt5_connected", login=self._login, server=self._server)

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

    def _build_mt5_request(self, order: Order) -> dict:
        """Build an MQL5 trade request dict from an Order."""
        is_buy = order.side == OrderSide.BUY
        action = _MT5_TRADE_ACTION_DEAL
        order_type = _MT5_ORDER_TYPE_BUY if is_buy else _MT5_ORDER_TYPE_SELL

        request: dict = {
            "action": action,
            "symbol": order.symbol,
            "volume": float(order.lot_size),
            "type": order_type,
            "deviation": self._deviation,
            "magic": 12345,  # magic number to identify our EA
            "comment": order.comment or "metatrade",
            "type_filling": _MT5_FILLING_FOK,
        }
        if order.price is not None:
            request["price"] = float(order.price)
        if order.stop_loss is not None:
            request["sl"] = float(order.stop_loss)
        if order.take_profit is not None:
            request["tp"] = float(order.take_profit)

        return request
