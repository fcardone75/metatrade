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
from datetime import UTC, datetime
from decimal import ROUND_DOWN, Decimal

from metatrade.broker.interface import IBrokerAdapter
from metatrade.core.contracts.account import AccountState
from metatrade.core.contracts.order import Order
from metatrade.core.enums import OrderSide, OrderStatus, RunMode
from metatrade.core.errors import (
    BrokerConnectionError,
    OrderRejectedError,
)
from metatrade.core.log import get_logger

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
_MT5_RETCODE_DONE = 10009          # order_send() success
_MT5_RETCODE_DONE_PARTIAL = 10010  # order_send() partial fill
_MT5_RETCODE_CHECK_OK = 0          # order_check() success (different scale from order_send)
_MT5_TRADE_ACTION_SLTP = 6         # Modify stop-loss / take-profit of open position
_MT5_RETCODE_INVALID_VOLUME = 10014


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
        min_stop_pips: float = 10.0,
    ) -> None:
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._timeout = timeout
        self._run_mode = run_mode
        self._deviation = deviation
        self._magic_number = magic_number
        self._min_stop_pips = min_stop_pips
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

        # Pre-flight validation — catches volume / stop / margin issues before
        # they become retcode errors that show up in the dashboard as failures.
        # order_check() uses retcode=0 for success, NOT 10009 (which is the
        # order_send() success code).  Using the wrong constant here caused every
        # order to be incorrectly rejected even when the pre-check passed.
        check = mt5.order_check(request)
        if check is not None and check.retcode != _MT5_RETCODE_CHECK_OK:
            if check.retcode == _MT5_RETCODE_INVALID_VOLUME:
                # Re-normalize volume against the live symbol constraints and retry once
                new_volume = float(self._normalize_volume(
                    Decimal(str(request["volume"])), order.symbol
                ))
                if new_volume != request["volume"]:
                    log.warning(
                        "mt5_order_check_volume_renormalized",
                        symbol=order.symbol,
                        original_volume=request["volume"],
                        renormalized_volume=new_volume,
                    )
                    request = dict(request)
                    request["volume"] = new_volume
                    check = mt5.order_check(request)
            if check is not None and check.retcode != _MT5_RETCODE_CHECK_OK:
                log.error(
                    "mt5_order_check_failed",
                    retcode=check.retcode,
                    comment=check.comment,
                    symbol=order.symbol,
                    volume=request.get("volume"),
                    margin=check.margin if hasattr(check, "margin") else None,
                )
                raise OrderRejectedError(
                    message=(
                        f"MT5 order pre-check failed: retcode={check.retcode} "
                        f"comment={check.comment}"
                    ),
                    code="MT5_ORDER_CHECK_FAILED",
                    context={"retcode": check.retcode, "comment": check.comment},
                )

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            raise OrderRejectedError(
                message=f"MT5 order_send returned None: {err}",
                code="MT5_ORDER_SEND_NULL",
                context={"error": str(err), "symbol": order.symbol},
            )

        if result.retcode not in (_MT5_RETCODE_DONE, _MT5_RETCODE_DONE_PARTIAL):
            log.error(
                "mt5_order_rejected",
                retcode=result.retcode,
                comment=result.comment,
                order_id=str(order.order_id),
                symbol=order.symbol,
                side=order.side.value,
                volume=request.get("volume"),
                sl=request.get("sl"),
                tp=request.get("tp"),
            )
            raise OrderRejectedError(
                message=(
                    f"MT5 order rejected: retcode={result.retcode} "
                    f"comment={result.comment}"
                ),
                code="MT5_ORDER_REJECTED",
                context={
                    "retcode": result.retcode,
                    "comment": result.comment,
                    "order_id": str(order.order_id),
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

        # MT5 can report negative margin_free when the account is over-margined
        # (margin call territory).  AccountState rejects negatives, so we clamp
        # to zero and warn — the PreTradeChecker will veto new entries via the
        # INSUFFICIENT_MARGIN check without crashing the service.
        raw_free_margin = float(info.margin_free)
        raw_used_margin = float(info.margin)
        if raw_free_margin < 0:
            log.warning(
                "mt5_negative_free_margin",
                raw_free_margin=round(raw_free_margin, 2),
                action="clamped_to_zero",
            )
            raw_free_margin = 0.0
        if raw_used_margin < 0:
            log.warning(
                "mt5_negative_used_margin",
                raw_used_margin=round(raw_used_margin, 2),
                action="clamped_to_zero",
            )
            raw_used_margin = 0.0

        # Count open positions from the broker's live position list, not from
        # account_info (which does not expose this count directly in MT5).
        try:
            positions = mt5.positions_get() or []
            open_positions_count = len(positions)
        except Exception:  # noqa: BLE001
            open_positions_count = 0

        return AccountState(
            balance=Decimal(str(info.balance)),
            equity=Decimal(str(info.equity)),
            free_margin=Decimal(str(raw_free_margin)),
            used_margin=Decimal(str(raw_used_margin)),
            timestamp_utc=datetime.now(UTC),
            currency=info.currency,
            open_positions_count=open_positions_count,
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

    # ── Symbol info helpers ───────────────────────────────────────────────────

    def _symbol_info(self, symbol: str):  # type: ignore[return]
        """Return mt5.symbol_info() or None without raising."""
        try:
            return _get_mt5().symbol_info(symbol)
        except Exception:
            return None

    def _symbol_digits(self, symbol: str) -> int:
        """Decimal places for prices (default 5).  EURUSD → 5, USDJPY → 3."""
        info = self._symbol_info(symbol)
        return int(info.digits) if info is not None else 5

    def _symbol_volume_step(self, symbol: str) -> Decimal:
        """Minimum lot increment advertised by the broker (default 0.01)."""
        info = self._symbol_info(symbol)
        if info is not None:
            step = getattr(info, "volume_step", None)
            if step and step > 0:
                return Decimal(str(step))
        return Decimal("0.01")

    def _symbol_volume_min(self, symbol: str) -> Decimal:
        """Minimum tradeable lot size (default 0.01)."""
        info = self._symbol_info(symbol)
        if info is not None:
            vmin = getattr(info, "volume_min", None)
            if vmin and vmin > 0:
                return Decimal(str(vmin))
        return Decimal("0.01")

    def _symbol_volume_max(self, symbol: str) -> Decimal:
        """Maximum tradeable lot size (default 100)."""
        info = self._symbol_info(symbol)
        if info is not None:
            vmax = getattr(info, "volume_max", None)
            if vmax and vmax > 0:
                return Decimal(str(vmax))
        return Decimal("100")

    def _normalize_volume(self, lots: Decimal, symbol: str) -> Decimal:
        """Round lots to the broker's volume_step and clamp to [volume_min, volume_max].

        This is the single source of truth for valid volume values.
        Needed because:
        - Drawdown-recovery and intermarket scaling multiply without rounding.
        - Python Decimal arithmetic can produce values like 2.9850000000000001.
        - MT5 rejects volumes that are not exact multiples of volume_step
          (retcode=10014 TRADE_RETCODE_INVALID_VOLUME).
        """
        step = self._symbol_volume_step(symbol)
        vmin = self._symbol_volume_min(symbol)
        vmax = self._symbol_volume_max(symbol)
        # Floor to step boundary
        normalized = (lots / step).to_integral_value(rounding=ROUND_DOWN) * step
        return max(vmin, min(normalized, vmax))

    # ── Margin clamping ───────────────────────────────────────────────────────

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

        # Scale lot size down proportionally, then round to valid broker step.
        scale = usable / required
        new_lots = self._normalize_volume(
            order.lot_size * Decimal(str(scale)), order.symbol
        )

        log.warning(
            "mt5_lot_clamped_for_margin",
            symbol=order.symbol,
            original_lots=str(order.lot_size),
            clamped_lots=str(new_lots),
            required_margin=round(required, 2),
            free_margin=round(free, 2),
        )
        return dc_replace(order, lot_size=new_lots)

    # ── Minimum stop distance ─────────────────────────────────────────────────

    def _adjust_stops_for_min_distance(
        self,
        order: Order,
        request: dict,
        digits: int,
    ) -> dict:
        """Ensure SL/TP respect the broker's minimum stop distance.

        MT5 brokers advertise ``symbol_info.stops_level`` (minimum distance
        between current price and a pending stop, in *points*).  Violating it
        causes retcode=10016 TRADE_RETCODE_INVALID_STOPS.

        We also apply ``self._min_stop_pips`` as a hard floor on top of
        ``stops_level``.  This prevents *immediate stop-out*: when ATR is
        small (e.g. on M1), the calculated SL may sit within spread distance
        of the fill price, causing MT5 to trigger it on the very next tick.

        The adjustment only *widens* stop distance — it never moves SL/TP in
        the wrong direction (i.e. past entry price).
        """
        if request.get("sl") is None and request.get("tp") is None:
            return request

        mt5 = _get_mt5()
        tick = mt5.symbol_info_tick(order.symbol)
        info = mt5.symbol_info(order.symbol)

        if tick is None or info is None:
            return request

        point: float = getattr(info, "point", 0.00001)
        stops_level: int = getattr(info, "stops_level", 0)

        # Minimum distance in price units:
        #   max(broker stops_level, user-configured min_stop_pips in points)
        # 1 pip = 0.0001 for 4/5-digit pairs (EURUSD), 0.01 for JPY pairs.
        # In points: 1 pip = 10 points for 5-digit, 1 point for 3-digit.
        # Safest approach: compute 1 pip in price units and divide by point.
        pip_size = 0.0001 if point < 0.001 else 0.01  # 4/5-digit vs JPY
        pip_points = max(1, round(pip_size / point))
        min_points = max(stops_level, int(self._min_stop_pips * pip_points))
        min_distance = min_points * point

        is_buy = order.side == OrderSide.BUY
        # Reference price: ask for BUY fills, bid for SELL fills
        ref_price = tick.ask if is_buy else tick.bid

        result = dict(request)

        if is_buy:
            # BUY SL must be at least min_distance below ref_price
            if result.get("sl") is not None:
                max_valid_sl = round(ref_price - min_distance, digits)
                if result["sl"] > max_valid_sl:
                    log.warning(
                        "mt5_sl_adjusted_min_distance",
                        symbol=order.symbol,
                        side="BUY",
                        original_sl=result["sl"],
                        adjusted_sl=max_valid_sl,
                        ref_price=round(ref_price, digits),
                        min_distance_pips=self._min_stop_pips,
                    )
                    result["sl"] = max_valid_sl
            # BUY TP must be at least min_distance above ref_price
            if result.get("tp") is not None:
                min_valid_tp = round(ref_price + min_distance, digits)
                if result["tp"] < min_valid_tp:
                    log.warning(
                        "mt5_tp_adjusted_min_distance",
                        symbol=order.symbol,
                        side="BUY",
                        original_tp=result["tp"],
                        adjusted_tp=min_valid_tp,
                        ref_price=round(ref_price, digits),
                        min_distance_pips=self._min_stop_pips,
                    )
                    result["tp"] = min_valid_tp
        else:
            # SELL SL must be at least min_distance above ref_price
            if result.get("sl") is not None:
                min_valid_sl = round(ref_price + min_distance, digits)
                if result["sl"] < min_valid_sl:
                    log.warning(
                        "mt5_sl_adjusted_min_distance",
                        symbol=order.symbol,
                        side="SELL",
                        original_sl=result["sl"],
                        adjusted_sl=min_valid_sl,
                        ref_price=round(ref_price, digits),
                        min_distance_pips=self._min_stop_pips,
                    )
                    result["sl"] = min_valid_sl
            # SELL TP must be at least min_distance below ref_price
            if result.get("tp") is not None:
                max_valid_tp = round(ref_price - min_distance, digits)
                if result["tp"] > max_valid_tp:
                    log.warning(
                        "mt5_tp_adjusted_min_distance",
                        symbol=order.symbol,
                        side="SELL",
                        original_tp=result["tp"],
                        adjusted_tp=max_valid_tp,
                        ref_price=round(ref_price, digits),
                        min_distance_pips=self._min_stop_pips,
                    )
                    result["tp"] = max_valid_tp

        return result

    # ── Filling mode ──────────────────────────────────────────────────────────

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

    # ── Request builder ───────────────────────────────────────────────────────

    def _build_mt5_request(self, order: Order) -> dict:
        """Build an MQL5 trade request dict from an Order.

        All prices are normalised to the symbol's ``digits`` decimal places
        (prevents retcode=10016 TRADE_RETCODE_INVALID_STOPS).

        Volume is normalised to the broker's ``volume_step`` and clamped to
        [volume_min, volume_max] (prevents retcode=10014
        TRADE_RETCODE_INVALID_VOLUME).  This is a safety net: the upstream
        chain (PositionSizer, drawdown scaling, intermarket scaling) may leave
        unrounded Decimal values.
        """
        is_buy = order.side == OrderSide.BUY
        action = _MT5_TRADE_ACTION_DEAL
        order_type = _MT5_ORDER_TYPE_BUY if is_buy else _MT5_ORDER_TYPE_SELL

        digits = self._symbol_digits(order.symbol)

        def _norm_price(price: Decimal) -> float:
            return round(float(price), digits)

        volume = float(self._normalize_volume(order.lot_size, order.symbol))

        request: dict = {
            "action":       action,
            "symbol":       order.symbol,
            "volume":       volume,
            "type":         order_type,
            "deviation":    self._deviation,
            "magic":        self._magic_number,
            "comment":      order.comment or "metatrade",
            "type_filling": self._get_filling_mode(order.symbol),
        }
        if order.price is not None:
            request["price"] = _norm_price(order.price)
        if order.stop_loss is not None:
            request["sl"] = _norm_price(order.stop_loss)
        if order.take_profit is not None:
            request["tp"] = _norm_price(order.take_profit)

        # Enforce minimum stop distance: pushes SL/TP out if too close to
        # current price (prevents retcode=10016 and immediate stop-out).
        request = self._adjust_stops_for_min_distance(order, request, digits)

        return request

    # ── Position management ───────────────────────────────────────────────────

    def get_latest_tick(self, symbol: str) -> tuple[int, float, float] | None:
        """Return ``(time_msc, bid, ask)`` for the latest tick on *symbol*.

        ``time_msc`` is MT5's ``tick.time_msc`` — milliseconds since epoch.
        Callers compare successive values to detect whether a new tick arrived.
        Returns None when not connected or the symbol has no tick yet.
        """
        if not self._connected:
            return None
        mt5 = _get_mt5()
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        # time_msc may not exist on all MT5 builds; fall back to time * 1000
        time_msc = getattr(tick, "time_msc", None) or (tick.time * 1000)
        return int(time_msc), float(tick.bid), float(tick.ask)

    def get_open_positions(self, symbol: str | None = None) -> list[dict]:
        """Return all open positions (optionally filtered by symbol).

        Each dict contains:
            ticket, symbol, volume, type (0=buy, 1=sell),
            price_open, sl, tp, profit, time_utc (datetime).
        Returns an empty list when not connected or no positions exist.
        """
        if not self._connected:
            return []
        mt5 = _get_mt5()
        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        if positions is None:
            return []
        result = []
        for p in positions:
            result.append({
                "ticket":      p.ticket,
                "symbol":      p.symbol,
                "volume":      p.volume,
                "type":        p.type,          # 0 = buy, 1 = sell
                "price_open":  p.price_open,
                "sl":          p.sl,
                "tp":          p.tp,
                "profit":      p.profit,
                "magic":       p.magic,
                "comment":     p.comment,
                "time_utc":    datetime.fromtimestamp(p.time, tz=UTC),
            })
        return result

    def get_deals_for_position(self, position_ticket: int) -> list[dict]:
        """Return all deals for a position ticket (entry + exit).

        Useful for computing realized PnL of a position closed by MT5 natively
        (e.g. SL/TP hit on the broker server).

        Each dict: ticket, time_utc, type, entry (0=IN 1=OUT 2=INOUT),
                   volume, price, profit, commission, swap, symbol.
        """
        if not self._connected:
            return []
        mt5 = _get_mt5()
        deals = mt5.history_deals_get(position=position_ticket)
        if deals is None:
            return []
        result = []
        for d in deals:
            result.append({
                "ticket":     d.ticket,
                "time_utc":   datetime.fromtimestamp(d.time, tz=UTC),
                "type":       d.type,
                "entry":      d.entry,   # 0=IN, 1=OUT, 2=INOUT
                "volume":     d.volume,
                "price":      d.price,
                "profit":     d.profit,
                "commission": d.commission,
                "swap":       d.swap,
                "symbol":     d.symbol,
            })
        return result

    def get_deals_since(self, from_dt: datetime, to_dt: datetime) -> list[dict]:
        """Return all deals in [from_dt, to_dt] (inclusive).

        Useful for reconstructing daily PnL after a restart.
        Each dict: ticket, position, time_utc, type, entry (0=IN 1=OUT 2=INOUT),
                   volume, price, profit, commission, swap, symbol, comment.
        """
        if not self._connected:
            return []
        mt5 = _get_mt5()
        deals = mt5.history_deals_get(from_dt, to_dt)
        if deals is None:
            return []
        result = []
        for d in deals:
            result.append({
                "ticket":     d.ticket,
                "position":   d.position_id,
                "time_utc":   datetime.fromtimestamp(d.time, tz=UTC),
                "type":       d.type,
                "entry":      d.entry,
                "volume":     d.volume,
                "price":      d.price,
                "profit":     d.profit,
                "commission": d.commission,
                "swap":       d.swap,
                "symbol":     d.symbol,
                "comment":    d.comment,
            })
        return result

    def modify_sl_tp(
        self,
        ticket: int,
        symbol: str,
        new_sl: float | None,
        new_tp: float | None,
    ) -> bool:
        """Modify the stop-loss and/or take-profit of an open position.

        Args:
            ticket:  MT5 position ticket.
            symbol:  Symbol (needed for price normalisation).
            new_sl:  New stop-loss price, or None to leave unchanged.
            new_tp:  New take-profit price, or None to leave unchanged.

        Returns:
            True if MT5 accepted the modification, False otherwise.
        """
        if not self._connected:
            return False
        mt5 = _get_mt5()
        digits = self._symbol_digits(symbol)

        # Fetch current position to fill in unchanged SL/TP
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            log.warning("mt5_modify_sltp_position_not_found", ticket=ticket)
            return False
        pos = positions[0]

        sl_val = round(new_sl, digits) if new_sl is not None else pos.sl
        tp_val = round(new_tp, digits) if new_tp is not None else pos.tp

        # Short-circuit: if the rounded values equal what MT5 already has,
        # skip the network round-trip entirely (avoids retcode=10025 spam).
        if abs(sl_val - pos.sl) < 10 ** -digits and abs(tp_val - pos.tp) < 10 ** -digits:
            log.debug(
                "mt5_modify_sltp_skipped_no_change",
                ticket=ticket, sl=sl_val, tp=tp_val,
            )
            return True

        # Guard against retcode=10016 (INVALID_STOPS): skip the modification
        # this tick if the new SL or TP would violate the broker's minimum
        # stop distance from the current price.  The trailing stop will retry
        # on the next tick once the price moves far enough away.
        tick = mt5.symbol_info_tick(symbol)
        info = mt5.symbol_info(symbol)
        if tick is not None and info is not None:
            point: float = getattr(info, "point", 0.00001)
            stops_level: int = getattr(info, "stops_level", 0)
            pip_size = 0.0001 if point < 0.001 else 0.01
            pip_points = max(1, round(pip_size / point))
            min_pts = max(stops_level, int(self._min_stop_pips * pip_points))
            min_dist = min_pts * point
            is_buy = pos.type == 0  # 0=POSITION_TYPE_BUY
            ref = tick.ask if is_buy else tick.bid
            if is_buy:
                sl_too_close = sl_val > ref - min_dist
                tp_too_close = tp_val != 0.0 and tp_val < ref + min_dist
            else:
                sl_too_close = sl_val < ref + min_dist
                tp_too_close = tp_val != 0.0 and tp_val > ref - min_dist
            if sl_too_close or tp_too_close:
                log.debug(
                    "mt5_modify_sltp_skipped_too_close",
                    ticket=ticket,
                    symbol=symbol,
                    sl=sl_val,
                    tp=tp_val,
                    ref_price=round(ref, digits),
                    min_dist_pips=self._min_stop_pips,
                    sl_too_close=sl_too_close,
                    tp_too_close=tp_too_close,
                )
                return True

        request = {
            "action":   _MT5_TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol":   symbol,
            "sl":       sl_val,
            "tp":       tp_val,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode not in (
            _MT5_RETCODE_DONE, _MT5_RETCODE_DONE_PARTIAL
        ):
            retcode = result.retcode if result else -1
            comment = result.comment if result else "none"
            # retcode=10025 means MT5 itself detected no change after rounding
            # on its side — treat as silent success, not a failure.
            if retcode == 10025:
                log.debug(
                    "mt5_modify_sltp_no_change",
                    ticket=ticket, symbol=symbol, sl=sl_val, tp=tp_val,
                )
                return True
            log.warning(
                "mt5_modify_sltp_failed",
                ticket=ticket,
                symbol=symbol,
                new_sl=sl_val,
                new_tp=tp_val,
                retcode=retcode,
                comment=comment,
            )
            return False
        log.info(
            "mt5_modify_sltp_ok",
            ticket=ticket,
            symbol=symbol,
            new_sl=sl_val,
            new_tp=tp_val,
        )
        return True

    def close_position_by_ticket(
        self,
        ticket: int,
        symbol: str,
        volume: float,
        is_buy: bool,
    ) -> bool:
        """Close (fully or partially) an open position by ticket.

        Args:
            ticket:  MT5 position ticket.
            symbol:  Symbol.
            volume:  Lots to close (must be <= position volume).
            is_buy:  True if the position to close is a BUY (we'll sell to close).

        Returns:
            True if accepted, False otherwise.
        """
        if not self._connected:
            return False
        mt5 = _get_mt5()
        close_type = _MT5_ORDER_TYPE_SELL if is_buy else _MT5_ORDER_TYPE_BUY
        norm_vol = float(self._normalize_volume(Decimal(str(volume)), symbol))

        request = {
            "action":       _MT5_TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       norm_vol,
            "type":         close_type,
            "position":     ticket,
            "deviation":    self._deviation,
            "magic":        self._magic_number,
            "comment":      "exit_engine",
            "type_filling": self._get_filling_mode(symbol),
        }
        result = mt5.order_send(request)
        if result is None or result.retcode not in (
            _MT5_RETCODE_DONE, _MT5_RETCODE_DONE_PARTIAL
        ):
            retcode = result.retcode if result else -1
            comment = result.comment if result else "none"
            log.warning(
                "mt5_close_position_failed",
                ticket=ticket,
                symbol=symbol,
                volume=norm_vol,
                retcode=retcode,
                comment=comment,
            )
            return False
        log.info(
            "mt5_close_position_ok",
            ticket=ticket,
            symbol=symbol,
            volume=norm_vol,
        )
        return True
