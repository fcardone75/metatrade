"""Read-only MT5 helpers for the dashboard/API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from metatrade.core.log import get_logger
from metatrade.market_data.config import MarketDataConfig

log = get_logger(__name__)


class MT5RuntimeReader:
    """Best-effort reader for current MT5 account and open positions."""

    def __init__(self) -> None:
        self._cfg = MarketDataConfig.load()

    def _initialize(self) -> Any | None:
        try:
            import MetaTrader5 as mt5  # type: ignore[import]
        except ImportError:
            return None

        kwargs: dict[str, Any] = {"timeout": self._cfg.mt5_timeout_ms}
        if self._cfg.mt5_path:
            kwargs["path"] = self._cfg.mt5_path

        ok = mt5.initialize(**kwargs)
        if not ok:
            return None
        return mt5

    def get_account(self) -> dict[str, Any] | None:
        mt5 = self._initialize()
        if mt5 is None:
            return None
        try:
            info = mt5.account_info()
            if info is None:
                return None
            return {
                "login": getattr(info, "login", None),
                "server": getattr(info, "server", None),
                "balance": getattr(info, "balance", None),
                "equity": getattr(info, "equity", None),
                "margin_free": getattr(info, "margin_free", None),
                "margin": getattr(info, "margin", None),
                "profit": getattr(info, "profit", None),
                "currency": getattr(info, "currency", None),
                "trade_allowed": getattr(info, "trade_allowed", None),
                "trade_expert": getattr(info, "trade_expert", None),
                "company": getattr(info, "company", None),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            mt5.shutdown()

    def get_open_positions(self) -> list[dict[str, Any]]:
        mt5 = self._initialize()
        if mt5 is None:
            return []
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            result: list[dict[str, Any]] = []
            for position in positions:
                result.append(
                    {
                        "ticket": getattr(position, "ticket", None),
                        "symbol": getattr(position, "symbol", None),
                        "type": getattr(position, "type", None),
                        "volume": getattr(position, "volume", None),
                        "price_open": getattr(position, "price_open", None),
                        "price_current": getattr(position, "price_current", None),
                        "sl": getattr(position, "sl", None),
                        "tp": getattr(position, "tp", None),
                        "profit": getattr(position, "profit", None),
                        "swap": getattr(position, "swap", None),
                        "comment": getattr(position, "comment", None),
                        "magic": getattr(position, "magic", None),
                        "time": getattr(position, "time", None),
                    }
                )
            return result
        finally:
            mt5.shutdown()

    def get_closed_deals(
        self,
        limit: int = 500,
        days_back: int | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return closed deals from MT5 history.

        Args:
            limit:     Maximum number of results (applied after date filtering).
            days_back: If set, fetch deals from now minus this many days.
            date_from: Explicit start datetime (UTC). Overrides days_back.
            date_to:   Explicit end datetime (UTC). Defaults to now.
        """
        from datetime import timedelta

        mt5 = self._initialize()
        if mt5 is None:
            return []
        try:
            if date_to is None:
                date_to = datetime.now(timezone.utc)
            if date_from is None:
                days = days_back if days_back is not None else 90
                date_from = date_to - timedelta(days=days)
            # Richiede il periodo nello storico terminale (senza questo spesso history_deals_get è vuoto).
            if hasattr(mt5, "history_select"):
                if not mt5.history_select(date_from, date_to):
                    try:
                        err = mt5.last_error()
                    except Exception:
                        err = None
                    log.warning("mt5_history_select_failed", error=err)
            deals = mt5.history_deals_get(date_from, date_to)
            if deals is None:
                return []
            # Keep only DEAL_ENTRY_OUT (2) = closing deals, skip in/inout if desired
            # type 0=buy, 1=sell; entry 0=in, 1=out, 2=inout
            result: list[dict[str, Any]] = []
            for d in deals:
                entry = getattr(d, "entry", None)
                if entry not in (1, 2):   # 1=OUT, 2=INOUT
                    continue
                result.append({
                    "ticket":      getattr(d, "ticket", None),
                    "order":       getattr(d, "order", None),
                    "position_id": getattr(d, "position_id", None),
                    "symbol":      getattr(d, "symbol", None),
                    "type":        getattr(d, "type", None),   # 0=buy, 1=sell
                    "volume":      getattr(d, "volume", None),
                    "price":       getattr(d, "price", None),  # close price
                    "profit":      getattr(d, "profit", None),
                    "swap":        getattr(d, "swap", None),
                    "commission":  getattr(d, "commission", None),
                    "comment":     getattr(d, "comment", None),
                    "time":        getattr(d, "time", None),   # Unix timestamp
                })
            # Most recent first
            result.sort(key=lambda x: x["time"] or 0, reverse=True)
            return result[:limit]
        finally:
            mt5.shutdown()

    def close_position(self, ticket: int) -> dict[str, Any]:
        """Close an open position by ticket using a market order in the opposite direction.

        Returns a dict with keys: ok (bool), retcode (int), comment (str).
        """
        mt5 = self._initialize()
        if mt5 is None:
            return {"ok": False, "retcode": -1, "comment": "MT5 not available"}
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return {"ok": False, "retcode": -1, "comment": f"Position {ticket} not found"}
            pos = positions[0]
            symbol  = pos.symbol
            volume  = pos.volume
            pos_type = pos.type  # 0=BUY, 1=SELL

            # Close: opposite direction
            # ORDER_TYPE_SELL=1 to close a BUY, ORDER_TYPE_BUY=0 to close a SELL
            close_type = 1 if pos_type == 0 else 0

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return {"ok": False, "retcode": -1, "comment": f"No tick for {symbol}"}
            price = tick.bid if close_type == 1 else tick.ask

            # Detect supported filling mode
            filling = 2  # RETURN fallback
            info = mt5.symbol_info(symbol)
            if info is not None:
                flags = info.filling_mode
                if flags & 1:
                    filling = 0   # FOK
                elif flags & 2:
                    filling = 1   # IOC

            request = {
                "action":       1,        # TRADE_ACTION_DEAL
                "symbol":       symbol,
                "volume":       volume,
                "type":         close_type,
                "position":     ticket,
                "price":        price,
                "deviation":    20,
                "type_filling": filling,
                "comment":      "dashboard close",
            }
            result = mt5.order_send(request)
            if result is None:
                err = mt5.last_error()
                return {"ok": False, "retcode": -1, "comment": str(err)}
            ok = result.retcode in (10009, 10010)
            return {"ok": ok, "retcode": result.retcode, "comment": result.comment}
        finally:
            mt5.shutdown()
