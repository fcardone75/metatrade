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
