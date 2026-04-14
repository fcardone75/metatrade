"""MT5 historical data collector.

Pulls OHLCV bars from MetaTrader 5 and stores them via IBarStore.

IMPORTANT: The MetaTrader5 Python library is Windows-only.
This module is designed to import cleanly on all platforms.
If MT5 is not available, MT5NotAvailableError is raised at collect() time,
not at import time — so the rest of the system (tests, backtest) can import
this module on macOS/Linux without issues.

Typical usage:
    collector = MT5Collector(store=bar_store, normalizer=normalizer)
    collector.initialize(login=12345, password="...", server="BrokerName-Demo")
    bars = collector.collect("EURUSD", Timeframe.H1, date_from, date_to)
    # bars are also persisted in bar_store automatically
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.core.log import get_logger
from metatrade.market_data.errors import (
    CollectionError,
    InsufficientHistoryError,
    MT5NotAvailableError,
    MT5NotConnectedError,
    SymbolNotFoundError,
)
from metatrade.market_data.interface import IBarStore, IHistoricalCollector
from metatrade.market_data.models import CollectionReport
from metatrade.market_data.normalizer.bar_normalizer import BarNormalizer

log = get_logger(__name__)

# Lazy import — resolved at runtime so non-Windows CI does not fail at import
_mt5: Any = None


def _get_mt5() -> Any:
    global _mt5
    if _mt5 is None:
        try:
            import MetaTrader5 as mt5
            _mt5 = mt5
        except ImportError as exc:
            raise MT5NotAvailableError(
                message=(
                    "MetaTrader5 Python library is not installed. "
                    "Run: pip install MetaTrader5  (Windows only)"
                ),
                code="MT5_NOT_INSTALLED",
            ) from exc
    return _mt5


# Map our Timeframe enum to MT5 TIMEFRAME constants
_TF_MAP: dict[Timeframe, int] = {
    Timeframe.M1:  1,
    Timeframe.M5:  5,
    Timeframe.M15: 15,
    Timeframe.M30: 30,
    Timeframe.H1:  16385,
    Timeframe.H4:  16388,
    Timeframe.D1:  16408,
    Timeframe.W1:  32769,
    Timeframe.MN1: 49153,
}


class MT5Collector(IHistoricalCollector):
    """Collects historical OHLCV bars from MetaTrader 5.

    Args:
        store:      Where to persist collected bars.
        normalizer: Converts raw MT5 arrays to Bar contracts.
        batch_size: Max bars per MT5 API call (MT5 has per-request limits).
    """

    def __init__(
        self,
        store: IBarStore | None,
        normalizer: BarNormalizer | None = None,
        batch_size: int = 50_000,
    ) -> None:
        self._store = store
        self._normalizer = normalizer or BarNormalizer()
        self._batch_size = batch_size
        self._initialized = False

    def initialize(
        self,
        login: int = 0,
        password: str = "",
        server: str = "",
        path: str = "",
        timeout: int = 60_000,
    ) -> None:
        """Connect to the MT5 terminal.

        If login/password/server are empty, MT5 uses the currently
        logged-in account in the open terminal.
        """
        mt5 = _get_mt5()

        kwargs: dict[str, Any] = {"timeout": timeout}
        if path:
            kwargs["path"] = path
        if login:
            kwargs["login"] = login
        if password:
            kwargs["password"] = password
        if server:
            kwargs["server"] = server

        if not mt5.initialize(**kwargs):
            error = mt5.last_error()
            raise MT5NotConnectedError(
                message=f"MT5 initialize() failed: {error}",
                code="MT5_INIT_FAILED",
                context={"error": str(error)},
            )

        info = mt5.account_info()
        if info is None:
            raise MT5NotConnectedError(
                message="MT5 connected but account_info() returned None — not logged in",
                code="MT5_NOT_LOGGED_IN",
            )

        self._initialized = True
        log.info(
            "mt5_connected",
            login=info.login,
            server=info.server,
            balance=info.balance,
            currency=info.currency,
        )

    def shutdown(self) -> None:
        if self._initialized:
            mt5 = _get_mt5()
            mt5.shutdown()
            self._initialized = False
            log.info("mt5_shutdown")

    # ── IHistoricalCollector ──────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            mt5 = _get_mt5()
            info = mt5.terminal_info()
            return info is not None and info.connected
        except MT5NotAvailableError:
            return False

    def available_symbols(self) -> list[str]:
        mt5 = _get_mt5()
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        return [s.name for s in symbols]

    def collect(
        self,
        symbol: str,
        timeframe: Timeframe,
        date_from: datetime,
        date_to: datetime,
    ) -> list[Bar]:
        """Fetch bars from MT5 and persist them in the bar store.

        Returns the list of Bar objects collected.
        Bars already in the store are still returned but not double-inserted
        (DuckDB upsert is idempotent).
        """
        self._assert_initialized()
        mt5 = _get_mt5()

        mt5_tf = _TF_MAP.get(timeframe)
        if mt5_tf is None:
            raise CollectionError(
                message=f"Unsupported timeframe: {timeframe}",
                code="UNSUPPORTED_TIMEFRAME",
            )

        # Validate symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise SymbolNotFoundError(
                message=f"Symbol {symbol!r} not found in MT5",
                code="SYMBOL_NOT_FOUND",
                context={"symbol": symbol},
            )

        log.info(
            "collecting_bars",
            symbol=symbol,
            timeframe=timeframe.value,
            date_from=date_from.isoformat(),
            date_to=date_to.isoformat(),
        )

        rates = mt5.copy_rates_range(symbol, mt5_tf, date_from, date_to)

        if rates is None or len(rates) == 0:
            error = mt5.last_error()
            if error[0] != 1:  # 1 = ERR_SUCCESS (no data, but not an error)
                raise InsufficientHistoryError(
                    message=f"MT5 returned no data for {symbol}/{timeframe.value}: {error}",
                    code="MT5_NO_DATA",
                    context={"symbol": symbol, "error": str(error)},
                )
            log.warning("mt5_no_data", symbol=symbol, timeframe=timeframe.value)
            return []

        bars, skipped = self._normalizer.normalize_mt5_rates(symbol, timeframe, rates)
        saved = self._store.save_bars(bars) if self._store is not None else 0

        log.info(
            "collection_complete",
            symbol=symbol,
            timeframe=timeframe.value,
            bars_fetched=len(rates),
            bars_valid=len(bars),
            bars_skipped=skipped,
            bars_saved=saved,
        )

        return bars

    def collect_multiple(
        self,
        symbols: list[str],
        timeframes: list[Timeframe],
        date_from: datetime,
        date_to: datetime,
    ) -> CollectionReport:
        """Collect bars for all symbol/timeframe combinations."""
        from metatrade.core.clock import SystemClock
        clock = SystemClock()

        report = CollectionReport(
            started_at_utc=clock.now_utc(),
            symbols_requested=symbols,
        )

        for symbol in symbols:
            for tf in timeframes:
                try:
                    bars = self.collect(symbol, tf, date_from, date_to)
                    report.bars_collected += len(bars)
                    if symbol not in report.symbols_succeeded:
                        report.symbols_succeeded.append(symbol)
                except Exception as exc:
                    log.error(
                        "collection_failed",
                        symbol=symbol,
                        timeframe=tf.value,
                        error=str(exc),
                    )
                    report.errors.append(f"{symbol}/{tf.value}: {exc}")
                    if symbol not in report.symbols_failed:
                        report.symbols_failed.append(symbol)

        report.completed_at_utc = clock.now_utc()
        log.info(
            "collection_report",
            bars_collected=report.bars_collected,
            symbols_ok=len(report.symbols_succeeded),
            symbols_failed=len(report.symbols_failed),
            duration_s=report.duration_seconds,
        )
        return report

    # ── Internal ──────────────────────────────────────────────────────────────

    def _assert_initialized(self) -> None:
        if not self._initialized:
            raise MT5NotConnectedError(
                message="MT5Collector not initialized — call initialize() first",
                code="MT5_NOT_INITIALIZED",
            )
