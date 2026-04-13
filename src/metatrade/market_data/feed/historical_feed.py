"""Historical feed — serves bars from DuckDB for backtest and analysis warmup.

Used in BACKTEST mode and for loading the warmup window on startup (all modes).
Reads directly from the DuckDB bar store — no network calls, deterministic.
"""

from __future__ import annotations

from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.core.log import get_logger
from metatrade.market_data.errors import FeedNotInitializedError, InsufficientHistoryError
from metatrade.market_data.interface import IBarStore, IMarketDataFeed
from metatrade.market_data.models import BarQuery

log = get_logger(__name__)


class HistoricalFeed(IMarketDataFeed):
    """Serves historical bars from DuckDB storage.

    Suitable for:
    - Backtest: replays bars in chronological order.
    - Warmup: loads the initial N bars before live trading starts.
    """

    def __init__(self, store: IBarStore) -> None:
        self._store = store
        self._ready = False

    def initialize(self) -> None:
        self._ready = True
        log.info("historical_feed_initialized")

    def is_connected(self) -> bool:
        return self._ready

    def shutdown(self) -> None:
        self._ready = False

    def get_latest_bars(self, symbol: str, timeframe: Timeframe, count: int) -> list[Bar]:
        """Return the most recent `count` bars available in storage."""
        self._assert_ready()
        latest = self._store.get_latest_bar_time(symbol, timeframe)
        if latest is None:
            raise InsufficientHistoryError(
                message=f"No bars in storage for {symbol}/{timeframe.value}",
                code="NO_BARS_IN_STORE",
            )
        oldest = self._store.get_oldest_bar_time(symbol, timeframe)
        assert oldest is not None

        query = BarQuery(
            symbol=symbol,
            timeframe=timeframe,
            date_from=oldest,
            date_to=latest,
        )
        all_bars = self._store.get_bars(query)
        return all_bars[-count:] if len(all_bars) > count else all_bars

    def get_bars_range(
        self,
        symbol: str,
        timeframe: Timeframe,
        date_from: datetime,
        date_to: datetime,
    ) -> list[Bar]:
        self._assert_ready()
        return self._store.get_bars(
            BarQuery(symbol=symbol, timeframe=timeframe, date_from=date_from, date_to=date_to)
        )

    def _assert_ready(self) -> None:
        if not self._ready:
            raise FeedNotInitializedError(
                message="HistoricalFeed not initialized — call initialize() first",
                code="FEED_NOT_INITIALIZED",
            )
