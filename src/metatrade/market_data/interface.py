"""Market data module interfaces (abstract contracts).

These ABCs define what every implementation must provide.
Concrete implementations live in storage/, collectors/, feed/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.market_data.models import (
    BarQuery,
    CollectionReport,
    GapInfo,
    SymbolCoverage,
)


class IBarStore(ABC):
    """Persistent storage for historical OHLCV bars (DuckDB implementation)."""

    @abstractmethod
    def save_bars(self, bars: list[Bar]) -> int:
        """Persist bars. Returns number of bars actually written (skips duplicates)."""
        ...

    @abstractmethod
    def get_bars(self, query: BarQuery) -> list[Bar]:
        """Retrieve bars for a symbol/timeframe within a date range, ordered by time asc."""
        ...

    @abstractmethod
    def get_latest_bar_time(self, symbol: str, timeframe: Timeframe) -> datetime | None:
        """Return the timestamp of the most recent stored bar, or None if no data."""
        ...

    @abstractmethod
    def get_oldest_bar_time(self, symbol: str, timeframe: Timeframe) -> datetime | None:
        """Return the timestamp of the oldest stored bar, or None if no data."""
        ...

    @abstractmethod
    def get_coverage(self, symbol: str, timeframe: Timeframe) -> SymbolCoverage:
        """Return storage coverage statistics for a symbol/timeframe."""
        ...

    @abstractmethod
    def count_bars(self, symbol: str, timeframe: Timeframe) -> int:
        """Return the total number of stored bars for a symbol/timeframe."""
        ...

    @abstractmethod
    def delete_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        date_from: datetime,
        date_to: datetime,
    ) -> int:
        """Delete bars in a date range. Returns number deleted. Used for gap repair."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release the database connection."""
        ...


class IHistoricalCollector(ABC):
    """Collects historical bars from an external source (e.g. MT5, CSV)."""

    @abstractmethod
    def collect(
        self,
        symbol: str,
        timeframe: Timeframe,
        date_from: datetime,
        date_to: datetime,
    ) -> list[Bar]:
        """Fetch bars from the source for the given date range."""
        ...

    @abstractmethod
    def available_symbols(self) -> list[str]:
        """Return list of symbols available from this source."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the source is reachable and ready."""
        ...


class IMarketDataFeed(ABC):
    """Real-time or replay market data feed."""

    @abstractmethod
    def initialize(self) -> None:
        """Set up the feed connection. Raises FeedError on failure."""
        ...

    @abstractmethod
    def get_latest_bars(self, symbol: str, timeframe: Timeframe, count: int) -> list[Bar]:
        """Return the most recent `count` closed bars for symbol/timeframe."""
        ...

    @abstractmethod
    def get_bars_range(
        self,
        symbol: str,
        timeframe: Timeframe,
        date_from: datetime,
        date_to: datetime,
    ) -> list[Bar]:
        """Return bars within a date range."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if the feed is live and delivering data."""
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanly close the feed connection."""
        ...


class IGapDetector(ABC):
    """Detects gaps in stored bar series."""

    @abstractmethod
    def detect_gaps(
        self,
        symbol: str,
        timeframe: Timeframe,
        date_from: datetime,
        date_to: datetime,
    ) -> list[GapInfo]:
        """Return all gaps found in the stored bars for the given range."""
        ...
