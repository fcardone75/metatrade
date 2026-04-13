"""Market data domain errors."""

from metatrade.core.errors import DataError, MetaTradeError


class MarketDataError(MetaTradeError):
    """Base for market data errors."""


class FeedError(MarketDataError):
    """Real-time feed failed or disconnected."""


class FeedNotInitializedError(FeedError):
    """Feed has not been initialized yet."""


class BarStoreError(MarketDataError):
    """DuckDB bar storage operation failed."""


class AuditStoreError(MarketDataError):
    """SQLite audit storage operation failed."""


class NormalizationError(MarketDataError):
    """Raw broker data could not be converted to a Bar contract."""


class CollectionError(MarketDataError):
    """Historical data collection from broker failed."""


class MT5NotAvailableError(MarketDataError):
    """MetaTrader5 Python library is not installed or not on Windows."""


class MT5NotConnectedError(MarketDataError):
    """MT5 terminal is installed but not connected to a broker account."""


class SymbolNotFoundError(MarketDataError):
    """Symbol is not available in the broker's symbol list."""


class InsufficientHistoryError(DataError):
    """Broker does not have enough history for the requested date range."""
