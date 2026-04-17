"""Market data domain models (not contracts — these are internal to the module)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from metatrade.core.enums import Timeframe


@dataclass(frozen=True)
class BarQuery:
    """Parameters for a bar retrieval query."""

    symbol: str
    timeframe: Timeframe
    date_from: datetime
    date_to: datetime

    def __post_init__(self) -> None:
        if self.date_from.tzinfo is None or self.date_to.tzinfo is None:
            raise ValueError("BarQuery dates must be timezone-aware (UTC)")
        if self.date_from >= self.date_to:
            raise ValueError(
                f"BarQuery.date_from ({self.date_from}) must be < date_to ({self.date_to})"
            )
        if not self.symbol:
            raise ValueError("BarQuery.symbol cannot be empty")


@dataclass(frozen=True)
class SymbolCoverage:
    """How much historical data we have stored for a symbol/timeframe pair."""

    symbol: str
    timeframe: Timeframe
    oldest_bar_utc: datetime | None
    newest_bar_utc: datetime | None
    total_bars: int

    @property
    def has_data(self) -> bool:
        return self.total_bars > 0

    @property
    def coverage_days(self) -> float | None:
        if self.oldest_bar_utc is None or self.newest_bar_utc is None:
            return None
        delta = self.newest_bar_utc - self.oldest_bar_utc
        return delta.total_seconds() / 86400


@dataclass
class CollectionReport:
    """Summary of a data collection run."""

    started_at_utc: datetime
    completed_at_utc: datetime | None = None
    symbols_requested: list[str] = field(default_factory=list)
    symbols_succeeded: list[str] = field(default_factory=list)
    symbols_failed: list[str] = field(default_factory=list)
    bars_collected: int = 0
    bars_skipped: int = 0        # already in storage
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = len(self.symbols_requested)
        if total == 0:
            return 1.0
        return len(self.symbols_succeeded) / total

    @property
    def duration_seconds(self) -> float | None:
        if self.completed_at_utc is None:
            return None
        return (self.completed_at_utc - self.started_at_utc).total_seconds()


@dataclass(frozen=True)
class RawBar:
    """Raw OHLCV data as received from a broker before normalization.

    Fields use native Python types — normalization converts to Decimal/datetime.
    timestamp_unix is seconds since Unix epoch (UTC).
    """

    symbol: str
    timeframe: Timeframe
    timestamp_unix: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float | None = None


@dataclass(frozen=True)
class GapInfo:
    """A detected gap in a bar series."""

    symbol: str
    timeframe: Timeframe
    gap_start_utc: datetime     # Last bar before the gap
    gap_end_utc: datetime       # First bar after the gap
    expected_bars_missing: int

    @property
    def gap_duration_seconds(self) -> float:
        return (self.gap_end_utc - self.gap_start_utc).total_seconds()
