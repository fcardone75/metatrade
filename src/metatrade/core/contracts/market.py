"""Market data contracts: Instrument, Bar, Tick.

Design notes:
- All prices are Decimal to avoid float rounding errors on financial data.
- Bar validates OHLC invariants on construction (high >= low, open/close within range).
- Tick validates bid <= ask.
- Every timestamp is UTC-aware; naive datetimes raise ValueError immediately.
- Bar carries both `timestamp_utc` (bar open time) and `received_at_utc`
  (when we received it from the feed) to enable latency tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from metatrade.core.enums import Timeframe


@dataclass(frozen=True)
class Instrument:
    """Static metadata for a tradeable Forex instrument."""

    symbol: str             # e.g. "EURUSD"
    base_currency: str      # e.g. "EUR"
    quote_currency: str     # e.g. "USD"
    pip_size: Decimal       # e.g. Decimal("0.0001") for EURUSD
    lot_size: Decimal       # e.g. Decimal("100000") — units per standard lot
    min_lot: Decimal        # e.g. Decimal("0.01")
    max_lot: Decimal        # e.g. Decimal("500.0")
    lot_step: Decimal       # e.g. Decimal("0.01")
    digits: int             # Decimal places for price display

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("Instrument.symbol cannot be empty")
        if self.pip_size <= 0:
            raise ValueError(f"Instrument.pip_size must be > 0, got {self.pip_size}")
        if self.lot_size <= 0:
            raise ValueError(f"Instrument.lot_size must be > 0, got {self.lot_size}")
        if self.min_lot <= 0:
            raise ValueError(f"Instrument.min_lot must be > 0, got {self.min_lot}")
        if self.max_lot <= 0:
            raise ValueError(f"Instrument.max_lot must be > 0, got {self.max_lot}")
        if self.min_lot > self.max_lot:
            raise ValueError(
                f"Instrument.min_lot ({self.min_lot}) > max_lot ({self.max_lot})"
            )
        if self.lot_step <= 0:
            raise ValueError(f"Instrument.lot_step must be > 0, got {self.lot_step}")
        if self.digits < 0:
            raise ValueError(f"Instrument.digits must be >= 0, got {self.digits}")

    def pip_value_per_lot(self, account_currency: str = "USD") -> Decimal:
        """Pip value in account currency per standard lot.

        For pairs quoted in USD (EURUSD, GBPUSD): pip_size * lot_size.
        For JPY pairs and crosses, the caller must pass the conversion rate.
        This method handles the simple USD-quoted case only.
        """
        if self.quote_currency == account_currency:
            return self.pip_size * self.lot_size
        raise NotImplementedError(
            f"pip_value_per_lot requires a rate converter for "
            f"{self.quote_currency}/{account_currency} pairs"
        )


@dataclass(frozen=True)
class Bar:
    """One OHLCV candle for a symbol/timeframe combination.

    Invariants enforced on construction:
      high >= low
      low <= open <= high
      low <= close <= high
      volume >= 0
    """

    symbol: str
    timeframe: Timeframe
    timestamp_utc: datetime         # Bar OPEN time (not close)
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    spread: Decimal | None = None
    received_at_utc: datetime | None = None  # When feed delivered this bar

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("Bar.timestamp_utc must be timezone-aware (UTC)")
        if self.received_at_utc is not None and self.received_at_utc.tzinfo is None:
            raise ValueError("Bar.received_at_utc must be timezone-aware (UTC)")
        if self.high < self.low:
            raise ValueError(f"Bar.high ({self.high}) < low ({self.low})")
        if not (self.low <= self.open <= self.high):
            raise ValueError(
                f"Bar.open ({self.open}) outside [low={self.low}, high={self.high}]"
            )
        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"Bar.close ({self.close}) outside [low={self.low}, high={self.high}]"
            )
        if self.volume < 0:
            raise ValueError(f"Bar.volume cannot be negative, got {self.volume}")
        if self.spread is not None and self.spread < 0:
            raise ValueError(f"Bar.spread cannot be negative, got {self.spread}")

    @property
    def body_size(self) -> Decimal:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> Decimal:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> Decimal:
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> Decimal:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        """Body is less than 10% of total range."""
        if self.range == 0:
            return True
        return self.body_size / self.range < Decimal("0.1")


@dataclass(frozen=True)
class Tick:
    """A single bid/ask price update."""

    symbol: str
    timestamp_utc: datetime
    bid: Decimal
    ask: Decimal
    last: Decimal | None = None
    volume: Decimal | None = None

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("Tick.timestamp_utc must be timezone-aware (UTC)")
        if self.bid <= 0:
            raise ValueError(f"Tick.bid must be > 0, got {self.bid}")
        if self.ask <= 0:
            raise ValueError(f"Tick.ask must be > 0, got {self.ask}")
        if self.ask < self.bid:
            raise ValueError(f"Tick.ask ({self.ask}) < bid ({self.bid})")

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid
