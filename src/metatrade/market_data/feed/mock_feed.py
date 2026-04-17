"""Mock feed — deterministic bar generator for unit tests.

Generates synthetic OHLCV bars with configurable parameters.
No external dependencies, no files, no network.

Used in:
- Unit tests for analysis modules (need N bars of input).
- Unit tests for the consensus engine.
- Unit tests for the risk manager.

The generated bars form a simple random walk around a base price,
producing realistic-looking OHLCV data with valid invariants.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime
from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.market_data.interface import IMarketDataFeed


class MockFeed(IMarketDataFeed):
    """Generates deterministic synthetic bars for testing.

    Args:
        symbol:     Symbol name for generated bars.
        timeframe:  Timeframe for generated bars.
        base_price: Starting mid-price (e.g. Decimal("1.10000")).
        volatility: Max price movement per bar as a fraction of base_price.
        spread:     Fixed spread in price units.
        seed:       Random seed for reproducibility.
    """

    def __init__(
        self,
        symbol: str = "EURUSD",
        timeframe: Timeframe = Timeframe.H1,
        base_price: Decimal = Decimal("1.10000"),
        volatility: float = 0.0005,
        spread: Decimal = Decimal("0.00020"),
        seed: int = 42,
    ) -> None:
        self._symbol = symbol
        self._timeframe = timeframe
        self._base_price = base_price
        self._volatility = volatility
        self._spread = spread
        self._rng = random.Random(seed)
        self._ready = False

    def initialize(self) -> None:
        self._ready = True

    def is_connected(self) -> bool:
        return self._ready

    def shutdown(self) -> None:
        self._ready = False

    def get_latest_bars(self, symbol: str, timeframe: Timeframe, count: int) -> list[Bar]:
        """Generate `count` bars ending at a fixed reference time."""
        end_ts = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
        step = self._timeframe.seconds
        bars = []
        price = self._base_price
        for i in range(count):
            ts_unix = int(end_ts.timestamp()) - (count - i - 1) * step
            bar, price = self._generate_bar(ts_unix, price)
            bars.append(bar)
        return bars

    def get_bars_range(
        self,
        symbol: str,
        timeframe: Timeframe,
        date_from: datetime,
        date_to: datetime,
    ) -> list[Bar]:
        """Generate bars filling the date range at the configured timeframe step."""
        step = self._timeframe.seconds
        ts = int(date_from.timestamp())
        end = int(date_to.timestamp())
        bars = []
        price = self._base_price
        while ts <= end:
            bar, price = self._generate_bar(ts, price)
            bars.append(bar)
            ts += step
        return bars

    def generate_bars(self, count: int, start_ts: datetime | None = None) -> list[Bar]:
        """Generate `count` consecutive bars from start_ts (or a default time)."""
        if start_ts is None:
            start_ts = datetime(2024, 1, 1, tzinfo=UTC)
        step = self._timeframe.seconds
        bars = []
        price = self._base_price
        ts = int(start_ts.timestamp())
        for _ in range(count):
            bar, price = self._generate_bar(ts, price)
            bars.append(bar)
            ts += step
        return bars

    # ── Internal ──────────────────────────────────────────────────────────────

    def _generate_bar(self, ts_unix: int, open_price: Decimal) -> tuple[Bar, Decimal]:
        """Generate one bar with random walk from open_price.
        Returns (bar, close_price) where close_price is the next open.
        """
        move = Decimal(str(self._rng.uniform(-self._volatility, self._volatility)))
        close = (open_price + move * open_price).quantize(Decimal("0.00001"))
        vol = Decimal(str(self._volatility))
        wick_range = float(vol * open_price / 2)
        high = max(open_price, close) + Decimal(
            str(abs(self._rng.uniform(0, wick_range)))
        ).quantize(Decimal("0.00001"))
        low = min(open_price, close) - Decimal(
            str(abs(self._rng.uniform(0, wick_range)))
        ).quantize(Decimal("0.00001"))
        low = max(low, Decimal("0.00001"))  # prices cannot go negative

        bar = Bar(
            symbol=self._symbol,
            timeframe=self._timeframe,
            timestamp_utc=datetime.fromtimestamp(ts_unix, tz=UTC),
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=Decimal(str(self._rng.randint(100, 5000))),
            spread=self._spread,
        )
        return bar, close
