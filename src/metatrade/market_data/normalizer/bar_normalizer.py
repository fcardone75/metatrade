"""Bar normalizer — converts raw broker data to Bar contracts.

MT5 returns OHLCV data as a numpy structured array with this dtype:
    [('time', '<i8'), ('open', '<f8'), ('high', '<f8'), ('low', '<f8'),
     ('close', '<f8'), ('tick_volume', '<u8'), ('spread', '<i4'), ('real_volume', '<u8')]

Where:
    time         = Unix timestamp (seconds, UTC)
    open/high/low/close = float64 prices
    tick_volume  = number of ticks in the bar
    spread       = spread in POINTS (not pips — divide by 10 for 5-digit brokers)
    real_volume  = actual traded volume (0 for most Forex brokers)

This normalizer converts the raw array rows into validated Bar contracts.
It is also used by the CSV collector to convert pandas DataFrame rows.

Key invariants enforced:
- Prices are quantized to `price_digits` decimal places.
- High >= Low, Open and Close within [Low, High].
- Volume >= 0.
- Corrupt rows are skipped and counted, never silently passed through.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.core.log import get_logger
from metatrade.market_data.errors import NormalizationError
from metatrade.market_data.models import RawBar

log = get_logger(__name__)


class BarNormalizer:
    """Converts raw OHLCV data into validated Bar contracts.

    Args:
        price_digits: Number of decimal places to quantize prices to.
                      5 for most FX pairs (e.g. EURUSD = 1.10000),
                      3 for JPY pairs (e.g. USDJPY = 150.000).
        spread_divisor: Factor to convert raw spread points to price units.
                        10 for 5-digit brokers (standard), 1 for 4-digit.
        skip_corrupt: If True, log and skip corrupt rows instead of raising.
                      Set False in tests to surface bad data immediately.
    """

    def __init__(
        self,
        price_digits: int = 5,
        spread_divisor: int = 10,
        skip_corrupt: bool = True,
    ) -> None:
        if price_digits < 1 or price_digits > 8:
            raise ValueError(f"price_digits must be in [1, 8], got {price_digits}")
        self._price_digits = price_digits
        self._quant = Decimal(10) ** -price_digits
        self._spread_divisor = float(spread_divisor)
        self._skip_corrupt = skip_corrupt

    # ── Public API ────────────────────────────────────────────────────────────

    def normalize_mt5_rates(
        self,
        symbol: str,
        timeframe: Timeframe,
        rates: Any,  # numpy structured array from MetaTrader5.copy_rates_*
    ) -> tuple[list[Bar], int]:
        """Convert MT5 numpy rates array to Bar list.

        Returns:
            (bars, skipped_count) — bars is the valid list,
            skipped_count is how many rows were corrupt and dropped.
        """
        bars: list[Bar] = []
        skipped = 0

        for row in rates:
            try:
                raw = RawBar(
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp_unix=int(row["time"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["tick_volume"]),
                    spread=float(row["spread"]) if row["spread"] > 0 else None,  # raw points
                )
                bars.append(self._raw_to_bar(raw))
            except Exception as exc:
                skipped += 1
                if not self._skip_corrupt:
                    raise NormalizationError(
                        message=f"Corrupt MT5 row for {symbol}: {exc}",
                        code="NORMALIZATION_CORRUPT_ROW",
                        context={"symbol": symbol, "timeframe": timeframe.value},
                    ) from exc
                log.warning(
                    "bar_normalization_skipped",
                    symbol=symbol,
                    timeframe=timeframe.value,
                    reason=str(exc),
                )

        if skipped:
            log.warning(
                "bars_skipped_during_normalization",
                symbol=symbol,
                total_skipped=skipped,
                total_valid=len(bars),
            )

        return bars, skipped

    def normalize_csv_row(
        self,
        symbol: str,
        timeframe: Timeframe,
        timestamp_unix: int,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        spread: float | None = None,
    ) -> Bar:
        """Convert a single CSV row (already parsed) to a Bar contract."""
        raw = RawBar(
            symbol=symbol,
            timeframe=timeframe,
            timestamp_unix=timestamp_unix,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            spread=spread,
        )
        return self._raw_to_bar(raw)

    def normalize_raw(self, raw: RawBar) -> Bar:
        """Convert a RawBar to a Bar contract."""
        return self._raw_to_bar(raw)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _raw_to_bar(self, raw: RawBar) -> Bar:
        """Core conversion logic with full validation."""
        ts = datetime.fromtimestamp(raw.timestamp_unix, tz=timezone.utc)

        open_ = self._to_decimal(raw.open)
        high = self._to_decimal(raw.high)
        low = self._to_decimal(raw.low)
        close = self._to_decimal(raw.close)
        volume = Decimal(str(max(0.0, raw.volume))).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
        # Convert raw points to price units: points / spread_divisor / 10^(price_digits-2)
        spread = (
            self._to_decimal(
                raw.spread / self._spread_divisor / 10 ** (self._price_digits - 2)
            )
            if raw.spread is not None and raw.spread != 0.0
            else None
        )

        # Bar constructor validates OHLC invariants — let it raise
        return Bar(
            symbol=raw.symbol,
            timeframe=raw.timeframe,
            timestamp_utc=ts,
            open=open_,
            high=high,
            low=low,
            close=close,
            volume=volume,
            spread=spread,
            received_at_utc=datetime.now(timezone.utc),
        )

    def _to_decimal(self, value: float) -> Decimal:
        return Decimal(str(value)).quantize(self._quant, rounding=ROUND_HALF_UP)
