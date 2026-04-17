"""CSV historical data collector — for backtest and offline testing.

Reads OHLCV data from CSV files and stores them via IBarStore.

Expected CSV format (configurable column names):
    datetime,open,high,low,close,volume
    2024-01-01 00:00:00,1.10000,1.10500,1.09800,1.10200,1500
    ...

The datetime column is parsed as UTC unless a timezone is explicitly included.
Column names are configurable to support different data provider formats
(e.g. Dukascopy, HistData, MetaTrader exported CSV).
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.core.log import get_logger
from metatrade.market_data.errors import CollectionError, MissingDataError
from metatrade.market_data.interface import IBarStore, IHistoricalCollector
from metatrade.market_data.normalizer.bar_normalizer import BarNormalizer

log = get_logger(__name__)


class CsvColumnMap:
    """Maps CSV column names to our internal field names.

    Supports the most common CSV formats from data providers.
    """

    def __init__(
        self,
        datetime_col: str = "datetime",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        spread_col: str | None = None,
        datetime_format: str = "%Y-%m-%d %H:%M:%S",
    ) -> None:
        self.datetime_col = datetime_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.spread_col = spread_col
        self.datetime_format = datetime_format

    @classmethod
    def metatrader(cls) -> CsvColumnMap:
        """Column map for MetaTrader exported CSV format."""
        return cls(
            datetime_col="<DATE> <TIME>",  # combined in pre-processing
            open_col="<OPEN>",
            high_col="<HIGH>",
            low_col="<LOW>",
            close_col="<CLOSE>",
            volume_col="<VOL>",
            datetime_format="%Y.%m.%d %H:%M",
        )

    @classmethod
    def dukascopy(cls) -> CsvColumnMap:
        """Column map for Dukascopy exported CSV format."""
        return cls(
            datetime_col="Gmt time",
            open_col="Open",
            high_col="High",
            low_col="Low",
            close_col="Close",
            volume_col="Volume",
            datetime_format="%d.%m.%Y %H:%M:%S.%f",
        )


class CsvCollector(IHistoricalCollector):
    """Reads historical bars from CSV files.

    Args:
        store:      Where to persist collected bars (can be None for read-only use).
        normalizer: Converts rows to Bar contracts.
        column_map: Describes the CSV column layout.
    """

    def __init__(
        self,
        store: IBarStore | None = None,
        normalizer: BarNormalizer | None = None,
        column_map: CsvColumnMap | None = None,
    ) -> None:
        self._store = store
        self._normalizer = normalizer or BarNormalizer()
        self._column_map = column_map or CsvColumnMap()

    # ── IHistoricalCollector ──────────────────────────────────────────────────

    def is_available(self) -> bool:
        return True  # CSV is always available

    def available_symbols(self) -> list[str]:
        return []  # CSV collector does not know symbols without a file path

    def collect(
        self,
        symbol: str,
        timeframe: Timeframe,
        date_from: datetime,
        date_to: datetime,
    ) -> list[Bar]:
        raise NotImplementedError(
            "CsvCollector requires a file path — use collect_file() instead"
        )

    # ── CSV-specific API ──────────────────────────────────────────────────────

    def collect_file(
        self,
        file_path: str | Path,
        symbol: str,
        timeframe: Timeframe,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        skip_errors: bool = True,
    ) -> list[Bar]:
        """Read bars from a CSV file, optionally filtered by date range.

        Args:
            file_path:   Path to the CSV file.
            symbol:      Symbol name to assign to loaded bars.
            timeframe:   Timeframe to assign to loaded bars.
            date_from:   Optional start filter (UTC-aware).
            date_to:     Optional end filter (UTC-aware).
            skip_errors: If True, log and skip malformed rows.

        Returns:
            List of Bar objects in chronological order.
        """
        path = Path(file_path)
        if not path.exists():
            raise MissingDataError(
                message=f"CSV file not found: {path}",
                code="CSV_FILE_NOT_FOUND",
                context={"path": str(path)},
            )

        bars: list[Bar] = []
        skipped = 0
        cm = self._column_map

        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader, start=2):
                    try:
                        bar = self._parse_row(row, symbol, timeframe, cm)
                        if date_from and bar.timestamp_utc < date_from:
                            continue
                        if date_to and bar.timestamp_utc > date_to:
                            continue
                        bars.append(bar)
                    except Exception as exc:
                        skipped += 1
                        if not skip_errors:
                            raise CollectionError(
                                message=f"CSV parse error at row {row_num}: {exc}",
                                code="CSV_PARSE_ERROR",
                            ) from exc
                        log.warning(
                            "csv_row_skipped",
                            file=str(path),
                            row=row_num,
                            reason=str(exc),
                        )
        except OSError as exc:
            raise CollectionError(
                message=f"Cannot read CSV file {path}: {exc}",
                code="CSV_READ_ERROR",
            ) from exc

        # Sort chronologically — CSV files aren't always sorted
        bars.sort(key=lambda b: b.timestamp_utc)

        if self._store is not None and bars:
            saved = self._store.save_bars(bars)
            log.info(
                "csv_collection_complete",
                file=path.name,
                symbol=symbol,
                timeframe=timeframe.value,
                bars_read=len(bars),
                bars_skipped=skipped,
                bars_saved=saved,
            )

        return bars

    # ── Internal ──────────────────────────────────────────────────────────────

    def _parse_row(
        self,
        row: dict[str, str],
        symbol: str,
        timeframe: Timeframe,
        cm: CsvColumnMap,
    ) -> Bar:
        dt_str = row[cm.datetime_col].strip()
        dt = datetime.strptime(dt_str, cm.datetime_format)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)

        spread_raw = None
        if cm.spread_col and cm.spread_col in row:
            try:
                spread_raw = float(row[cm.spread_col])
            except (ValueError, KeyError):
                pass

        return self._normalizer.normalize_csv_row(
            symbol=symbol,
            timeframe=timeframe,
            timestamp_unix=int(dt.timestamp()),
            open_=float(row[cm.open_col]),
            high=float(row[cm.high_col]),
            low=float(row[cm.low_col]),
            close=float(row[cm.close_col]),
            volume=float(row[cm.volume_col]),
            spread=spread_raw,
        )
