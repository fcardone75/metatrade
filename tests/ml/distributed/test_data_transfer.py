"""Tests for DataTransfer — bar CSV+gzip serialization (no real MongoDB needed)."""

from __future__ import annotations

import gzip
import io
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.ml.distributed.data_transfer import _bars_to_csv_gz, _csv_gz_to_bars


BASE_TS = datetime(2024, 3, 1, 10, 0, 0, tzinfo=UTC)


def _make_bar(i: int) -> Bar:
    ts = BASE_TS + timedelta(minutes=i)
    price = Decimal(str(round(1.1 + i * 0.0001, 5)))
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.M1,
        timestamp_utc=ts,
        open=price,
        high=price + Decimal("0.0005"),
        low=price - Decimal("0.0005"),
        close=price,
        volume=Decimal("1000"),
    )


class TestBarsSerialization:
    def test_round_trip_count(self):
        bars = [_make_bar(i) for i in range(100)]
        gz = _bars_to_csv_gz(bars)
        rows = _csv_gz_to_bars(gz)
        assert len(rows) == 100

    def test_output_is_gzip(self):
        bars = [_make_bar(i) for i in range(10)]
        gz = _bars_to_csv_gz(bars)
        buf = io.BytesIO(gz)
        with gzip.GzipFile(fileobj=buf) as f:
            content = f.read().decode()
        assert "datetime" in content
        assert "open" in content

    def test_round_trip_fields(self):
        bars = [_make_bar(0)]
        gz = _bars_to_csv_gz(bars)
        rows = _csv_gz_to_bars(gz)
        row = rows[0]
        assert "datetime" in row
        assert "open" in row
        assert "close" in row

    def test_datetime_format_matches_csv_collector(self):
        """datetime column must use %Y-%m-%d %H:%M:%S — the CsvCollector default."""
        bars = [_make_bar(0)]
        gz = _bars_to_csv_gz(bars)
        rows = _csv_gz_to_bars(gz)
        from datetime import datetime
        # Should parse without error using the expected format
        dt = datetime.strptime(rows[0]["datetime"], "%Y-%m-%d %H:%M:%S")
        assert dt.year == 2024

    def test_compression_reduces_size(self):
        bars = [_make_bar(i) for i in range(1000)]
        gz = _bars_to_csv_gz(bars)
        # raw CSV would be ~1000 * ~60 bytes = ~60KB; gzip should be << that
        assert len(gz) < 60_000

    def test_empty_bars(self):
        gz = _bars_to_csv_gz([])
        rows = _csv_gz_to_bars(gz)
        assert rows == []

    def test_all_rows_have_ohlcv(self):
        bars = [_make_bar(i) for i in range(5)]
        gz = _bars_to_csv_gz(bars)
        rows = _csv_gz_to_bars(gz)
        for row in rows:
            assert "datetime" in row
            assert "open" in row
            assert "high" in row
            assert "low" in row
            assert "close" in row
            assert "volume" in row
