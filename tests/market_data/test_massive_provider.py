"""Tests for Massive.com forex data provider (no live HTTP)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
import pytest

from metatrade.market_data.providers.massive.aggregates import (
    max_chunk_calendar_days,
)
from metatrade.market_data.providers.massive.http_client import append_query
from metatrade.market_data.providers.massive.postprocess import fill_micro_gaps
from metatrade.market_data.providers.massive.service import (
    collect_timeframes_for_training,
    ensure_massive_forex_csv,
)
from metatrade.market_data.providers.massive.settings import MassiveSettings
from metatrade.market_data.providers.massive.ticker import normalize_forex_ticker, strip_massive_prefix
from metatrade.market_data.providers.massive.timeframes import timeframe_minutes, timeframe_to_aggs_params
from metatrade.market_data.providers.massive.window import max_adaptive_bars, massive_date_window


def test_normalize_forex_ticker() -> None:
    assert normalize_forex_ticker("eurusd") == "C:EURUSD"
    assert normalize_forex_ticker("C:GBPUSD") == "C:GBPUSD"


def test_strip_massive_prefix() -> None:
    assert strip_massive_prefix("C:EURUSD") == "EURUSD"


def test_timeframe_to_aggs_params() -> None:
    assert timeframe_to_aggs_params("M1") == (1, "minute")
    assert timeframe_to_aggs_params("H1") == (60, "minute")
    assert timeframe_to_aggs_params("D1") == (1, "day")


def test_timeframe_minutes() -> None:
    assert timeframe_minutes("M5") == 5
    assert timeframe_minutes("H4") == 240


def test_max_chunk_calendar_days() -> None:
    s = MassiveSettings(api_key="x", max_aggs_limit=50_000, max_history_days=730)
    d = max_chunk_calendar_days("M1", s)
    assert 20 <= d <= 35


def test_massive_date_window_from_bars() -> None:
    d_to = date(2025, 6, 1)
    # patch now indirectly via explicit overrides in function — use overrides
    d_from, d2 = massive_date_window(
        timeframe="H1",
        bars=24 * 10,
        date_from_override=date(2025, 5, 20),
        date_to_override=d_to,
        max_history_days=730,
    )
    assert d_from == date(2025, 5, 20)
    assert d2 == d_to


def test_massive_date_window_clamp_history() -> None:
    d_to = date(2025, 6, 1)
    d_from, out_to = massive_date_window(
        timeframe="M1",
        bars=10_000_000,
        date_from_override=date(2020, 1, 1),
        date_to_override=d_to,
        max_history_days=30,
    )
    assert out_to == d_to
    assert (d_to - d_from).days <= 30


def test_fill_micro_gaps_inserts_flat_bars() -> None:
    step = 60_000
    base = 1_700_000_000_000
    rows = [
        (base, 1.0, 1.0, 1.0, 1.0, 10.0),
        (base + 3 * step, 1.1, 1.1, 1.1, 1.1, 5.0),
    ]
    filled = fill_micro_gaps(rows, "M1", gap_fill_max_bars=5)
    assert len(filled) == 4
    assert filled[1][4] == 1.0 and filled[1][5] == 0.0
    assert filled[2][4] == 1.0


def test_fill_micro_gaps_skips_macro_gap() -> None:
    step = 60_000
    base = 1_700_000_000_000
    rows = [
        (base, 1.0, 1.0, 1.0, 1.0, 1.0),
        (base + 10 * step, 1.2, 1.2, 1.2, 1.2, 1.0),
    ]
    filled = fill_micro_gaps(rows, "M1", gap_fill_max_bars=5)
    assert len(filled) == 2


def test_append_query_adds_key() -> None:
    u = append_query("https://api.massive.com/v3/reference/tickers", {"apiKey": "secret"})
    assert "apiKey=secret" in u


def test_max_adaptive_bars() -> None:
    assert max_adaptive_bars(10_000) == 20_000


def test_collect_timeframes_for_training_multires() -> None:
    got = collect_timeframes_for_training(["M1"], multires=True)
    assert got == ["M1", "M5", "M15"]


def test_collect_timeframes_for_training_no_dup() -> None:
    got = collect_timeframes_for_training(["M5", "M15", "M30"], multires=True)
    assert got == ["M5", "M15", "M30"]


def test_settings_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setattr(
        "metatrade.market_data.providers.massive.settings.load_dotenv",
        lambda *a, **k: None,
    )
    with pytest.raises(OSError, match="MASSIVE_API_KEY"):
        MassiveSettings.from_env()


def test_ensure_massive_cache_hit_no_http(tmp_path: Path) -> None:
    settings = MassiveSettings(api_key="test-key")
    out = tmp_path / "EURUSD_M1_20250101_20250110.csv"
    out.write_text("datetime,open,high,low,close,volume\n", encoding="utf-8")
    # path must match cache_path convention
    got = ensure_massive_forex_csv(
        symbol="EURUSD",
        timeframe="M1",
        bars=100,
        cache_dir=tmp_path,
        refresh=False,
        date_from=date(2025, 1, 1),
        date_to=date(2025, 1, 10),
        settings=settings,
    )
    assert got == out


def test_ensure_massive_fetch_writes_csv(tmp_path: Path) -> None:
    settings = MassiveSettings(api_key="k", min_interval_sec=0.0, max_aggs_limit=50_000)

    class FakeClient:
        def get_json(self, path: str, extra_params: dict[str, str] | None = None) -> dict:
            ms = int(datetime(2025, 1, 2, 12, 0, tzinfo=UTC).timestamp() * 1000)
            return {
                "status": "OK",
                "results": [{"o": 1.0, "h": 1.1, "l": 0.9, "c": 1.05, "v": 100, "n": 100, "t": ms}],
                "resultsCount": 1,
            }

    path = ensure_massive_forex_csv(
        symbol="EURUSD",
        timeframe="H1",
        bars=50,
        cache_dir=tmp_path,
        refresh=True,
        date_from=date(2025, 1, 1),
        date_to=date(2025, 1, 5),
        settings=settings,
        client=FakeClient(),
    )
    text = path.read_text(encoding="utf-8")
    assert "datetime" in text
    assert "1.05000" in text or "1.05" in text
