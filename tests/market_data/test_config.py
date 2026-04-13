"""Tests for MarketDataConfig."""

from __future__ import annotations

import pytest

from metatrade.core.enums import Timeframe
from metatrade.market_data.config import MarketDataConfig


class TestMarketDataConfig:
    def test_default_instantiation(self) -> None:
        cfg = MarketDataConfig()
        assert cfg.symbols == ["EURUSD"]
        assert cfg.primary_timeframe == Timeframe.H1
        assert cfg.warmup_bars == 500
        assert cfg.price_precision == 5

    def test_symbols_uppercased(self) -> None:
        cfg = MarketDataConfig(symbols=["eurusd", "gbpusd"])
        assert cfg.symbols == ["EURUSD", "GBPUSD"]

    def test_symbols_stripped(self) -> None:
        cfg = MarketDataConfig(symbols=["  EURUSD  "])
        assert cfg.symbols == ["EURUSD"]

    def test_empty_symbols_raises(self) -> None:
        with pytest.raises(Exception):
            MarketDataConfig(symbols=[])

    def test_blank_symbol_raises(self) -> None:
        with pytest.raises(Exception):
            MarketDataConfig(symbols=[""])

    def test_warmup_bars_bounds(self) -> None:
        # Valid bounds
        cfg_min = MarketDataConfig(warmup_bars=50)
        assert cfg_min.warmup_bars == 50
        cfg_max = MarketDataConfig(warmup_bars=10000)
        assert cfg_max.warmup_bars == 10000

    def test_warmup_bars_below_min_raises(self) -> None:
        with pytest.raises(Exception):
            MarketDataConfig(warmup_bars=49)

    def test_warmup_bars_above_max_raises(self) -> None:
        with pytest.raises(Exception):
            MarketDataConfig(warmup_bars=10001)

    def test_custom_paths(self) -> None:
        cfg = MarketDataConfig(duckdb_path="custom/bars.duckdb", sqlite_path="custom/ops.db")
        assert cfg.duckdb_path == "custom/bars.duckdb"
        assert cfg.sqlite_path == "custom/ops.db"

    def test_secondary_timeframes_default(self) -> None:
        cfg = MarketDataConfig()
        assert Timeframe.H4 in cfg.secondary_timeframes
        assert Timeframe.D1 in cfg.secondary_timeframes

    def test_config_is_immutable(self) -> None:
        cfg = MarketDataConfig()
        with pytest.raises(Exception):
            cfg.warmup_bars = 999  # type: ignore[misc]

    def test_price_precision_bounds(self) -> None:
        cfg_low = MarketDataConfig(price_precision=2)
        assert cfg_low.price_precision == 2
        cfg_high = MarketDataConfig(price_precision=8)
        assert cfg_high.price_precision == 8

    def test_price_precision_out_of_bounds_raises(self) -> None:
        with pytest.raises(Exception):
            MarketDataConfig(price_precision=1)
        with pytest.raises(Exception):
            MarketDataConfig(price_precision=9)
