"""Market data module configuration."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from metatrade.core.config_base import BaseConfig
from metatrade.core.enums import Timeframe


class MarketDataConfig(BaseConfig):
    """Configuration for the market data module.

    Loaded from environment variables (see .env.example).
    """

    # ── Symbols & timeframes ───────────────────────────────────────────────────
    symbols: list[str] = Field(default=["EURUSD"])
    primary_timeframe: Timeframe = Timeframe.H1
    # Additional timeframes for multi-timeframe analysis
    secondary_timeframes: list[Timeframe] = Field(default=[Timeframe.H4, Timeframe.D1])

    # Number of bars to load into memory on startup (warmup period)
    warmup_bars: int = Field(default=500, ge=50, le=10000)

    # ── Storage paths ─────────────────────────────────────────────────────────
    # DuckDB — historical OHLCV bars
    duckdb_path: str = Field(default="data/bars.duckdb")

    # SQLite — operational data (audit trail, kill switch state)
    sqlite_path: str = Field(default="data/operational.db")

    # ── MT5 connection ─────────────────────────────────────────────────────────
    mt5_login: int = Field(default=0)
    mt5_password: str = Field(default="")
    mt5_server: str = Field(default="")
    mt5_path: str = Field(default="")
    mt5_timeout_ms: int = Field(default=60000, ge=1000, le=300000)

    # ── Data quality ──────────────────────────────────────────────────────────
    # A bar is considered stale if its timestamp is older than this
    max_bar_age_seconds: int = Field(default=7200, ge=60)

    # Maximum allowed gap in a bar series before it's flagged
    # (expressed in number of expected bars, e.g. 5 means 5 missing bars)
    max_gap_bars: int = Field(default=5, ge=1)

    # Decimal precision for price normalization (digits after decimal point)
    price_precision: int = Field(default=5, ge=2, le=8)

    # ── Historical collection ─────────────────────────────────────────────────
    # How many years of history to collect on initial setup
    initial_history_years: int = Field(default=3, ge=1, le=10)

    # Batch size for MT5 history requests (MT5 has limits per request)
    collection_batch_size: int = Field(default=50000, ge=1000, le=1000000)

    model_config = {
        "env_prefix": "MARKET_DATA_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
        "frozen": True,
    }

    @field_validator("symbols")
    @classmethod
    def symbols_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("symbols list cannot be empty")
        for s in v:
            if not s.strip():
                raise ValueError(f"Symbol cannot be blank: {s!r}")
        return [s.upper().strip() for s in v]
