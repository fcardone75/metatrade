"""Module enable/disable configuration.

A single place to turn each analysis module on or off and to provide
API credentials that some modules need.  If an API key is absent, the
relevant module falls back to a hardcoded offline mode (where supported)
or is silently skipped.

Usage example::

    cfg = ModuleConfig(
        multi_tf_d1=False,   # skip D1 alignment (not enough warmup bars)
        news_calendar=True,
        finnhub_api_key="my_key",   # enables real-time calendar
        ml=False,                    # run without ML module
    )
    modules = build_modules("h1", cfg, registry=None)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModuleConfig:
    """Boolean switches for every module in the analysis stack.

    All modules default to *enabled*.  Set a flag to ``False`` to exclude
    that module from the signal pipeline.

    API keys
    --------
    ``finnhub_api_key``
        Required for live economic-calendar lookups in ``NewsCalendarModule``.
        When absent, the module uses a hardcoded schedule of recurring events
        (NFP, FOMC, ECB) and is still useful — just not dynamically updated.

    Module groups
    -------------
    Trend-following
        ema_crossover, multi_tf_h4, multi_tf_d1, adx

    Breakout
        donchian_breakout  ← N-period price-range breakout (Turtle Trading)

    Volatility / squeeze
        keltner_squeeze    ← Bollinger/Keltner squeeze (replaces StochRSI)

    Mean-reversion / oscillators
        adaptive_rsi, bollinger_bands

    Support / resistance
        pivot_points, swing_levels

    Regime filter
        market_regime   ← centralized TRENDING / RANGING / VOLATILE classifier

    Time / calendar filters
        seasonality, news_calendar

    Machine learning
        ml
    """

    # ── Trend-following ───────────────────────────────────────────────────────
    ema_crossover: bool = False
    """EMA(9)/HMA(21) crossover — DISABLED: redundant with MACD."""

    macd: bool = True
    """MACD(12,26,9) trend/momentum signal — preferred over EMA Crossover."""

    multi_tf_h4: bool = True
    """Multi-timeframe H1 → H4 trend alignment."""

    multi_tf_d1: bool = True
    """Multi-timeframe H1 → D1 macro trend alignment (needs ~500 bars warmup)."""

    adx: bool = False
    """ADX trend-strength filter — DISABLED: redundant with Market Regime module."""

    # ── Breakout ──────────────────────────────────────────────────────────────
    donchian_breakout: bool = True
    """Donchian Channel breakout — period adapts to timeframe (50 on M1/M5, 20 on M15+)."""

    # ── Volatility / squeeze ─────────────────────────────────────────────────
    keltner_squeeze: bool = False
    """Bollinger/Keltner squeeze — DISABLED: redundant with Bollinger Bands module."""

    # ── Mean-reversion / oscillators ─────────────────────────────────────────
    adaptive_rsi: bool = True
    """RSI with ADX-adaptive oversold/overbought thresholds."""

    bollinger_bands: bool = True
    """Bollinger Band bounce at upper/lower bands."""

    # ── Support / resistance ──────────────────────────────────────────────────
    pivot_points: bool = True
    """Classic daily pivot points (ATR-adaptive touch distance)."""

    swing_levels: bool = False
    """Dynamic S/R from swing highs/lows — DISABLED: redundant with Pivot Points + Donchian."""

    # ── Regime ────────────────────────────────────────────────────────────────
    market_regime: bool = True
    """Centralized regime classifier (TRENDING/RANGING/VOLATILE) + Hurst Exponent.
    Provides a meta-vote that reinforces or dampens other modules."""

    # ── Time / calendar filters ───────────────────────────────────────────────
    seasonality: bool = False
    """Session quality filter — DISABLED: replaced by session_filter_utc_start/end in RunnerConfig."""

    news_calendar: bool = False
    """Economic event filter — DISABLED: hardcoded schedule is unreliable.
    Use the spread filter (max_spread_pips) as a dynamic volatility gate instead."""

    # ── Machine learning ─────────────────────────────────────────────────────
    ml: bool = True
    """HistGradientBoosting classifier trained via walk-forward.
    Automatically disabled when no trained model is available in the
    registry — no need to set this flag False for that case."""

    # ── API keys ──────────────────────────────────────────────────────────────
    finnhub_api_key: str | None = None
    """Finnhub.io API key for the economic calendar endpoint.
    Free tier is sufficient (500 req/day).
    Get one at https://finnhub.io/register.
    When None, NewsCalendarModule uses its hardcoded event schedule."""

    def summary(self) -> dict[str, bool]:
        """Return a dict of {module_name: enabled} for logging."""
        return {
            k: v for k, v in self.__dict__.items()
            if isinstance(v, bool)
        }
