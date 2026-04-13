"""Centralized module assembler.

Single source of truth for building the analysis module stack.
All runner scripts (paper, live, backtest) call ``build_modules()`` so
the stack is always consistent.

Conditional inclusion rules
----------------------------
- Each module is included only if its flag in ``ModuleConfig`` is True.
- ``multi_tf_d1`` requires ~24 × 21 = 504 warmup bars; disable if bar
  history is short.
- ``news_calendar`` works in offline mode even without a Finnhub key.
- ``ml`` is included when ``module_cfg.ml=True`` AND a registry with a
  trained model is provided; the MLModule itself returns HOLD gracefully
  when no model is available, so the registry check is just an optimisation.
"""

from __future__ import annotations

from metatrade.technical_analysis.interface import ITechnicalModule
from metatrade.runner.module_config import ModuleConfig


def build_modules(
    timeframe_id: str,
    module_cfg: ModuleConfig | None = None,
    registry=None,   # ModelRegistry | None — avoid hard import cycle
) -> list[ITechnicalModule]:
    """Assemble the signal pipeline based on ModuleConfig flags.

    Args:
        timeframe_id:  Lower-case timeframe string used as module suffix,
                       e.g. ``"h1"``.
        module_cfg:    ModuleConfig instance.  Defaults to ``ModuleConfig()``
                       (all modules enabled, no API keys).
        registry:      ModelRegistry holding a trained ML model.  Passed to
                       MLModule.  When None, MLModule returns HOLD but is
                       still included for weight tracking.

    Returns:
        Ordered list of ITechnicalModule instances ready for the runner.
    """
    if module_cfg is None:
        module_cfg = ModuleConfig()

    tf = timeframe_id.lower()
    modules: list[ITechnicalModule] = []

    # ── Trend-following ───────────────────────────────────────────────────────
    if module_cfg.ema_crossover:
        from metatrade.technical_analysis.modules.ema_crossover import EmaCrossoverModule
        modules.append(EmaCrossoverModule(fast_period=9, slow_period=21, timeframe_id=tf))

    if module_cfg.multi_tf_h4:
        from metatrade.technical_analysis.modules.multi_timeframe_module import MultiTimeframeModule
        modules.append(MultiTimeframeModule(
            resample_factor=4, fast_period=9, slow_period=21,
            timeframe_id=f"{tf}_to_h4",
        ))

    if module_cfg.multi_tf_d1:
        from metatrade.technical_analysis.modules.multi_timeframe_module import MultiTimeframeModule
        modules.append(MultiTimeframeModule(
            resample_factor=24, fast_period=9, slow_period=21,
            timeframe_id=f"{tf}_to_d1",
        ))

    if module_cfg.adx:
        from metatrade.technical_analysis.modules.adx_module import AdxModule
        modules.append(AdxModule(period=14, adx_threshold=25.0, timeframe_id=tf))

    # ── Breakout ──────────────────────────────────────────────────────────────
    if module_cfg.donchian_breakout:
        from metatrade.technical_analysis.modules.donchian_module import DonchianBreakoutModule
        modules.append(DonchianBreakoutModule(period=20, timeframe_id=tf))

    # ── Regime (placed early so it casts a broad vote before oscillators) ─────
    if module_cfg.market_regime:
        from metatrade.technical_analysis.modules.market_regime_module import MarketRegimeModule
        modules.append(MarketRegimeModule(timeframe_id=tf))

    # ── Volatility / squeeze ─────────────────────────────────────────────────
    if module_cfg.keltner_squeeze:
        from metatrade.technical_analysis.modules.keltner_squeeze_module import KeltnerSqueezeModule
        modules.append(KeltnerSqueezeModule(timeframe_id=tf))

    # ── Mean-reversion / oscillators ─────────────────────────────────────────
    if module_cfg.adaptive_rsi:
        from metatrade.technical_analysis.modules.adaptive_rsi_module import AdaptiveRsiModule
        modules.append(AdaptiveRsiModule(rsi_period=14, adx_period=14, timeframe_id=tf))

    if module_cfg.bollinger_bands:
        from metatrade.technical_analysis.modules.bollinger_module import BollingerBandsModule
        modules.append(BollingerBandsModule(period=20, num_std=2.0, timeframe_id=tf))

    # ── Support / resistance ──────────────────────────────────────────────────
    if module_cfg.pivot_points:
        from metatrade.technical_analysis.modules.pivot_points_module import PivotPointsModule
        modules.append(PivotPointsModule(
            session_bars=24, atr_period=14, atr_touch_mult=0.5, timeframe_id=tf,
        ))

    if module_cfg.swing_levels:
        from metatrade.technical_analysis.modules.swing_level_module import SwingLevelModule
        modules.append(SwingLevelModule(
            lookback=5, max_levels=5, atr_period=14, atr_touch_mult=0.5, timeframe_id=tf,
        ))

    # ── Time / calendar filters ───────────────────────────────────────────────
    if module_cfg.seasonality:
        from metatrade.technical_analysis.modules.seasonality_module import SeasonalityModule
        modules.append(SeasonalityModule(timeframe_id=tf))

    if module_cfg.news_calendar:
        from metatrade.technical_analysis.modules.news_calendar_module import NewsCalendarModule
        modules.append(NewsCalendarModule(
            finnhub_api_key=module_cfg.finnhub_api_key,
            timeframe_id=tf,
        ))

    # ── Machine learning ─────────────────────────────────────────────────────
    if module_cfg.ml:
        from metatrade.ml.module import MLModule
        modules.append(MLModule(
            registry=registry,
            timeframe_id=tf,
        ))

    return modules
