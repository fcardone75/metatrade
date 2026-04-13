"""Feature extraction for ML models.

Transforms a window of OHLCV bars into a fixed-length numeric feature vector
suitable for scikit-learn classifiers or any ML framework.

Design principles:
- Pure functions: no side effects, deterministic output
- All features normalized to make them scale-invariant
- Each feature is documented with its economic interpretation
- NaN-safe: returns None if the window is too short

Feature groups:
    Returns / momentum : returns_1, returns_2, returns_3, returns_5, returns_10
    EMA alignment      : ema9_dist, ema21_dist, ema9_21_cross
    RSI                : rsi14 (scaled), rsi14_change (momentum of RSI)
    Volatility         : atr14_rel, rolling_std5, rolling_std20, atr_zscore
    Candle anatomy     : body_ratio, upper_wick, lower_wick
    Volume             : vol_rel
    Time / cyclical    : hour_sin, hour_cos, weekday_sin, weekday_cos
    Trend quality      : adx_slope
    Breakout position  : donchian_pos, bb_pct_b
    Divergence         : rsi_divergence
    Regime             : hurst_short
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.technical_analysis.indicators.ema import ema
from metatrade.technical_analysis.indicators.rsi import rsi
from metatrade.technical_analysis.indicators.atr import atr
from metatrade.technical_analysis.indicators.adx import adx as compute_adx
from metatrade.technical_analysis.indicators.bollinger import bollinger_bands
from metatrade.technical_analysis.indicators.donchian import donchian_channel
from metatrade.technical_analysis.indicators.hurst import hurst_exponent


# Minimum bars required to compute all features.
# Raised from 35 → 55 to accommodate:
#   - atr_zscore: needs 30-bar ATR mean/std  → ~44 bars for ATR(14)+30
#   - adx_slope:  needs ADX + 5 look-back    → 14 + 5 + some warm-up
#   - hurst_short: needs 40 bars for R/S
MIN_FEATURE_BARS = 55


@dataclass(frozen=True)
class FeatureVector:
    """Numeric feature vector extracted from a bar window.

    All features are dimensionless floats (returns, ratios, normalised values).

    Fields (21 original + 6 new = 27 total):
        returns_1:       1-bar price return (close[t] / close[t-1] - 1)
        returns_2:       2-bar price return
        returns_3:       3-bar price return
        returns_5:       5-bar cumulative return
        returns_10:      10-bar cumulative return
        ema9_dist:       (close - EMA9) / EMA9  — momentum distance
        ema21_dist:      (close - EMA21) / EMA21
        ema9_21_cross:   (EMA9 - EMA21) / close — crossover magnitude
        rsi14:           RSI(14) scaled to [0, 1]
        rsi14_change:    rsi14[t] - rsi14[t-1]  — RSI momentum
        atr14_rel:       ATR(14) / close — relative volatility
        rolling_std5:    Std-dev of last-5 1-bar returns / close
        rolling_std20:   Std-dev of last-20 1-bar returns / close
        body_ratio:      abs(close - open) / (high - low + ε)
        upper_wick:      (high - max(open, close)) / atr14
        lower_wick:      (min(open, close) - low) / atr14
        vol_rel:         volume / mean(volume[-10:])
        hour_sin:        sin(2π × hour / 24) — cyclical hour encoding
        hour_cos:        cos(2π × hour / 24)
        weekday_sin:     sin(2π × weekday / 5)
        weekday_cos:     cos(2π × weekday / 5)
        adx_slope:       (adx[-1] - adx[-5]) / 100 — trend accelerating/decelerating
        atr_zscore:      (atr_curr - mean(atr[-30:])) / std(atr[-30:]) — vol anomaly
        donchian_pos:    (close - don_low) / (don_up - don_low) — range position
        bb_pct_b:        (close - bb_low) / (bb_up - bb_low) — Bollinger %B
        rsi_divergence:  sign(returns_5) × (rsi14 - 0.5) — RSI confirm/diverge
        hurst_short:     hurst_exponent(closes[-40:], min_period=4) — short regime
    """

    returns_1: float
    returns_2: float
    returns_3: float
    returns_5: float
    returns_10: float
    ema9_dist: float
    ema21_dist: float
    ema9_21_cross: float
    rsi14: float
    rsi14_change: float
    atr14_rel: float
    rolling_std5: float
    rolling_std20: float
    body_ratio: float
    upper_wick: float
    lower_wick: float
    vol_rel: float
    hour_sin: float
    hour_cos: float
    weekday_sin: float
    weekday_cos: float
    adx_slope: float
    atr_zscore: float
    donchian_pos: float
    bb_pct_b: float
    rsi_divergence: float
    hurst_short: float

    def to_list(self) -> list[float]:
        """Ordered list of all feature values (27 elements)."""
        return [
            self.returns_1,
            self.returns_2,
            self.returns_3,
            self.returns_5,
            self.returns_10,
            self.ema9_dist,
            self.ema21_dist,
            self.ema9_21_cross,
            self.rsi14,
            self.rsi14_change,
            self.atr14_rel,
            self.rolling_std5,
            self.rolling_std20,
            self.body_ratio,
            self.upper_wick,
            self.lower_wick,
            self.vol_rel,
            self.hour_sin,
            self.hour_cos,
            self.weekday_sin,
            self.weekday_cos,
            self.adx_slope,
            self.atr_zscore,
            self.donchian_pos,
            self.bb_pct_b,
            self.rsi_divergence,
            self.hurst_short,
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Stable ordered feature names matching to_list() (27 names)."""
        return [
            "returns_1",
            "returns_2",
            "returns_3",
            "returns_5",
            "returns_10",
            "ema9_dist",
            "ema21_dist",
            "ema9_21_cross",
            "rsi14",
            "rsi14_change",
            "atr14_rel",
            "rolling_std5",
            "rolling_std20",
            "body_ratio",
            "upper_wick",
            "lower_wick",
            "vol_rel",
            "hour_sin",
            "hour_cos",
            "weekday_sin",
            "weekday_cos",
            "adx_slope",
            "atr_zscore",
            "donchian_pos",
            "bb_pct_b",
            "rsi_divergence",
            "hurst_short",
        ]


def _d(v: Decimal) -> float:
    return float(v)


def _rolling_std(returns: list[float], window: int) -> float:
    """Sample std-dev of the last ``window`` values in ``returns``."""
    subset = returns[-window:]
    if len(subset) < 2:
        return 0.0
    mean = sum(subset) / len(subset)
    variance = sum((x - mean) ** 2 for x in subset) / (len(subset) - 1)
    return math.sqrt(variance)


def extract_features(
    bars: list[Bar],
    timestamp_utc: datetime | None = None,
) -> FeatureVector | None:
    """Extract a feature vector from a window of bars.

    Args:
        bars:           Historical bars, oldest first, newest last.
                        Must contain at least MIN_FEATURE_BARS entries.
        timestamp_utc:  Optional timestamp for the current bar.  When
                        provided, time-of-day and day-of-week cyclical
                        features are computed from it.  When None, both
                        hour and weekday features are set to 0.0.

    Returns:
        FeatureVector (27 features) if enough data is available, None otherwise.
    """
    if len(bars) < MIN_FEATURE_BARS:
        return None

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [_d(b.volume) for b in bars]

    close_f = [_d(c) for c in closes]
    c = close_f[-1]  # current close

    eps = 1e-10  # avoid division by zero

    # ── Returns ───────────────────────────────────────────────────────────────
    returns_1  = (c / (close_f[-2]  + eps)) - 1.0
    returns_2  = (c / (close_f[-3]  + eps)) - 1.0
    returns_3  = (c / (close_f[-4]  + eps)) - 1.0
    returns_5  = (c / (close_f[-6]  + eps)) - 1.0
    returns_10 = (c / (close_f[-11] + eps)) - 1.0

    # Series of 1-bar returns for rolling std computation
    bar_returns = [
        (close_f[i] / (close_f[i - 1] + eps)) - 1.0
        for i in range(1, len(close_f))
    ]

    # ── EMA distances ─────────────────────────────────────────────────────────
    ema9_vals  = ema(closes, 9)
    ema21_vals = ema(closes, 21)
    e9  = _d(ema9_vals[-1])
    e21 = _d(ema21_vals[-1])

    ema9_dist     = (c - e9)   / (e9  + eps)
    ema21_dist    = (c - e21)  / (e21 + eps)
    ema9_21_cross = (e9 - e21) / (c   + eps)

    # ── RSI ───────────────────────────────────────────────────────────────────
    rsi14_vals = rsi(closes, 14)
    rsi14        = float(rsi14_vals[-1]) / 100.0          # [0, 1]
    rsi14_prev   = float(rsi14_vals[-2]) / 100.0
    rsi14_change = rsi14 - rsi14_prev                     # RSI momentum

    # ── ATR-based features ────────────────────────────────────────────────────
    atr14_vals = atr(highs, lows, closes, 14)
    atr14_f    = [_d(v) for v in atr14_vals]
    atr14     = atr14_f[-1]
    atr14_rel = atr14 / (c + eps)

    # ── Rolling volatility ────────────────────────────────────────────────────
    rolling_std5  = _rolling_std(bar_returns, 5)  / (c + eps)
    rolling_std20 = _rolling_std(bar_returns, 20) / (c + eps)

    # ── Candle anatomy ────────────────────────────────────────────────────────
    bar = bars[-1]
    o = _d(bar.open)
    h = _d(bar.high)
    lo = _d(bar.low)
    candle_range = h - lo + eps
    body_ratio  = abs(c - o) / candle_range
    upper_wick  = (h - max(o, c)) / (atr14 + eps)
    lower_wick  = (min(o, c) - lo) / (atr14 + eps)

    # ── Volume ────────────────────────────────────────────────────────────────
    recent_vols = volumes[-10:]
    mean_vol    = sum(recent_vols) / len(recent_vols)
    vol_rel     = volumes[-1] / (mean_vol + eps)

    # ── Cyclical time features ────────────────────────────────────────────────
    if timestamp_utc is not None:
        hour    = timestamp_utc.hour
        weekday = timestamp_utc.weekday()  # 0=Mon … 6=Sun
        hour_sin    = math.sin(2 * math.pi * hour    / 24)
        hour_cos    = math.cos(2 * math.pi * hour    / 24)
        weekday_sin = math.sin(2 * math.pi * weekday / 5)
        weekday_cos = math.cos(2 * math.pi * weekday / 5)
    else:
        hour_sin = hour_cos = weekday_sin = weekday_cos = 0.0

    # ── NEW: ADX slope ────────────────────────────────────────────────────────
    # (adx[-1] - adx[-5]) / 100 — positive = trend strengthening
    try:
        adx_vals = compute_adx(highs, lows, closes, period=14)
        adx_slope = (float(adx_vals[-1]) - float(adx_vals[-5])) / 100.0
    except Exception:
        adx_slope = 0.0

    # ── NEW: ATR z-score ──────────────────────────────────────────────────────
    # (current ATR - mean of last 30 ATR values) / std of last 30 ATR values
    # Positive = currently more volatile than recent norm; negative = calmer.
    try:
        atr_window = atr14_f[-30:]
        if len(atr_window) >= 2:
            atr_mean = sum(atr_window) / len(atr_window)
            atr_std  = math.sqrt(
                sum((v - atr_mean) ** 2 for v in atr_window) / len(atr_window)
            )
            atr_zscore = (atr14 - atr_mean) / (atr_std + eps)
        else:
            atr_zscore = 0.0
    except Exception:
        atr_zscore = 0.0

    # ── NEW: Donchian channel position ────────────────────────────────────────
    # (close - 20-period low) / (20-period high - 20-period low)
    # 0.0 = at channel bottom, 1.0 = at channel top.
    try:
        don_up, _, don_low = donchian_channel(highs, lows, period=20)
        don_range = float(don_up[-1] - don_low[-1])
        if don_range > 0:
            donchian_pos = float(closes[-1] - don_low[-1]) / don_range
        else:
            donchian_pos = 0.5
    except Exception:
        donchian_pos = 0.5

    # ── NEW: Bollinger %B ─────────────────────────────────────────────────────
    # (close - bb_lower) / (bb_upper - bb_lower) — 0 = at lower band, 1 = at upper
    try:
        bb_up, _, bb_low_band = bollinger_bands(closes, period=20, num_std=2.0)
        bb_range = float(bb_up[-1] - bb_low_band[-1])
        if bb_range > 0:
            bb_pct_b = float(closes[-1] - bb_low_band[-1]) / bb_range
        else:
            bb_pct_b = 0.5
    except Exception:
        bb_pct_b = 0.5

    # ── NEW: RSI divergence ───────────────────────────────────────────────────
    # sign(returns_5) × (rsi14 - 0.5)
    # Positive = RSI confirms 5-bar price move; negative = divergence.
    sign_r5 = 1.0 if returns_5 >= 0 else -1.0
    rsi_divergence = sign_r5 * (rsi14 - 0.5)

    # ── NEW: Short-term Hurst exponent ────────────────────────────────────────
    # hurst_exponent over last 40 bars with min_period=4
    # > 0.55 = trending short-term; < 0.45 = mean-reverting short-term
    try:
        hurst_short = hurst_exponent(closes[-40:], min_period=4)
    except Exception:
        hurst_short = 0.5

    return FeatureVector(
        returns_1=returns_1,
        returns_2=returns_2,
        returns_3=returns_3,
        returns_5=returns_5,
        returns_10=returns_10,
        ema9_dist=ema9_dist,
        ema21_dist=ema21_dist,
        ema9_21_cross=ema9_21_cross,
        rsi14=rsi14,
        rsi14_change=rsi14_change,
        atr14_rel=atr14_rel,
        rolling_std5=rolling_std5,
        rolling_std20=rolling_std20,
        body_ratio=body_ratio,
        upper_wick=upper_wick,
        lower_wick=lower_wick,
        vol_rel=vol_rel,
        hour_sin=hour_sin,
        hour_cos=hour_cos,
        weekday_sin=weekday_sin,
        weekday_cos=weekday_cos,
        adx_slope=adx_slope,
        atr_zscore=atr_zscore,
        donchian_pos=donchian_pos,
        bb_pct_b=bb_pct_b,
        rsi_divergence=rsi_divergence,
        hurst_short=hurst_short,
    )
