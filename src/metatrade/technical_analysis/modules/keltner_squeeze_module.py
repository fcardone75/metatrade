"""Keltner Channel Squeeze module.

Combines Bollinger Bands with the Keltner Channel to detect:

1. **Squeeze** — when Bollinger Bands contract *inside* the Keltner Channel
   (BB_upper < KC_upper AND BB_lower > KC_lower).
   The market is coiling; volatility is compressed.  Signal: HOLD.

2. **Bullish breakout** — squeeze has just released (or never was) and
   the close is *above* the Keltner upper band.  Signal: BUY.

3. **Bearish breakout** — squeeze has just released and the close is
   *below* the Keltner lower band.  Signal: SELL.

4. **No signal** — no squeeze, price within KC bands.  Signal: HOLD.

Why this replaces StochasticRSI
    Stochastic RSI and Adaptive RSI both measure momentum via the RSI
    oscillator — they are informationally redundant.  The Keltner Squeeze
    measures *volatility compression* followed by *directional expansion*,
    a completely different category of information that has no overlap with
    the existing Bollinger Bands module (which looks at band position, not
    the Bollinger/Keltner relationship).

Confidence scaling
    Breakout:   base 0.60 + depth × 0.25  (depth = % close beyond KC band)
    Squeeze:    0.50  (neutral — prevents entry but doesn't dominate)
"""

from __future__ import annotations

from datetime import datetime

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.indicators.bollinger import bollinger_bands
from metatrade.technical_analysis.indicators.keltner import is_squeeze, keltner_channel
from metatrade.technical_analysis.interface import ITechnicalModule

_VERSION = ModuleVersion(1, 0, 0)


class KeltnerSqueezeModule(ITechnicalModule):
    """Bollinger/Keltner Squeeze volatility detector.

    Args:
        bb_period:    Bollinger Bands look-back period (default 20).
        bb_std:       Bollinger standard-deviation multiplier (default 2.0).
        kc_ema_period: Keltner central EMA period (default 20).
        kc_atr_period: Keltner ATR period (default 10).
        kc_mult:       Keltner ATR multiplier (default 1.5).
                       Lower than 2.0 makes it easier to detect squeezes.
        timeframe_id:  Suffix for ``module_id``.
        confidence_cap: Maximum confidence a directional signal can carry.
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_ema_period: int = 20,
        kc_atr_period: int = 10,
        kc_mult: float = 1.5,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.85,
    ) -> None:
        if bb_period < 2:
            raise ValueError(f"bb_period must be >= 2, got {bb_period}")
        if kc_ema_period < 2:
            raise ValueError(f"kc_ema_period must be >= 2, got {kc_ema_period}")
        if kc_atr_period < 1:
            raise ValueError(f"kc_atr_period must be >= 1, got {kc_atr_period}")
        self._bb_period     = bb_period
        self._bb_std        = bb_std
        self._kc_ema_period = kc_ema_period
        self._kc_atr_period = kc_atr_period
        self._kc_mult       = kc_mult
        self._timeframe_id  = timeframe_id
        self._confidence_cap = confidence_cap

    @property
    def module_id(self) -> str:
        return f"keltner_squeeze_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # KC needs max(ema_period, atr_period+1); BB needs bb_period
        return max(
            self._bb_period,
            self._kc_ema_period,
            self._kc_atr_period + 1,
        ) + 2

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        highs  = [b.high  for b in bars]
        lows   = [b.low   for b in bars]
        closes = [b.close for b in bars]

        # ── Bollinger Bands ───────────────────────────────────────────────────
        bb_upper, bb_middle, bb_lower = bollinger_bands(
            closes, self._bb_period, self._bb_std
        )

        # ── Keltner Channel ───────────────────────────────────────────────────
        kc_upper, kc_middle, kc_lower = keltner_channel(
            highs, lows, closes,
            ema_period=self._kc_ema_period,
            atr_period=self._kc_atr_period,
            multiplier=self._kc_mult,
        )

        curr_close  = closes[-1]
        curr_bb_u   = bb_upper[-1]
        curr_bb_l   = bb_lower[-1]
        curr_kc_u   = kc_upper[-1]
        curr_kc_m   = kc_middle[-1]
        curr_kc_l   = kc_lower[-1]

        # Squeeze state: Bollinger Bands fully inside the Keltner Channel
        squeezing = is_squeeze(curr_bb_u, curr_bb_l, curr_kc_u, curr_kc_l)

        # ── Signal logic ─────────────────────────────────────────────────────
        if squeezing:
            direction  = SignalDirection.HOLD
            confidence = 0.50
            reason = (
                f"KeltnerSqueeze: volatility compressed "
                f"(BB [{float(curr_bb_l):.5f}–{float(curr_bb_u):.5f}] "
                f"inside KC [{float(curr_kc_l):.5f}–{float(curr_kc_u):.5f}])"
            )

        elif curr_close > curr_kc_u:
            # Bullish expansion — price broke above KC upper
            kc_width = float(curr_kc_u - curr_kc_l) if curr_kc_u > curr_kc_l else 1e-8
            depth    = min(1.0, float(curr_close - curr_kc_u) / kc_width)
            confidence = min(self._confidence_cap, 0.60 + depth * 0.25)
            direction  = SignalDirection.BUY
            reason = (
                f"KeltnerSqueeze: bullish breakout "
                f"(close={float(curr_close):.5f} > KC_upper={float(curr_kc_u):.5f}, "
                f"depth={depth:.3f})"
            )

        elif curr_close < curr_kc_l:
            # Bearish expansion — price broke below KC lower
            kc_width = float(curr_kc_u - curr_kc_l) if curr_kc_u > curr_kc_l else 1e-8
            depth    = min(1.0, float(curr_kc_l - curr_close) / kc_width)
            confidence = min(self._confidence_cap, 0.60 + depth * 0.25)
            direction  = SignalDirection.SELL
            reason = (
                f"KeltnerSqueeze: bearish breakout "
                f"(close={float(curr_close):.5f} < KC_lower={float(curr_kc_l):.5f}, "
                f"depth={depth:.3f})"
            )

        else:
            # Price within KC bands — no clear breakout
            direction  = SignalDirection.HOLD
            confidence = 0.50
            reason = (
                f"KeltnerSqueeze: price within KC bands "
                f"(close={float(curr_close):.5f}, "
                f"KC=[{float(curr_kc_l):.5f}–{float(curr_kc_u):.5f}])"
            )

        return AnalysisSignal(
            direction=direction,
            confidence=round(confidence, 4),
            reason=reason,
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
            metadata={
                "squeeze":      squeezing,
                "bb_upper":     float(curr_bb_u),
                "bb_lower":     float(curr_bb_l),
                "kc_upper":     float(curr_kc_u),
                "kc_middle":    float(curr_kc_m),
                "kc_lower":     float(curr_kc_l),
                "close":        float(curr_close),
            },
        )
