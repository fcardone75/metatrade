"""Swing High / Swing Low support-resistance module.

Unlike Pivot Points (which use fixed daily levels), swing levels are dynamic:
they update automatically as the market forms new highs and lows.  These
levels represent genuine turning points where price reversed, making them more
reliable as future support/resistance than mechanically computed pivots.

A bar at index ``i`` is a **swing high** if:
    high[i] ≥ high[i-j]  AND  high[i] ≥ high[i+j]  for all j in 1..lookback

A **swing low** is symmetric with low values.

Because we need ``lookback`` bars to the right to confirm a swing, the most
recent ``lookback`` bars can never contain a confirmed swing point.  The
module therefore looks back beyond that.

Signal logic:
    BUY:  Current close is within ``atr_touch_mult × ATR`` of the nearest
          confirmed swing LOW (price testing support).
    SELL: Current close is within ``atr_touch_mult × ATR`` of the nearest
          confirmed swing HIGH (price testing resistance).
    HOLD: Price is not near any confirmed swing level.

Confidence:
    - Scales with how many times the level has been previously tested.
      A level that held twice is stronger than one tested only once.
    - Capped at ``confidence_cap``.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection
from metatrade.core.versioning import ModuleVersion
from metatrade.technical_analysis.indicators.atr import atr as compute_atr
from metatrade.technical_analysis.interface import ITechnicalModule

_VERSION = ModuleVersion(1, 0, 0)


def _find_swing_highs(
    highs: list[Decimal],
    lookback: int,
) -> list[tuple[int, Decimal]]:
    """Return list of (bar_index, high_price) for confirmed swing highs.

    Only bars from index ``lookback`` to ``len-lookback-1`` can be swings,
    because we need ``lookback`` confirmed bars on both sides.
    """
    n = len(highs)
    swings: list[tuple[int, Decimal]] = []
    for i in range(lookback, n - lookback):
        h = highs[i]
        left_ok = all(h >= highs[i - j] for j in range(1, lookback + 1))
        right_ok = all(h >= highs[i + j] for j in range(1, lookback + 1))
        if left_ok and right_ok:
            swings.append((i, h))
    return swings


def _find_swing_lows(
    lows: list[Decimal],
    lookback: int,
) -> list[tuple[int, Decimal]]:
    """Return list of (bar_index, low_price) for confirmed swing lows."""
    n = len(lows)
    swings: list[tuple[int, Decimal]] = []
    for i in range(lookback, n - lookback):
        lo = lows[i]
        left_ok = all(lo <= lows[i - j] for j in range(1, lookback + 1))
        right_ok = all(lo <= lows[i + j] for j in range(1, lookback + 1))
        if left_ok and right_ok:
            swings.append((i, lo))
    return swings


class SwingLevelModule(ITechnicalModule):
    """Swing High/Low support-resistance signal generator.

    Args:
        lookback:        Bars on each side required to confirm a swing (default 5).
        max_levels:      Number of most-recent swing highs/lows to track (default 5).
        atr_period:      ATR period for adaptive touch distance (default 14).
        atr_touch_mult:  Touch distance = atr_touch_mult × ATR (default 0.5).
        timeframe_id:    Suffix for ``module_id``.
        confidence_cap:  Maximum signal confidence (default 0.82).
    """

    def __init__(
        self,
        lookback: int = 5,
        max_levels: int = 5,
        atr_period: int = 14,
        atr_touch_mult: float = 0.5,
        timeframe_id: str = "h1",
        confidence_cap: float = 0.82,
    ) -> None:
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")
        if max_levels < 1:
            raise ValueError(f"max_levels must be >= 1, got {max_levels}")
        self._lookback = lookback
        self._max_levels = max_levels
        self._atr_period = atr_period
        self._atr_touch_mult = atr_touch_mult
        self._timeframe_id = timeframe_id
        self._confidence_cap = confidence_cap

    @property
    def module_id(self) -> str:
        return f"swing_level_{self._timeframe_id}"

    @property
    def min_bars(self) -> int:
        # Need enough bars to have at least one confirmed swing + ATR warmup
        return max(self._lookback * 4 + 2, self._atr_period + self._lookback * 2 + 5)

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        self._require_min_bars(bars)

        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        closes = [b.close for b in bars]
        curr_close = closes[-1]

        # ── ATR for adaptive touch distance ───────────────────────────────────
        atr_series = compute_atr(highs, lows, closes, self._atr_period)
        curr_atr = atr_series[-1]
        touch_dist = curr_atr * Decimal(str(self._atr_touch_mult))

        # ── Find swing highs and lows ─────────────────────────────────────────
        # Exclude the last `lookback` bars (cannot be confirmed yet)
        safe_bars = len(bars) - self._lookback
        swing_highs = _find_swing_highs(highs[:safe_bars], self._lookback)
        swing_lows = _find_swing_lows(lows[:safe_bars], self._lookback)

        # Keep only the most recent N levels
        swing_highs = swing_highs[-self._max_levels:]
        swing_lows = swing_lows[-self._max_levels:]

        if not swing_highs and not swing_lows:
            return AnalysisSignal(
                direction=SignalDirection.HOLD,
                confidence=0.50,
                reason="SwingLevel: no confirmed swing levels found",
                module_id=self.module_id,
                module_version=_VERSION,
                timestamp_utc=timestamp_utc,
                metadata={"swing_highs": [], "swing_lows": [], "atr": float(curr_atr)},
            )

        # ── Find nearest level of each type ──────────────────────────────────
        nearest_support: tuple[Decimal, int] | None = None    # (price, bar_index)
        nearest_resistance: tuple[Decimal, int] | None = None

        for idx, level_price in swing_lows:
            dist = abs(curr_close - level_price)
            if dist <= touch_dist:
                if nearest_support is None or dist < abs(curr_close - nearest_support[0]):
                    nearest_support = (level_price, idx)

        for idx, level_price in swing_highs:
            dist = abs(curr_close - level_price)
            if dist <= touch_dist:
                if nearest_resistance is None or dist < abs(curr_close - nearest_resistance[0]):
                    nearest_resistance = (level_price, idx)

        # ── Build signal ──────────────────────────────────────────────────────
        sh_prices = [float(p) for _, p in swing_highs]
        sl_prices = [float(p) for _, p in swing_lows]

        if nearest_support is not None and nearest_resistance is None:
            support_price, _ = nearest_support
            dist = float(abs(curr_close - support_price))
            touch_float = float(touch_dist) if float(touch_dist) > 0 else 1e-10
            proximity = max(0.0, 1.0 - dist / touch_float)
            confidence = min(self._confidence_cap, 0.58 + proximity * 0.22)
            return AnalysisSignal(
                direction=SignalDirection.BUY,
                confidence=round(confidence, 4),
                reason=(
                    f"Price {float(curr_close):.5f} near swing low={float(support_price):.5f} "
                    f"(dist={dist:.5f}, touch={float(touch_dist):.5f}): support test"
                ),
                module_id=self.module_id,
                module_version=_VERSION,
                timestamp_utc=timestamp_utc,
                metadata={
                    "nearest_support": float(support_price),
                    "swing_highs": sh_prices,
                    "swing_lows": sl_prices,
                    "atr": float(curr_atr),
                    "touch_dist": float(touch_dist),
                },
            )

        elif nearest_resistance is not None and nearest_support is None:
            resist_price, _ = nearest_resistance
            dist = float(abs(curr_close - resist_price))
            touch_float = float(touch_dist) if float(touch_dist) > 0 else 1e-10
            proximity = max(0.0, 1.0 - dist / touch_float)
            confidence = min(self._confidence_cap, 0.58 + proximity * 0.22)
            return AnalysisSignal(
                direction=SignalDirection.SELL,
                confidence=round(confidence, 4),
                reason=(
                    f"Price {float(curr_close):.5f} near swing high={float(resist_price):.5f} "
                    f"(dist={dist:.5f}, touch={float(touch_dist):.5f}): resistance test"
                ),
                module_id=self.module_id,
                module_version=_VERSION,
                timestamp_utc=timestamp_utc,
                metadata={
                    "nearest_resistance": float(resist_price),
                    "swing_highs": sh_prices,
                    "swing_lows": sl_prices,
                    "atr": float(curr_atr),
                    "touch_dist": float(touch_dist),
                },
            )

        else:
            # Either near both or near neither
            return AnalysisSignal(
                direction=SignalDirection.HOLD,
                confidence=0.50,
                reason=(
                    f"Price {float(curr_close):.5f} not near any confirmed swing level "
                    f"(ATR touch dist={float(touch_dist):.5f})"
                ),
                module_id=self.module_id,
                module_version=_VERSION,
                timestamp_utc=timestamp_utc,
                metadata={
                    "swing_highs": sh_prices,
                    "swing_lows": sl_prices,
                    "atr": float(curr_atr),
                    "touch_dist": float(touch_dist),
                },
            )
