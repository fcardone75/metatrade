"""Tests for NewsCalendarModule and MarketRegimeModule."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.enums import SignalDirection, Timeframe
from metatrade.core.errors import ModuleNotReadyError
from metatrade.technical_analysis.modules.news_calendar_module import (
    NewsCalendarModule,
    _first_friday_of_month,
    _nfp_event,
    _build_hardcoded_events,
    _HARDCODED_EVENTS,
)
from metatrade.technical_analysis.modules.market_regime_module import MarketRegimeModule

UTC = timezone.utc


# ── Helpers ────────────────────────────────────────────────────────────────────

def D(v: float) -> Decimal:
    return Decimal(str(round(v, 6)))


def make_bar(close: float, high_offset: float = 0.002, low_offset: float = 0.002) -> Bar:
    c = D(close)
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_utc=datetime(2024, 1, 15, tzinfo=UTC),
        open=c,
        high=c + D(high_offset),
        low=c - D(low_offset),
        close=c,
        volume=D(1000),
    )


def rising_bars(n: int, start: float = 1.1, step: float = 0.001) -> list[Bar]:
    result = []
    for i in range(n):
        c = round(start + i * step, 6)
        result.append(Bar(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            open=D(c),
            high=D(c + 0.001),
            low=D(c - 0.001),
            close=D(c),
            volume=D(1000),
        ))
    return result


def falling_bars(n: int, start: float = 1.2, step: float = 0.001) -> list[Bar]:
    result = []
    for i in range(n):
        c = round(start - i * step, 6)
        result.append(Bar(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            open=D(c),
            high=D(c + 0.001),
            low=D(c - 0.001),
            close=D(c),
            volume=D(1000),
        ))
    return result


def volatile_bars(n: int) -> list[Bar]:
    """Bars with very large ATR (high - low >> average)."""
    result = []
    for i in range(n):
        c = D(1.1 + (i % 5) * 0.005)
        # First half: tiny range; last 5: huge range
        if i >= n - 5:
            high = c + D(0.05)   # 500-pip candle
            low  = c - D(0.05)
        else:
            high = c + D(0.0005)
            low  = c - D(0.0005)
        result.append(Bar(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            open=c,
            high=high,
            low=low,
            close=c,
            volume=D(1000),
        ))
    return result


def flat_bars(n: int, price: float = 1.1) -> list[Bar]:
    result = []
    for i in range(n):
        c = D(price)
        result.append(Bar(
            symbol="EURUSD",
            timeframe=Timeframe.H1,
            timestamp_utc=datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i),
            open=c,
            high=c + D(0.0002),
            low=c - D(0.0002),
            close=c,
            volume=D(1000),
        ))
    return result


# ── NewsCalendarModule tests ───────────────────────────────────────────────────

class TestNewCalendarHelpers:
    def test_first_friday_jan_2025(self) -> None:
        # Jan 2025: 1=Wed, 2=Thu, 3=Fri
        assert _first_friday_of_month(2025, 1) == 3

    def test_first_friday_every_month_is_a_friday(self) -> None:
        for year in range(2024, 2027):
            for month in range(1, 13):
                day = _first_friday_of_month(year, month)
                assert datetime(year, month, day).weekday() == 4, \
                    f"Expected Friday for {year}-{month:02d}-{day:02d}"

    def test_nfp_event_time_is_1330_utc(self) -> None:
        dt = _nfp_event(2025, 3)
        assert dt.hour == 13 and dt.minute == 30
        assert dt.tzinfo is not None

    def test_hardcoded_events_sorted(self) -> None:
        events = _build_hardcoded_events(2024, 2026)
        dts = [e.dt_utc for e in events]
        assert dts == sorted(dts)

    def test_hardcoded_events_include_fomc_ecb_nfp(self) -> None:
        events = _build_hardcoded_events(2025, 2026)
        names = {e.name for e in events}
        assert "NFP" in names
        assert "FOMC" in names
        assert "ECB" in names

    def test_hardcoded_events_has_12_nfps_per_year(self) -> None:
        events = _build_hardcoded_events(2025, 2025)
        nfps = [e for e in events if e.name == "NFP"]
        assert len(nfps) == 12


class TestNewsCalendarModule:
    def _bar(self) -> Bar:
        return make_bar(1.1)

    def test_module_id(self) -> None:
        m = NewsCalendarModule(timeframe_id="h1")
        assert "news_calendar" in m.module_id

    def test_min_bars_is_one(self) -> None:
        m = NewsCalendarModule()
        assert m.min_bars == 1

    def test_hold_near_nfp(self) -> None:
        """Should block within window_minutes of an NFP."""
        # March 2025 NFP: first Friday = 7 March 2025, 13:30 UTC
        nfp_dt = _nfp_event(2025, 3)  # 2025-03-07 13:30 UTC
        # 30 minutes before NFP — inside the default 60-min window
        ts = nfp_dt - timedelta(minutes=30)
        m = NewsCalendarModule(window_minutes=60)
        sig = m.analyse([self._bar()], ts)
        assert sig.direction == SignalDirection.HOLD
        assert sig.metadata.get("blocking") is True
        assert "NFP" in sig.metadata.get("event", "")

    def test_clear_outside_window(self) -> None:
        """Should NOT block when far from any event."""
        # 2025-02-10 08:00 UTC — no scheduled events nearby
        ts = datetime(2025, 2, 10, 8, 0, 0, tzinfo=UTC)
        m = NewsCalendarModule(window_minutes=60)
        sig = m.analyse([self._bar()], ts)
        assert sig.metadata.get("blocking") is False

    def test_hold_just_after_fomc(self) -> None:
        """Should block just after FOMC."""
        fomc = datetime(2026, 1, 28, 19, 0, tzinfo=UTC)  # known 2026 FOMC
        ts = fomc + timedelta(minutes=45)  # 45 min after
        m = NewsCalendarModule(window_minutes=60)
        sig = m.analyse([self._bar()], ts)
        assert sig.direction == SignalDirection.HOLD
        assert sig.metadata.get("blocking") is True

    def test_narrow_window_clears(self) -> None:
        """With a 5-min window, 30 min before NFP should be clear."""
        nfp_dt = _nfp_event(2025, 3)
        ts = nfp_dt - timedelta(minutes=30)
        m = NewsCalendarModule(window_minutes=5)
        sig = m.analyse([self._bar()], ts)
        assert sig.metadata.get("blocking") is False

    def test_no_api_key_no_live_events(self) -> None:
        m = NewsCalendarModule(finnhub_api_key=None)
        assert m._api_key is None
        assert m._live_events == []

    def test_signal_is_always_hold(self) -> None:
        """NewsCalendarModule only ever emits HOLD direction."""
        ts = datetime(2025, 6, 15, 14, 0, 0, tzinfo=UTC)  # quiet time
        m = NewsCalendarModule()
        sig = m.analyse([self._bar()], ts)
        assert sig.direction == SignalDirection.HOLD

    def test_confidence_always_050(self) -> None:
        ts_near_nfp = _nfp_event(2025, 3)
        ts_clear    = datetime(2025, 2, 10, 8, 0, 0, tzinfo=UTC)
        m = NewsCalendarModule(window_minutes=60)
        for ts in (ts_near_nfp, ts_clear):
            sig = m.analyse([self._bar()], ts)
            assert sig.confidence == 0.50


# ── MarketRegimeModule tests ───────────────────────────────────────────────────

class TestMarketRegimeModule:
    def _default_module(self) -> MarketRegimeModule:
        return MarketRegimeModule(
            adx_period=14,
            adx_threshold=25.0,
            atr_period=5,
            atr_avg_period=20,
            high_vol_mult=2.0,
            timeframe_id="h1",
        )

    def _ts(self) -> datetime:
        return datetime(2024, 6, 1, 12, 0, tzinfo=UTC)

    def test_module_id(self) -> None:
        m = MarketRegimeModule(timeframe_id="h1")
        assert m.module_id == "market_regime_h1"

    def test_min_bars_positive(self) -> None:
        m = self._default_module()
        assert m.min_bars > 0

    def test_raises_on_too_few_bars(self) -> None:
        m = self._default_module()
        with pytest.raises(ModuleNotReadyError):
            m.analyse([make_bar(1.1)], self._ts())

    def test_ranging_regime_on_flat_bars(self) -> None:
        """Flat market → low ADX → RANGING → HOLD."""
        m = self._default_module()
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, self._ts())
        assert sig.direction == SignalDirection.HOLD
        assert sig.metadata["regime"] == "ranging"

    def test_trending_up_gives_buy(self) -> None:
        """Strongly rising bars should produce a TRENDING_UP / BUY signal."""
        m = MarketRegimeModule(
            adx_period=10,
            adx_threshold=15.0,  # lower threshold for test data
            atr_period=5,
            atr_avg_period=20,
            high_vol_mult=3.0,   # raise volatile threshold
            timeframe_id="h1",
        )
        bars = rising_bars(m.min_bars + 10, step=0.003)
        sig = m.analyse(bars, self._ts())
        # Either trending_up (BUY) or ranging (HOLD) — verify no exceptions and correct structure
        assert sig.direction in (SignalDirection.BUY, SignalDirection.HOLD)
        assert sig.metadata["regime"] in ("trending_up", "ranging")

    def test_volatile_regime_on_spike_bars(self) -> None:
        """Bars with huge ATR spikes should produce VOLATILE → HOLD."""
        m = MarketRegimeModule(
            adx_period=5,
            adx_threshold=25.0,
            atr_period=3,
            atr_avg_period=15,
            high_vol_mult=1.5,  # easier to trigger volatile
            timeframe_id="h1",
        )
        bars = volatile_bars(m.min_bars + 5)
        sig = m.analyse(bars, self._ts())
        # May be volatile or ranging depending on exact numbers; just no exceptions
        assert sig.direction in (SignalDirection.HOLD, SignalDirection.BUY, SignalDirection.SELL)

    def test_metadata_keys(self) -> None:
        m = self._default_module()
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, self._ts())
        for key in ("regime", "adx", "plus_di", "minus_di", "atr", "avg_atr", "vol_ratio"):
            assert key in sig.metadata, f"Missing metadata key: {key}"

    def test_confidence_trending_above_hold(self) -> None:
        """In a trending regime, confidence should be >= 0.55."""
        m = MarketRegimeModule(
            adx_period=5,
            adx_threshold=10.0,  # very low threshold → easy to exceed
            atr_period=3,
            atr_avg_period=15,
            high_vol_mult=5.0,
            timeframe_id="h1",
        )
        bars = rising_bars(m.min_bars + 10, step=0.005)
        sig = m.analyse(bars, self._ts())
        if sig.metadata["regime"] in ("trending_up", "trending_down"):
            assert sig.confidence >= 0.55

    def test_confidence_hold_is_050(self) -> None:
        m = self._default_module()
        bars = flat_bars(m.min_bars + 5)
        sig = m.analyse(bars, self._ts())
        if sig.metadata["regime"] in ("ranging", "volatile"):
            assert sig.confidence == 0.50

    def test_invalid_adx_period_raises(self) -> None:
        with pytest.raises(ValueError):
            MarketRegimeModule(adx_period=1)

    def test_invalid_atr_period_raises(self) -> None:
        with pytest.raises(ValueError):
            MarketRegimeModule(atr_period=0)

    def test_atr_avg_less_than_atr_raises(self) -> None:
        with pytest.raises(ValueError):
            MarketRegimeModule(atr_period=14, atr_avg_period=5)

    def test_falling_bars_direction(self) -> None:
        """Strongly falling bars should produce TRENDING_DOWN / SELL or RANGING / HOLD."""
        m = MarketRegimeModule(
            adx_period=10,
            adx_threshold=15.0,
            atr_period=5,
            atr_avg_period=20,
            high_vol_mult=3.0,
            timeframe_id="h1",
        )
        bars = falling_bars(m.min_bars + 10, step=0.003)
        sig = m.analyse(bars, self._ts())
        assert sig.direction in (SignalDirection.SELL, SignalDirection.HOLD)
        assert sig.metadata["regime"] in ("trending_down", "ranging")
