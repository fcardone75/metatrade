"""Shared pytest fixtures for the entire test suite."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from metatrade.core.clock import BacktestClock, FixedClock
from metatrade.core.contracts.market import Bar, Instrument
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection, Timeframe
from metatrade.core.versioning import ModuleVersion

# ── Timestamps ────────────────────────────────────────────────────────────────

UTC = UTC
T0 = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
T1 = datetime(2024, 1, 15, 11, 0, 0, tzinfo=UTC)


@pytest.fixture
def t0() -> datetime:
    return T0


@pytest.fixture
def t1() -> datetime:
    return T1


@pytest.fixture
def fixed_clock() -> FixedClock:
    return FixedClock(T0)


@pytest.fixture
def backtest_clock() -> BacktestClock:
    return BacktestClock(T0)


# ── Instruments ───────────────────────────────────────────────────────────────

@pytest.fixture
def eurusd() -> Instrument:
    return Instrument(
        symbol="EURUSD",
        base_currency="EUR",
        quote_currency="USD",
        pip_size=Decimal("0.0001"),
        lot_size=Decimal("100000"),
        min_lot=Decimal("0.01"),
        max_lot=Decimal("500.0"),
        lot_step=Decimal("0.01"),
        digits=5,
    )


# ── Bars ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def bullish_bar() -> Bar:
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_utc=T0,
        open=Decimal("1.10000"),
        high=Decimal("1.10500"),
        low=Decimal("1.09800"),
        close=Decimal("1.10400"),
        volume=Decimal("1500"),
    )


@pytest.fixture
def bearish_bar() -> Bar:
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_utc=T0,
        open=Decimal("1.10400"),
        high=Decimal("1.10500"),
        low=Decimal("1.09800"),
        close=Decimal("1.10000"),
        volume=Decimal("1500"),
    )


# ── Signals ───────────────────────────────────────────────────────────────────

@pytest.fixture
def buy_signal() -> AnalysisSignal:
    return AnalysisSignal(
        direction=SignalDirection.BUY,
        confidence=0.75,
        reason="EMA crossover confirmed",
        module_id="ema_crossover_h1",
        module_version=ModuleVersion(1, 0, 0),
        timestamp_utc=T0,
    )


@pytest.fixture
def sell_signal() -> AnalysisSignal:
    return AnalysisSignal(
        direction=SignalDirection.SELL,
        confidence=0.65,
        reason="RSI overbought",
        module_id="rsi_overbought_h1",
        module_version=ModuleVersion(1, 0, 0),
        timestamp_utc=T0,
    )


@pytest.fixture
def hold_signal() -> AnalysisSignal:
    return AnalysisSignal(
        direction=SignalDirection.HOLD,
        confidence=0.60,
        reason="No clear signal",
        module_id="macd_h1",
        module_version=ModuleVersion(1, 0, 0),
        timestamp_utc=T0,
    )
