"""Tests for core/enums.py."""

from __future__ import annotations

import pytest

from metatrade.core.enums import (
    ConsensusMode,
    KillSwitchLevel,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    PositionState,
    RunMode,
    SignalDirection,
    Timeframe,
)


class TestSignalDirection:
    def test_values(self) -> None:
        assert SignalDirection.BUY.value == "BUY"
        assert SignalDirection.SELL.value == "SELL"
        assert SignalDirection.HOLD.value == "HOLD"

    def test_is_string(self) -> None:
        assert isinstance(SignalDirection.BUY, str)

    def test_from_string(self) -> None:
        assert SignalDirection("BUY") == SignalDirection.BUY


class TestRunMode:
    def test_all_modes_exist(self) -> None:
        assert RunMode.BACKTEST
        assert RunMode.PAPER
        assert RunMode.LIVE

    def test_values(self) -> None:
        assert RunMode.BACKTEST.value == "BACKTEST"
        assert RunMode.PAPER.value == "PAPER"
        assert RunMode.LIVE.value == "LIVE"


class TestTimeframe:
    def test_seconds_m1(self) -> None:
        assert Timeframe.M1.seconds == 60

    def test_seconds_h1(self) -> None:
        assert Timeframe.H1.seconds == 3600

    def test_seconds_h4(self) -> None:
        assert Timeframe.H4.seconds == 14400

    def test_seconds_d1(self) -> None:
        assert Timeframe.D1.seconds == 86400

    def test_all_timeframes_have_seconds(self) -> None:
        for tf in Timeframe:
            assert tf.seconds > 0

    def test_timeframes_ordered_by_seconds(self) -> None:
        tfs = [Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.M30,
               Timeframe.H1, Timeframe.H4, Timeframe.D1, Timeframe.W1]
        seconds = [tf.seconds for tf in tfs]
        assert seconds == sorted(seconds)


class TestKillSwitchLevel:
    def test_none_is_lowest(self) -> None:
        assert KillSwitchLevel.NONE == 0

    def test_ordering(self) -> None:
        assert KillSwitchLevel.NONE < KillSwitchLevel.TRADE_GATE
        assert KillSwitchLevel.TRADE_GATE < KillSwitchLevel.SESSION_GATE
        assert KillSwitchLevel.SESSION_GATE < KillSwitchLevel.EMERGENCY_HALT
        assert KillSwitchLevel.EMERGENCY_HALT < KillSwitchLevel.HARD_KILL

    def test_is_int(self) -> None:
        assert isinstance(KillSwitchLevel.NONE, int)

    def test_comparison_with_int(self) -> None:
        assert KillSwitchLevel.NONE == 0
        assert KillSwitchLevel.TRADE_GATE >= 1
        assert KillSwitchLevel.EMERGENCY_HALT >= 1


class TestOrderType:
    def test_values(self) -> None:
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP.value == "STOP"
        assert OrderType.STOP_LIMIT.value == "STOP_LIMIT"


class TestOrderStatus:
    def test_all_statuses(self) -> None:
        statuses = [
            OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]
        assert len(statuses) == 7
