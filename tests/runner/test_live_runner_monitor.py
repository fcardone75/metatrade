"""Tests for LiveRunner.monitor_positions_once() and _evaluate_and_act().

All tests run without MT5; the broker adapter is replaced with a MagicMock
so we can control what positions are returned and assert which broker calls
are made.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from metatrade.broker.interface import IBrokerAdapter
from metatrade.core.contracts.market import Bar
from metatrade.core.enums import Timeframe
from metatrade.exit_engine.contracts import ExitAction, ExitDecision, ExitSignal
from metatrade.runner.config import RunnerConfig
from metatrade.runner.live_runner import LiveRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _cfg() -> RunnerConfig:
    return RunnerConfig(
        symbol="EURUSD",
        min_signals=1,
        max_risk_pct=0.01,
        daily_loss_limit_pct=0.99,
        auto_kill_drawdown_pct=0.99,
        max_open_positions=5,
    )


def _bar(close: float = 1.10000, ts: datetime | None = None, i: int = 0) -> Bar:
    c = Decimal(str(round(close, 5)))
    ts = ts or datetime(2024, 1, 1, i % 24, 0, 0, tzinfo=timezone.utc)
    return Bar(
        symbol="EURUSD", timeframe=Timeframe.M5,
        timestamp_utc=ts,
        open=c, high=c + Decimal("0.0010"),
        low=c  - Decimal("0.0010"), close=c,
        volume=Decimal("1000"),
    )


def _make_runner(broker: IBrokerAdapter) -> LiveRunner:
    runner = LiveRunner(config=_cfg(), modules=[], broker=broker)
    runner._connected = True
    # pre-fill a bar buffer so PositionContext has bars_since_entry
    entry_ts = NOW - timedelta(minutes=10)
    runner._bar_buffer = [_bar(1.10000, ts=entry_ts + timedelta(minutes=i))
                          for i in range(10)]
    return runner


def _mock_broker(
    *,
    positions: list[dict] | None = None,
    bid: float = 1.10000,
    ask: float = 1.10010,
) -> MagicMock:
    b = MagicMock(spec=IBrokerAdapter)
    b.get_open_positions.return_value = positions or []
    b.get_current_price.return_value = (Decimal(str(bid)), Decimal(str(ask)))
    b.modify_sl_tp.return_value = True
    b.close_position_by_ticket.return_value = True
    return b


def _position_dict(
    ticket: int = 101,
    is_buy: bool = True,
    price_open: float = 1.09800,
    sl: float = 1.09600,
    tp: float = 1.10200,
    volume: float = 0.10,
) -> dict:
    return {
        "ticket":     ticket,
        "symbol":     "EURUSD",
        "volume":     volume,
        "type":       0 if is_buy else 1,
        "price_open": price_open,
        "sl":         sl,
        "tp":         tp,
        "profit":     5.0,
        "magic":      20240601,
        "comment":    "metatrade",
        "time_utc":   NOW - timedelta(minutes=10),
    }


def _exit_decision(
    action: ExitAction = ExitAction.HOLD,
    new_sl: Decimal | None = None,
    close_pct: float = 0.0,
    score: float = 10.0,
) -> ExitDecision:
    return ExitDecision(
        action=action,
        close_pct=close_pct if action != ExitAction.HOLD else 0.0,
        aggregate_score=score,
        signals=(),
        explanation=f"{action.value} score={score}",
        timestamp_utc=NOW,
        new_stop_loss=new_sl,
    )


# ---------------------------------------------------------------------------
# monitor_positions_once
# ---------------------------------------------------------------------------

class TestMonitorPositionsOnce:

    def test_returns_empty_when_not_connected(self) -> None:
        broker = _mock_broker()
        runner = LiveRunner(config=_cfg(), modules=[], broker=broker)
        # not connected
        engine = MagicMock()
        assert runner.monitor_positions_once(engine) == []

    def test_returns_empty_when_no_positions(self) -> None:
        broker = _mock_broker(positions=[])
        runner = _make_runner(broker)
        engine = MagicMock()
        assert runner.monitor_positions_once(engine) == []

    def test_calls_broker_get_open_positions(self) -> None:
        broker = _mock_broker(positions=[])
        runner = _make_runner(broker)
        engine = MagicMock()
        runner.monitor_positions_once(engine, symbol="EURUSD")
        broker.get_open_positions.assert_called_once_with(symbol="EURUSD")

    def test_returns_empty_on_broker_error(self) -> None:
        broker = _mock_broker()
        broker.get_open_positions.side_effect = RuntimeError("MT5 disconnected")
        runner = _make_runner(broker)
        engine = MagicMock()
        assert runner.monitor_positions_once(engine) == []

    def test_processes_multiple_positions(self) -> None:
        pos1 = _position_dict(ticket=101)
        pos2 = _position_dict(ticket=102)
        broker = _mock_broker(positions=[pos1, pos2])
        runner = _make_runner(broker)

        engine = MagicMock()
        engine.evaluate.return_value = _exit_decision(ExitAction.HOLD)

        result = runner.monitor_positions_once(engine)
        # Both evaluated; HOLD with no SL update → empty actions
        assert result == []
        assert engine.evaluate.call_count == 2


# ---------------------------------------------------------------------------
# _evaluate_and_act
# ---------------------------------------------------------------------------

class TestEvaluateAndAct:

    def _run(
        self,
        *,
        action: ExitAction = ExitAction.HOLD,
        new_sl: Decimal | None = None,
        close_pct: float = 1.0,
        score: float = 70.0,
        pos: dict | None = None,
    ) -> tuple[dict | None, MagicMock, MagicMock]:
        """Helper: run _evaluate_and_act and return (action_dict, broker, engine)."""
        if pos is None:
            pos = _position_dict()
        broker = _mock_broker()
        runner = _make_runner(broker)
        engine = MagicMock()
        engine.evaluate.return_value = _exit_decision(
            action, new_sl=new_sl, close_pct=close_pct, score=score
        )
        act = runner._evaluate_and_act(pos, engine, NOW)
        return act, broker, engine

    # ── HOLD with no SL update ────────────────────────────────────────────────

    def test_hold_without_new_sl_returns_none(self) -> None:
        act, broker, _ = self._run(action=ExitAction.HOLD, new_sl=None)
        assert act is None
        broker.modify_sl_tp.assert_not_called()
        broker.close_position_by_ticket.assert_not_called()

    # ── HOLD with SL update (trailing stop moved) ─────────────────────────────

    def test_hold_with_new_sl_calls_modify(self) -> None:
        new_sl = Decimal("1.09700")
        act, broker, _ = self._run(action=ExitAction.HOLD, new_sl=new_sl, score=20.0)
        assert act is not None
        assert act["action"] == "SL_UPDATE"
        assert act["new_sl"] == pytest.approx(float(new_sl))
        broker.modify_sl_tp.assert_called_once()
        args = broker.modify_sl_tp.call_args[1]  # called with keyword args
        assert args["new_sl"] == pytest.approx(float(new_sl))

    def test_sl_update_returns_none_when_modify_fails(self) -> None:
        broker = _mock_broker()
        broker.modify_sl_tp.return_value = False
        runner = _make_runner(broker)
        engine = MagicMock()
        engine.evaluate.return_value = _exit_decision(
            ExitAction.HOLD, new_sl=Decimal("1.09700")
        )
        act = runner._evaluate_and_act(_position_dict(), engine, NOW)
        # modify failed → no action reported
        assert act is None

    # ── CLOSE_FULL ────────────────────────────────────────────────────────────

    def test_close_full_calls_broker(self) -> None:
        act, broker, _ = self._run(action=ExitAction.CLOSE_FULL, close_pct=1.0, score=80.0)
        assert act is not None
        assert act["action"] == "CLOSE_FULL"
        broker.close_position_by_ticket.assert_called_once()

    def test_close_full_passes_correct_volume(self) -> None:
        pos = _position_dict(volume=0.20)
        act, broker, _ = self._run(action=ExitAction.CLOSE_FULL, pos=pos)
        kwargs = broker.close_position_by_ticket.call_args[1]
        assert kwargs["volume"] == pytest.approx(0.20)

    def test_close_full_buy_position_uses_is_buy_true(self) -> None:
        act, broker, _ = self._run(action=ExitAction.CLOSE_FULL,
                                    pos=_position_dict(is_buy=True))
        kwargs = broker.close_position_by_ticket.call_args[1]
        assert kwargs["is_buy"] is True

    def test_close_full_sell_position_uses_is_buy_false(self) -> None:
        act, broker, _ = self._run(action=ExitAction.CLOSE_FULL,
                                    pos=_position_dict(is_buy=False))
        kwargs = broker.close_position_by_ticket.call_args[1]
        assert kwargs["is_buy"] is False

    # ── CLOSE_PARTIAL ─────────────────────────────────────────────────────────

    def test_close_partial_closes_correct_fraction(self) -> None:
        pos = _position_dict(volume=0.40)
        act, broker, _ = self._run(
            action=ExitAction.CLOSE_PARTIAL, close_pct=0.5, pos=pos, score=50.0
        )
        assert act is not None
        assert act["action"] == "CLOSE_PARTIAL"
        kwargs = broker.close_position_by_ticket.call_args[1]
        assert kwargs["volume"] == pytest.approx(0.20)  # 50% of 0.40

    # ── No closed bar available → skip ───────────────────────────────────────

    def test_returns_none_when_no_bars_in_buffer(self) -> None:
        broker = _mock_broker(positions=[_position_dict()])
        runner = _make_runner(broker)
        runner._bar_buffer = []   # empty bar buffer

        engine = MagicMock()
        act = runner._evaluate_and_act(_position_dict(), engine, NOW)
        assert act is None
        engine.evaluate.assert_not_called()

    # ── Price fetch failure → graceful skip ───────────────────────────────────

    def test_returns_none_when_price_unavailable(self) -> None:
        broker = _mock_broker()
        broker.get_current_price.side_effect = RuntimeError("no tick")
        runner = _make_runner(broker)
        engine = MagicMock()
        act = runner._evaluate_and_act(_position_dict(), engine, NOW)
        assert act is None
        engine.evaluate.assert_not_called()
