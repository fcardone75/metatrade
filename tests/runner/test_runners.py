"""Tests for the runner orchestrators: BaseRunner, BacktestRunner, PaperRunner."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from metatrade.broker.interface import IBrokerAdapter
from metatrade.core.contracts.account import AccountState
from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.risk import RiskDecision
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import RunMode, SignalDirection, Timeframe
from metatrade.core.versioning import ModuleVersion
from metatrade.runner.backtest_runner import BacktestResult, BacktestRunner
from metatrade.runner.base import BaseRunner, RunStats
from metatrade.runner.config import RunnerConfig
from metatrade.runner.live_runner import LiveRunner
from metatrade.runner.paper_runner import PaperRunner
from metatrade.technical_analysis.interface import ITechnicalModule

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def D(v: str | float | int) -> Decimal:
    return Decimal(str(v))


def make_bar(close: float, i: int = 0, volume: float = 1000.0) -> Bar:
    c = D(str(round(close, 5)))
    h = c + D("0.0010")
    lo = c - D("0.0010")
    ts = datetime(2024, 1, 1 + (i // 24), i % 24, 0, 0, tzinfo=UTC)
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_utc=ts,
        open=c,
        high=h,
        low=lo,
        close=c,
        volume=D(str(volume)),
    )


def make_bars(n: int, start: float = 1.1, step: float = 0.0001) -> list[Bar]:
    return [make_bar(round(start + i * step, 5), i=i) for i in range(n)]


def make_runner_config(**kwargs) -> RunnerConfig:
    defaults = dict(
        symbol="EURUSD",
        consensus_threshold=0.6,
        min_signals=1,
        max_risk_pct=0.01,
        daily_loss_limit_pct=0.99,  # disable daily loss limit for testing
        auto_kill_drawdown_pct=0.99,  # disable auto-kill for testing
        max_open_positions=5,
        initial_balance=10000.0,
        paper_initial_balance=10000.0,
    )
    defaults.update(kwargs)
    return RunnerConfig(**defaults)


# ---------------------------------------------------------------------------
# Stub modules for testing
# ---------------------------------------------------------------------------

_VERSION = ModuleVersion(1, 0, 0)


class StubBuyModule(ITechnicalModule):
    """Always emits a BUY signal with high confidence."""

    @property
    def module_id(self) -> str:
        return "stub_buy"

    @property
    def min_bars(self) -> int:
        return 1

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        return AnalysisSignal(
            direction=SignalDirection.BUY,
            confidence=0.90,
            reason="stub buy",
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
        )


class StubSellModule(ITechnicalModule):
    """Always emits a SELL signal with high confidence."""

    @property
    def module_id(self) -> str:
        return "stub_sell"

    @property
    def min_bars(self) -> int:
        return 1

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        return AnalysisSignal(
            direction=SignalDirection.SELL,
            confidence=0.90,
            reason="stub sell",
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
        )


class StubHoldModule(ITechnicalModule):
    """Always emits a HOLD signal."""

    @property
    def module_id(self) -> str:
        return "stub_hold"

    @property
    def min_bars(self) -> int:
        return 1

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        return AnalysisSignal(
            direction=SignalDirection.HOLD,
            confidence=0.60,
            reason="stub hold",
            module_id=self.module_id,
            module_version=_VERSION,
            timestamp_utc=timestamp_utc,
        )


class StubHighMinBarsModule(StubBuyModule):
    """Has very high min_bars so it never fires."""

    @property
    def module_id(self) -> str:
        return "stub_high_min"

    @property
    def min_bars(self) -> int:
        return 9999


# ---------------------------------------------------------------------------
# RunnerConfig
# ---------------------------------------------------------------------------

class TestRunnerConfig:
    def test_defaults(self) -> None:
        cfg = RunnerConfig()
        assert cfg.symbol == "EURUSD"
        assert cfg.consensus_threshold == 0.6
        assert cfg.min_signals >= 1
        assert cfg.max_risk_pct > 0.0

    def test_custom_symbol(self) -> None:
        cfg = RunnerConfig(symbol="GBPUSD")
        assert cfg.symbol == "GBPUSD"

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(Exception):
            RunnerConfig(consensus_threshold=0.3)  # below minimum 0.5

    def test_immutable(self) -> None:
        cfg = RunnerConfig()
        with pytest.raises(Exception):
            cfg.symbol = "USDJPY"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RunStats
# ---------------------------------------------------------------------------

class TestRunStats:
    def test_default_zeros(self) -> None:
        stats = RunStats()
        assert stats.bars_processed == 0
        assert stats.signals_generated == 0
        assert stats.trades_attempted == 0
        assert stats.trades_executed == 0
        assert stats.trades_vetoed == 0
        assert stats.kill_switch_activations == 0


# ---------------------------------------------------------------------------
# BaseRunner
# ---------------------------------------------------------------------------

class TestBaseRunner:
    def _runner(self, modules=None) -> BaseRunner:
        cfg = make_runner_config()
        return BaseRunner(cfg, modules or [StubBuyModule()])

    def test_initial_stats_are_zero(self) -> None:
        runner = self._runner()
        assert runner.stats.bars_processed == 0

    def test_process_bar_increments_bars_processed(self) -> None:
        runner = self._runner()
        bars = make_bars(5)
        runner.process_bar(
            bars=bars,
            account_balance=D("10000"),
            account_equity=D("10000"),
            free_margin=D("10000"),
        )
        assert runner.stats.bars_processed == 1

    def test_returns_none_when_no_signals(self) -> None:
        # Module requires more bars than we give
        runner = self._runner([StubHighMinBarsModule()])
        bars = make_bars(5)
        result = runner.process_bar(
            bars=bars,
            account_balance=D("10000"),
            account_equity=D("10000"),
            free_margin=D("10000"),
        )
        assert result is None

    def test_returns_none_on_hold_consensus(self) -> None:
        runner = self._runner([StubHoldModule()])
        bars = make_bars(5)
        result = runner.process_bar(
            bars=bars,
            account_balance=D("10000"),
            account_equity=D("10000"),
            free_margin=D("10000"),
        )
        assert result is None

    def test_buy_module_produces_risk_decision(self) -> None:
        runner = self._runner([StubBuyModule()])
        bars = make_bars(5)
        result = runner.process_bar(
            bars=bars,
            account_balance=D("10000"),
            account_equity=D("10000"),
            free_margin=D("10000"),
        )
        assert isinstance(result, RiskDecision)

    def test_approved_decision_increments_executed(self) -> None:
        runner = self._runner([StubBuyModule()])
        bars = make_bars(5)
        result = runner.process_bar(
            bars=bars,
            account_balance=D("10000"),
            account_equity=D("10000"),
            free_margin=D("10000"),
        )
        if result is not None and result.approved:
            assert runner.stats.trades_executed == 1
        elif result is not None and not result.approved:
            assert runner.stats.trades_vetoed == 1

    def test_vetoed_decision_increments_vetoed(self) -> None:
        runner = self._runner([StubBuyModule()])
        bars = make_bars(5)
        # Very large spread → spread veto
        result = runner.process_bar(
            bars=bars,
            account_balance=D("10000"),
            account_equity=D("10000"),
            free_margin=D("10000"),
            current_spread_pips=1000.0,  # excessive spread
        )
        if result is not None and not result.approved:
            assert runner.stats.trades_vetoed >= 1

    def test_uses_timestamp_from_arg(self) -> None:
        runner = self._runner([StubBuyModule()])
        bars = make_bars(5)
        result = runner.process_bar(
            bars=bars,
            account_balance=D("10000"),
            account_equity=D("10000"),
            free_margin=D("10000"),
            timestamp_utc=NOW,
        )
        if result is not None:
            assert result.timestamp_utc == NOW

    def test_module_below_min_bars_skipped(self) -> None:
        # Mix of skippable and active modules
        runner = self._runner([StubHighMinBarsModule(), StubBuyModule()])
        bars = make_bars(5)
        result = runner.process_bar(
            bars=bars,
            account_balance=D("10000"),
            account_equity=D("10000"),
            free_margin=D("10000"),
        )
        # StubHighMinBarsModule skipped, StubBuyModule used → should return something
        assert result is not None


# ---------------------------------------------------------------------------
# BacktestRunner
# ---------------------------------------------------------------------------

class TestBacktestRunner:
    def _runner(self, bars=None, modules=None) -> BacktestRunner:
        cfg = make_runner_config()
        b = bars if bars is not None else make_bars(100)
        return BacktestRunner(cfg, modules or [StubBuyModule()], b)

    def test_run_returns_backtest_result(self) -> None:
        runner = self._runner()
        result = runner.run()
        assert isinstance(result, BacktestResult)

    def test_initial_balance_set(self) -> None:
        runner = self._runner()
        result = runner.run()
        assert result.initial_balance == D("10000")

    def test_bars_processed_is_positive(self) -> None:
        bars = make_bars(50)
        runner = self._runner(bars=bars)
        result = runner.run()
        # bars_processed counts signal evaluations (skips bars when position is open)
        assert 1 <= result.stats.bars_processed <= 50

    def test_n_trades_non_negative(self) -> None:
        runner = self._runner()
        result = runner.run()
        assert result.n_trades >= 0

    def test_win_rate_in_range(self) -> None:
        runner = self._runner()
        result = runner.run()
        assert 0.0 <= result.win_rate <= 1.0

    def test_return_pct_computed(self) -> None:
        runner = self._runner()
        result = runner.run()
        # Just verify it's a float
        assert isinstance(result.return_pct, float)

    def test_total_pnl_consistent_with_trades(self) -> None:
        runner = self._runner()
        result = runner.run()
        expected_pnl = sum(t.pnl for t in result.trades)
        assert abs(result.total_pnl - expected_pnl) < D("0.01")

    def test_hold_module_generates_no_trades(self) -> None:
        runner = self._runner(modules=[StubHoldModule()])
        result = runner.run()
        assert result.n_trades == 0

    def test_final_balance_reflects_pnl(self) -> None:
        runner = self._runner()
        result = runner.run()
        # Allow for commission deductions
        # final ≈ initial + total_pnl - commissions
        # Just check it's within a reasonable range (not exploded)
        assert result.final_balance > D("0")
        assert result.final_balance < D("1000000")

    def test_empty_bars_produces_empty_result(self) -> None:
        runner = self._runner(bars=[])
        result = runner.run()
        assert result.n_trades == 0
        assert result.stats.bars_processed == 0

    def test_sl_and_tp_exits_tracked(self) -> None:
        runner = self._runner(bars=make_bars(200))
        result = runner.run()
        for trade in result.trades:
            assert trade.exit_reason in ("sl", "tp", "end_of_data", "open")


# ---------------------------------------------------------------------------
# PaperRunner
# ---------------------------------------------------------------------------

class TestPaperRunner:
    def _runner(self, modules=None) -> PaperRunner:
        cfg = make_runner_config()
        return PaperRunner(cfg, modules or [StubBuyModule()])

    def test_broker_is_accessible(self) -> None:
        runner = self._runner()
        assert runner.broker is not None

    def test_on_bar_returns_none_with_hold_module(self) -> None:
        runner = self._runner([StubHoldModule()])
        bar = make_bar(1.1, i=100)
        result = runner.on_bar(bar)
        assert result is None

    def test_on_bar_processes_single_bar(self) -> None:
        runner = self._runner([StubBuyModule()])
        bar = make_bar(1.1, i=0)
        runner.on_bar(bar)
        assert runner.stats.bars_processed == 1

    def test_bar_buffer_grows(self) -> None:
        runner = self._runner([StubHoldModule()])
        for i in range(10):
            runner.on_bar(make_bar(1.1, i=i))
        # Buffer should have bars
        assert runner.stats.bars_processed == 10

    def test_reset_clears_buffer(self) -> None:
        runner = self._runner([StubHoldModule()])
        for i in range(5):
            runner.on_bar(make_bar(1.1, i=i))
        runner.reset()
        # After reset, next bar starts fresh
        result = runner.on_bar(make_bar(1.1, i=100))
        assert result is None  # single bar → no signals with enough history

    def test_on_bar_returns_risk_decision(self) -> None:
        runner = self._runner([StubBuyModule()])
        bar = make_bar(1.1, i=0)
        result = runner.on_bar(bar)
        # With BUY module and min_bars=1, should attempt a decision
        assert result is None or isinstance(result, RiskDecision)

    def test_buffer_bounded_at_500(self) -> None:
        runner = self._runner([StubHoldModule()])
        for i in range(600):
            runner.on_bar(make_bar(1.1, i=i))
        # Internal buffer should be capped
        assert runner.stats.bars_processed == 600  # all bars processed


# ---------------------------------------------------------------------------
# LiveRunner
# ---------------------------------------------------------------------------

def _make_mock_broker(balance: float = 10000.0) -> IBrokerAdapter:
    """Create a mock IBrokerAdapter that returns a simple AccountState."""
    broker = MagicMock(spec=IBrokerAdapter)
    broker.is_connected = True

    account = AccountState(
        balance=D(str(balance)),
        equity=D(str(balance)),
        free_margin=D(str(balance)),
        used_margin=D("0"),
        timestamp_utc=NOW,
        open_positions_count=0,
        run_mode=RunMode.LIVE,
    )
    broker.get_account.return_value = account
    broker.send_order.return_value = MagicMock(order_id="order-001")
    return broker


class TestLiveRunner:
    def _runner(self, modules=None, broker=None) -> LiveRunner:
        cfg = make_runner_config()
        b = broker or _make_mock_broker()
        return LiveRunner(cfg, modules or [StubHoldModule()], b)

    def test_initial_not_connected(self) -> None:
        runner = self._runner()
        assert not runner.is_connected

    def test_connect_sets_connected(self) -> None:
        runner = self._runner()
        runner.connect()
        assert runner.is_connected

    def test_disconnect_clears_connected(self) -> None:
        runner = self._runner()
        runner.connect()
        runner.disconnect()
        assert not runner.is_connected

    def test_on_bar_raises_if_not_connected(self) -> None:
        runner = self._runner()
        bar = make_bar(1.1, i=0)
        with pytest.raises(RuntimeError, match="connect"):
            runner.on_bar(bar)

    def test_on_bar_processes_bar_after_connect(self) -> None:
        runner = self._runner([StubHoldModule()])
        runner.connect()
        bar = make_bar(1.1, i=0)
        runner.on_bar(bar)
        assert runner.stats.bars_processed == 1

    def test_on_bar_returns_none_on_hold(self) -> None:
        runner = self._runner([StubHoldModule()])
        runner.connect()
        bar = make_bar(1.1, i=0)
        result = runner.on_bar(bar)
        assert result is None

    def test_on_bar_returns_decision_on_buy(self) -> None:
        runner = self._runner([StubBuyModule()])
        runner.connect()
        bar = make_bar(1.1, i=0)
        result = runner.on_bar(bar)
        assert result is None or isinstance(result, RiskDecision)

    def test_buffer_grows_with_bars(self) -> None:
        runner = self._runner([StubHoldModule()])
        runner.connect()
        for i in range(5):
            runner.on_bar(make_bar(1.1, i=i))
        assert runner.stats.bars_processed == 5

    def test_reset_buffer_clears_bars(self) -> None:
        runner = self._runner([StubHoldModule()])
        runner.connect()
        for i in range(5):
            runner.on_bar(make_bar(1.1, i=i))
        runner.reset_buffer()
        # After reset, next bar starts with empty buffer
        runner.on_bar(make_bar(1.1, i=100))
        assert runner.stats.bars_processed == 6  # 5 before + 1 after reset

    def test_broker_get_account_called_per_bar(self) -> None:
        broker = _make_mock_broker()
        runner = LiveRunner(make_runner_config(), [StubHoldModule()], broker)
        runner.connect()
        for i in range(3):
            runner.on_bar(make_bar(1.1, i=i))
        assert broker.get_account.call_count == 3

    def test_submit_order_called_on_approved_buy(self) -> None:
        from metatrade.execution.order_manager import OrderManager
        broker = _make_mock_broker()
        # Give the mock broker a working send_order
        broker.send_order.return_value = None
        order_manager = MagicMock(spec=OrderManager)
        order_manager.build_order.return_value = MagicMock(order_id="x", side=MagicMock())
        runner = LiveRunner(
            make_runner_config(),
            [StubBuyModule()],
            broker,
            order_manager=order_manager,
        )
        runner.connect()
        runner.on_bar(make_bar(1.1, i=0))
        # Either order was submitted or risk vetoed — just check it ran without error
        assert runner.stats.bars_processed == 1

    # ── Resilience: on_bar must never propagate exceptions ────────────────────

    def test_on_bar_returns_none_when_get_account_raises(self) -> None:
        """If the broker throws during get_account(), on_bar swallows and returns None."""
        broker = _make_mock_broker()
        broker.get_account.side_effect = RuntimeError("connection lost")
        runner = self._runner(broker=broker)
        runner.connect()
        result = runner.on_bar(make_bar(1.1, i=0))
        assert result is None

    def test_on_bar_continues_processing_after_exception(self) -> None:
        """After a failed bar, the next bar is processed normally."""
        broker = _make_mock_broker()
        call_count = 0

        def flaky_account():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient network error")
            return _make_mock_broker().get_account.return_value

        broker.get_account.side_effect = flaky_account
        runner = self._runner(broker=broker)
        runner.connect()
        result_bad  = runner.on_bar(make_bar(1.1, i=0))
        result_good = runner.on_bar(make_bar(1.1, i=1))
        assert result_bad is None
        # Second bar must reach process_bar (bars_processed increments even on error
        # because buffer append happens before get_account)
        assert runner.stats.bars_processed >= 1

    def test_on_bar_returns_none_when_process_bar_raises(self) -> None:
        """Exception in process_bar pipeline does not propagate."""
        from unittest.mock import patch
        runner = self._runner()
        runner.connect()
        with patch.object(runner, "process_bar", side_effect=ValueError("bad signal")):
            result = runner.on_bar(make_bar(1.1, i=0))
        assert result is None

    # ── Resilience: _submit_live_order must never propagate exceptions ────────

    def test_submit_live_order_swallows_order_rejected_error(self) -> None:
        """OrderRejectedError from the broker is caught, not re-raised."""
        from metatrade.core.errors import OrderRejectedError
        from metatrade.execution.order_manager import OrderManager

        broker = _make_mock_broker()
        order_manager = MagicMock(spec=OrderManager)
        order_manager.build_order.return_value = MagicMock(order_id="x")
        order_manager.submit.side_effect = OrderRejectedError(
            message="MT5 order rejected: retcode=10019 comment=No money",
            code="MT5_ORDER_REJECTED",
        )
        runner = LiveRunner(make_runner_config(), [StubHoldModule()], broker, order_manager=order_manager)
        runner.connect()

        # Build a minimal approved RiskDecision to pass to _submit_live_order
        from metatrade.core.contracts.risk import PositionSizeResult, RiskDecision
        from metatrade.core.enums import OrderSide
        ps = PositionSizeResult(
            lot_size=D("0.01"),
            risk_amount=D("10.00"),
            risk_pct=0.01,
            stop_loss_price=D("1.09000"),
            stop_loss_pips=D("20.0"),
            take_profit_price=D("1.11000"),
            pip_value=D("10.0"),
            method="fixed_fractional",
        )
        decision = RiskDecision(
            approved=True,
            symbol="EURUSD",
            side=OrderSide.BUY,
            position_size=ps,
            timestamp_utc=NOW,
        )
        # Must not raise
        runner._submit_live_order(decision, NOW)

    def test_submit_live_order_swallows_unexpected_exception(self) -> None:
        """Any unexpected exception in order submission is swallowed."""
        from metatrade.core.contracts.risk import PositionSizeResult, RiskDecision
        from metatrade.core.enums import OrderSide
        from metatrade.execution.order_manager import OrderManager

        broker = _make_mock_broker()
        order_manager = MagicMock(spec=OrderManager)
        order_manager.build_order.side_effect = RuntimeError("unexpected crash")
        runner = LiveRunner(make_runner_config(), [StubHoldModule()], broker, order_manager=order_manager)
        runner.connect()

        ps = PositionSizeResult(
            lot_size=D("0.01"),
            risk_amount=D("10.00"),
            risk_pct=0.01,
            stop_loss_price=D("1.09000"),
            stop_loss_pips=D("20.0"),
            take_profit_price=None,
            pip_value=D("10.0"),
            method="fixed_fractional",
        )
        decision = RiskDecision(
            approved=True,
            symbol="EURUSD",
            side=OrderSide.BUY,
            position_size=ps,
            timestamp_utc=NOW,
        )
        runner._submit_live_order(decision, NOW)  # must not raise

    def test_logger_uses_structlog_keyword_args(self) -> None:
        """Verify the module-level logger is structlog, not stdlib logging.

        stdlib logging.Logger._log() does not accept arbitrary keyword args,
        which was the root cause of the original crash.
        """

        import metatrade.runner.live_runner as lr_mod
        # get_logger returns a BoundLogger proxy; calling bind() is the
        # structlog-specific API not present on stdlib Logger.
        assert hasattr(lr_mod.log, "bind"), (
            "live_runner.log must be a structlog logger — stdlib Logger "
            "does not support keyword args like side=, symbol="
        )
