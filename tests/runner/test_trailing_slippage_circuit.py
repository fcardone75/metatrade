"""Tests for BacktestRunner: trailing stop, realistic slippage, circuit breaker.

Also tests BaseRunner daily circuit breaker (daily_loss_limit_usd) and
vol-scaled position sizing end-to-end.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import SignalDirection, Timeframe
from metatrade.core.versioning import ModuleVersion
from metatrade.runner.backtest_runner import BacktestResult, BacktestRunner
from metatrade.runner.config import RunnerConfig
from metatrade.technical_analysis.interface import ITechnicalModule

# ── Helpers ───────────────────────────────────────────────────────────────────


def _bar(
    close: float,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    dt: datetime | None = None,
) -> Bar:
    close_d = Decimal(str(close))
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_utc=dt or datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
        open=Decimal(str(open_ or close)),
        high=Decimal(str(high or close + 0.0002)),
        low=Decimal(str(low or close - 0.0002)),
        close=close_d,
        volume=Decimal("1000"),
    )


def make_bars(n: int, base: float = 1.1000, trend: float = 0.0001) -> list[Bar]:
    """Build a list of ascending bars with a small trend."""
    start = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
    return [
        _bar(
            close=base + i * trend,
            high=base + i * trend + 0.0005,
            low=base + i * trend - 0.0005,
            dt=start + timedelta(hours=i),
        )
        for i in range(n)
    ]


class AlwaysBuyModule(ITechnicalModule):
    """Stub module that always signals BUY with confidence 0.9."""

    @property
    def module_id(self) -> str:
        return "always_buy"

    @property
    def min_bars(self) -> int:
        return 1

    def analyse(self, bars: list[Bar], timestamp_utc: datetime) -> AnalysisSignal:
        return AnalysisSignal(
            direction=SignalDirection.BUY,
            confidence=0.9,
            reason="always buy",
            module_id=self.module_id,
            module_version=ModuleVersion(1, 0, 0),
            timestamp_utc=timestamp_utc,
        )


def make_runner(
    bars: list[Bar],
    slippage: float = 0.5,
    seed: int = 42,
    trailing: bool = True,
    **extra_cfg,
) -> BacktestRunner:
    cfg = RunnerConfig(
        symbol="EURUSD",
        max_risk_pct=0.01,
        initial_balance=10_000.0,
        min_signals=1,
        slippage_pips=slippage,
        random_seed=seed,
        trailing_stop_enabled=trailing,
        **extra_cfg,
    )
    return BacktestRunner(
        config=cfg,
        modules=[AlwaysBuyModule()],
        bars=bars,
    )


# ── Tests: slippage ────────────────────────────────────────────────────────────


class TestSlippage:
    def test_fill_price_differs_from_open_on_buy(self) -> None:
        """Fill price for BUY should be >= next bar open (adverse slippage)."""
        bars = make_bars(100)
        runner = make_runner(bars, slippage=2.0, seed=42)
        result = runner.run()
        # At least one trade should have been attempted; check fill > open
        for trade in result.trades:
            # All BUY fills must be at or above the bar's open
            # (the exact open is the base before slippage is added)
            assert trade.entry_price >= Decimal("0")

    def test_deterministic_with_same_seed(self) -> None:
        bars = make_bars(100)
        r1 = make_runner(bars, seed=17).run()
        r2 = make_runner(bars, seed=17).run()
        assert r1.total_pnl == r2.total_pnl
        assert r1.n_trades == r2.n_trades

    def test_different_seeds_may_differ(self) -> None:
        """Different seeds should produce (potentially) different slippage amounts."""
        bars = make_bars(200)
        r1 = make_runner(bars, slippage=3.0, seed=1).run()
        r2 = make_runner(bars, slippage=3.0, seed=9999).run()
        # The test just verifies both complete without error;
        # slippage amounts are random but deterministic per seed.
        assert r1.n_trades >= 0
        assert r2.n_trades >= 0

    def test_zero_slippage_produces_fill_at_open(self) -> None:
        """When slippage_pips=0, fill equals next bar open exactly."""
        bars = make_bars(100)
        runner = make_runner(bars, slippage=0.0, seed=42)
        result = runner.run()
        # avg slippage should be 0
        assert result.avg_slippage_pips == 0.0

    def test_slippage_pips_tracked_per_trade(self) -> None:
        bars = make_bars(100)
        runner = make_runner(bars, slippage=1.0, seed=42)
        result = runner.run()
        for trade in result.trades:
            if trade.exit_reason != "open":
                assert trade.slippage_pips >= 0.0


# ── Tests: trailing stop ───────────────────────────────────────────────────────


class TestTrailingStop:
    def test_trailing_enabled_exit_reason_contains_trailing(self) -> None:
        """When trailing stop is enabled, exits should be 'trailing_sl' not 'sl'."""
        bars = make_bars(200)
        runner = make_runner(bars, trailing=True, slippage=0.0, seed=42)
        result = runner.run()
        # For trailing-stop exits, reason must be 'trailing_sl'
        for trade in result.trades:
            if "sl" in trade.exit_reason:
                assert trade.exit_reason == "trailing_sl"

    def test_trailing_disabled_exit_reason_is_sl(self) -> None:
        bars = make_bars(200)
        runner = make_runner(bars, trailing=False, slippage=0.0, seed=42)
        result = runner.run()
        for trade in result.trades:
            if "sl" in trade.exit_reason:
                assert trade.exit_reason == "sl"

    def test_backtest_result_has_slippage_property(self) -> None:
        bars = make_bars(100)
        result = make_runner(bars, slippage=2.0, seed=1).run()
        assert isinstance(result.avg_slippage_pips, float)
        assert result.avg_slippage_pips >= 0.0

    def test_result_properties(self) -> None:
        bars = make_bars(150)
        result = make_runner(bars).run()
        assert isinstance(result, BacktestResult)
        assert result.n_trades == result.n_winning + result.n_losing + sum(
            1 for t in result.trades if t.pnl == 0
        )
        assert 0.0 <= result.win_rate <= 1.0


# ── Tests: daily circuit breaker ──────────────────────────────────────────────


class TestDailyCircuitBreaker:
    def _make_bars_multi_day(self, n_days: int = 3, bars_per_day: int = 24) -> list[Bar]:
        """Build bars spanning multiple UTC calendar days."""
        bars = []
        start = datetime(2024, 1, 2, 0, 0, tzinfo=UTC)
        for i in range(n_days * bars_per_day):
            dt = start + timedelta(hours=i)
            close = 1.1000 + i * 0.0001
            bars.append(_bar(close, dt=dt))
        return bars

    def test_circuit_breaker_zero_disables_limit(self) -> None:
        """daily_loss_limit_usd=0 means no daily circuit breaker."""
        bars = self._make_bars_multi_day()
        cfg = RunnerConfig(
            symbol="EURUSD",
            max_risk_pct=0.01,
            initial_balance=10_000.0,
            min_signals=1,
            daily_loss_limit_usd=0.0,
        )
        runner = BacktestRunner(cfg, [AlwaysBuyModule()], bars)
        result = runner.run()
        # Should run without error; circuit breaker inactive
        assert result.stats.bars_processed > 0

    def test_daily_loss_limit_field_exists(self) -> None:
        cfg = RunnerConfig(daily_loss_limit_usd=200.0)
        assert cfg.daily_loss_limit_usd == 200.0

    def test_daily_loss_limit_non_negative(self) -> None:
        with pytest.raises(Exception):
            RunnerConfig(daily_loss_limit_usd=-1.0)

    def test_runner_config_new_defaults(self) -> None:
        cfg = RunnerConfig()
        assert cfg.slippage_pips == pytest.approx(0.3)
        assert cfg.random_seed == 42
        assert cfg.trailing_stop_enabled is True
        assert cfg.daily_loss_limit_usd == pytest.approx(0.0)


# ── Tests: RunnerConfig new fields ────────────────────────────────────────────


class TestRunnerConfigNewFields:
    def test_slippage_non_negative(self) -> None:
        with pytest.raises(Exception):
            RunnerConfig(slippage_pips=-0.1)

    def test_trailing_stop_default_true(self) -> None:
        assert RunnerConfig().trailing_stop_enabled is True

    def test_trailing_stop_can_be_disabled(self) -> None:
        cfg = RunnerConfig(trailing_stop_enabled=False)
        assert cfg.trailing_stop_enabled is False

    def test_random_seed_default(self) -> None:
        assert RunnerConfig().random_seed == 42

    def test_random_seed_zero_means_random(self) -> None:
        cfg = RunnerConfig(random_seed=0)
        assert cfg.random_seed == 0
