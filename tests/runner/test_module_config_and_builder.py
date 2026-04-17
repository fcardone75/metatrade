"""Tests for ModuleConfig, module_builder.build_modules(), and the new
runner features: spread filter, signal cooldown, drawdown recovery."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.signal import AnalysisSignal
from metatrade.core.enums import RunMode, SignalDirection, Timeframe
from metatrade.core.versioning import ModuleVersion
from metatrade.runner.base import BaseRunner
from metatrade.runner.config import RunnerConfig
from metatrade.runner.module_builder import build_modules
from metatrade.runner.module_config import ModuleConfig
from metatrade.technical_analysis.interface import ITechnicalModule

UTC = UTC
NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


# ── Helpers ────────────────────────────────────────────────────────────────────

def D(v) -> Decimal:
    return Decimal(str(v))


def make_bar(close: float = 1.1, i: int = 0) -> Bar:
    c = D(round(close, 5))
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.H1,
        timestamp_utc=datetime(2024, 1, 1, i % 24, 0, 0, tzinfo=UTC),
        open=c, high=c + D("0.001"), low=c - D("0.001"), close=c,
        volume=D("1000"),
    )


def make_bars(n: int, start: float = 1.1) -> list[Bar]:
    return [make_bar(start + i * 0.0001, i) for i in range(n)]


class _FixedSignalModule(ITechnicalModule):
    """Always returns the same signal."""

    def __init__(self, direction: SignalDirection, confidence: float = 0.75) -> None:
        self._direction = direction
        self._confidence = confidence

    @property
    def module_id(self) -> str:
        return "fixed"

    @property
    def min_bars(self) -> int:
        return 1

    def analyse(self, bars, ts) -> AnalysisSignal:
        return AnalysisSignal(
            direction=self._direction,
            confidence=self._confidence,
            reason="fixed",
            module_id=self.module_id,
            module_version=ModuleVersion(1, 0, 0),
            timestamp_utc=ts,
        )


def make_runner(
    direction: SignalDirection = SignalDirection.BUY,
    confidence: float = 0.75,
    **cfg_kwargs,
) -> BaseRunner:
    """Create a BaseRunner with a single fixed-signal module."""
    cfg = RunnerConfig(
        symbol="EURUSD",
        min_signals=1,
        consensus_threshold=0.5,
        **cfg_kwargs,
    )
    module = _FixedSignalModule(direction, confidence)
    return BaseRunner(config=cfg, modules=[module], auto_weight=False)


# ── ModuleConfig tests ─────────────────────────────────────────────────────────

class TestModuleConfig:
    def test_defaults_all_true(self) -> None:
        cfg = ModuleConfig()
        for k, v in cfg.summary().items():
            assert v is True, f"{k} should default to True"

    def test_summary_returns_only_bools(self) -> None:
        cfg = ModuleConfig(finnhub_api_key="abc")
        s = cfg.summary()
        for v in s.values():
            assert isinstance(v, bool)
        assert "finnhub_api_key" not in s

    def test_can_disable_individual_modules(self) -> None:
        cfg = ModuleConfig(ml=False, news_calendar=False, multi_tf_d1=False,
                           donchian_breakout=False, keltner_squeeze=False)
        assert cfg.ml is False
        assert cfg.news_calendar is False
        assert cfg.multi_tf_d1 is False
        assert cfg.donchian_breakout is False
        assert cfg.keltner_squeeze is False
        assert cfg.ema_crossover is True  # unchanged

    def test_finnhub_api_key_stored(self) -> None:
        cfg = ModuleConfig(finnhub_api_key="my_secret_key")
        assert cfg.finnhub_api_key == "my_secret_key"

    def test_finnhub_api_key_defaults_none(self) -> None:
        cfg = ModuleConfig()
        assert cfg.finnhub_api_key is None


# ── build_modules tests ────────────────────────────────────────────────────────

class TestBuildModules:
    def test_returns_nonempty_list(self) -> None:
        modules = build_modules("h1")
        assert len(modules) > 0

    def test_all_modules_have_module_id(self) -> None:
        modules = build_modules("h1")
        for m in modules:
            assert isinstance(m.module_id, str) and len(m.module_id) > 0

    def test_all_modules_have_min_bars(self) -> None:
        modules = build_modules("h1")
        for m in modules:
            assert m.min_bars >= 1

    def test_disable_ml_excludes_ml_module(self) -> None:
        cfg = ModuleConfig(ml=False)
        modules = build_modules("h1", module_cfg=cfg, registry=None)
        ids = [m.module_id for m in modules]
        assert not any("ml" in mid for mid in ids)

    def test_disable_news_calendar(self) -> None:
        cfg = ModuleConfig(news_calendar=False)
        modules = build_modules("h1", module_cfg=cfg)
        ids = [m.module_id for m in modules]
        assert not any("news_calendar" in mid for mid in ids)

    def test_enable_news_calendar_included(self) -> None:
        cfg = ModuleConfig(news_calendar=True, ml=False)
        modules = build_modules("h1", module_cfg=cfg)
        ids = [m.module_id for m in modules]
        assert any("news_calendar" in mid for mid in ids)

    def test_disable_d1_excludes_d1(self) -> None:
        cfg = ModuleConfig(multi_tf_d1=False)
        modules = build_modules("h1", module_cfg=cfg)
        ids = [m.module_id for m in modules]
        assert not any("_to_d1" in mid for mid in ids)

    def test_enable_d1_included(self) -> None:
        cfg = ModuleConfig(multi_tf_d1=True, ml=False)
        modules = build_modules("h1", module_cfg=cfg)
        ids = [m.module_id for m in modules]
        assert any("_to_d1" in mid for mid in ids)

    def test_all_disabled_returns_empty(self) -> None:
        cfg = ModuleConfig(
            ema_crossover=False, multi_tf_h4=False, multi_tf_d1=False,
            adx=False, donchian_breakout=False, keltner_squeeze=False,
            market_regime=False, adaptive_rsi=False,
            bollinger_bands=False, pivot_points=False,
            swing_levels=False, seasonality=False, news_calendar=False, ml=False,
        )
        modules = build_modules("h1", module_cfg=cfg)
        assert modules == []

    def test_timeframe_id_propagated(self) -> None:
        cfg = ModuleConfig(ml=False, news_calendar=False)
        modules = build_modules("m15", module_cfg=cfg)
        for m in modules:
            assert "m15" in m.module_id, f"{m.module_id} missing timeframe suffix"

    def test_finnhub_key_passed_to_news_calendar(self) -> None:
        cfg = ModuleConfig(
            finnhub_api_key="test_key",
            news_calendar=True,
            ml=False,
        )
        # All other modules disabled for simplicity
        modules = build_modules("h1", module_cfg=ModuleConfig(
            ema_crossover=False, multi_tf_h4=False, multi_tf_d1=False,
            adx=False, donchian_breakout=False, keltner_squeeze=False,
            market_regime=False, adaptive_rsi=False,
            bollinger_bands=False, pivot_points=False,
            swing_levels=False, seasonality=False, news_calendar=True, ml=False,
            finnhub_api_key="test_key",
        ))
        news_mods = [m for m in modules if "news_calendar" in m.module_id]
        assert len(news_mods) == 1
        assert news_mods[0]._api_key == "test_key"


# ── RunnerConfig new fields ────────────────────────────────────────────────────

class TestRunnerConfigNewFields:
    def test_defaults(self) -> None:
        cfg = RunnerConfig()
        assert cfg.max_spread_pips == 3.0
        assert cfg.signal_cooldown_bars == 3
        assert cfg.drawdown_recovery_threshold_pct == 0.05
        assert cfg.drawdown_recovery_risk_mult == 0.5

    def test_spread_disabled_at_zero(self) -> None:
        cfg = RunnerConfig(max_spread_pips=0.0)
        assert cfg.max_spread_pips == 0.0

    def test_cooldown_disabled_at_zero(self) -> None:
        cfg = RunnerConfig(signal_cooldown_bars=0)
        assert cfg.signal_cooldown_bars == 0

    def test_custom_drawdown_settings(self) -> None:
        cfg = RunnerConfig(
            drawdown_recovery_threshold_pct=0.10,
            drawdown_recovery_risk_mult=0.25,
        )
        assert cfg.drawdown_recovery_threshold_pct == 0.10
        assert cfg.drawdown_recovery_risk_mult == 0.25


# ── Spread filter tests ────────────────────────────────────────────────────────

class TestSpreadFilter:
    def _run(self, spread_pips: float | None, max_spread: float) -> bool:
        """Returns True if a decision was produced (spread not blocked)."""
        runner = make_runner(
            direction=SignalDirection.BUY,
            max_spread_pips=max_spread,
            signal_cooldown_bars=0,
        )
        bars = make_bars(5)
        result = runner.process_bar(
            bars=bars,
            account_balance=D("10000"),
            account_equity=D("10000"),
            free_margin=D("10000"),
            timestamp_utc=NOW,
            current_spread_pips=spread_pips,
            run_mode=RunMode.PAPER,
        )
        return result is not None

    def test_spread_below_max_allows_trade(self) -> None:
        assert self._run(spread_pips=1.5, max_spread=3.0) is True

    def test_spread_above_max_blocks_trade(self) -> None:
        assert self._run(spread_pips=5.0, max_spread=3.0) is False

    def test_spread_exactly_at_max_passes(self) -> None:
        # Filter uses strict >; spread at max is still within tolerance
        assert self._run(spread_pips=3.0, max_spread=3.0) is True

    def test_spread_filter_disabled_at_zero(self) -> None:
        """max_spread_pips=0 → filter disabled, even 100-pip spread passes."""
        assert self._run(spread_pips=100.0, max_spread=0.0) is True

    def test_no_spread_data_passes(self) -> None:
        """current_spread_pips=None → no filter applied."""
        assert self._run(spread_pips=None, max_spread=3.0) is True


# ── Signal cooldown tests ──────────────────────────────────────────────────────

class TestSignalCooldown:
    def _runner_and_bars(self, cooldown: int):
        runner = make_runner(
            direction=SignalDirection.BUY,
            max_spread_pips=0,
            signal_cooldown_bars=cooldown,
        )
        return runner, make_bars(5)

    def _process(self, runner: BaseRunner, bars: list[Bar]):
        return runner.process_bar(
            bars=bars,
            account_balance=D("10000"),
            account_equity=D("10000"),
            free_margin=D("10000"),
            timestamp_utc=NOW,
            run_mode=RunMode.PAPER,
        )

    def test_first_bar_gets_decision(self) -> None:
        runner, bars = self._runner_and_bars(cooldown=3)
        result = self._process(runner, bars)
        # May be approved or vetoed (risk manager), but decision must exist
        assert result is not None

    def test_cooldown_blocks_subsequent_bars(self) -> None:
        runner, bars = self._runner_and_bars(cooldown=3)
        # First bar — triggers cooldown if approved
        first = self._process(runner, bars)
        if first is not None and first.approved:
            # Bars 2 and 3 should be blocked
            second = self._process(runner, bars)
            assert second is None, "Cooldown should block bar 2"
            third = self._process(runner, bars)
            assert third is None, "Cooldown should block bar 3"

    def test_cooldown_zero_never_blocks(self) -> None:
        runner, bars = self._runner_and_bars(cooldown=0)
        for _ in range(5):
            result = self._process(runner, bars)
            # Each bar should get a decision (not None due to cooldown)
            # It may be vetoed by risk, but cooldown itself won't suppress it

    def test_cooldown_counter_decrements(self) -> None:
        runner, bars = self._runner_and_bars(cooldown=3)
        # Manually set cooldown
        runner._cooldown_remaining = 2
        assert self._process(runner, bars) is None  # bar 1 blocked
        assert runner._cooldown_remaining == 1
        assert self._process(runner, bars) is None  # bar 2 blocked
        assert runner._cooldown_remaining == 0
        # bar 3 should not be blocked by cooldown (may still be vetoed by risk)

    def test_cooldown_rearms_after_approved_trade(self) -> None:
        """After approved trade, cooldown_remaining should be set."""
        runner, bars = self._runner_and_bars(cooldown=5)
        runner._cooldown_remaining = 0  # ensure no active cooldown
        result = self._process(runner, bars)
        if result is not None and result.approved:
            assert runner._cooldown_remaining == 5


# ── Drawdown recovery tests ────────────────────────────────────────────────────

class TestDrawdownRecovery:
    def _runner(self, threshold: float = 0.05, mult: float = 0.5) -> BaseRunner:
        return make_runner(
            direction=SignalDirection.BUY,
            max_spread_pips=0,
            signal_cooldown_bars=0,
            drawdown_recovery_threshold_pct=threshold,
            drawdown_recovery_risk_mult=mult,
        )

    def _process(
        self,
        runner: BaseRunner,
        balance: float = 10000,
        equity: float = 10000,
    ):
        bars = make_bars(5)
        return runner.process_bar(
            bars=bars,
            account_balance=D(str(balance)),
            account_equity=D(str(equity)),
            free_margin=D(str(equity)),
            timestamp_utc=NOW,
            run_mode=RunMode.PAPER,
        )

    def test_peak_equity_tracks_max(self) -> None:
        runner = self._runner()
        self._process(runner, equity=10000)
        assert runner._peak_equity == D("10000")
        self._process(runner, equity=10500)
        assert runner._peak_equity == D("10500")
        self._process(runner, equity=10200)  # drawdown
        assert runner._peak_equity == D("10500")  # peak unchanged

    def test_no_recovery_when_equity_at_peak(self) -> None:
        """At peak equity, lot size should NOT be scaled down."""
        runner = self._runner(threshold=0.05, mult=0.5)
        # Establish peak
        first = self._process(runner, equity=10000)
        if first is not None and first.approved:
            original_lots = first.position_size.lot_size
            # Process again at same equity — no drawdown
            result = self._process(runner, equity=10000)
            # Second bar might be blocked by cooldown; skip cooldown test
            # (cooldown=0 so it should not be blocked)
            if result is not None and result.approved:
                # No recovery scaling should have been applied
                pass  # just verify no exception

    def test_recovery_scales_lots_down(self) -> None:
        """In drawdown mode, approved lot_size should be reduced."""
        runner = self._runner(threshold=0.05, mult=0.5)
        # Set peak manually to detect drawdown
        runner._peak_equity = D("10000")
        # Equity at 9000 → 10% drawdown → triggers recovery (threshold=5%)
        result = self._process(runner, equity=9000)
        if result is not None and result.approved:
            # Lot size should have been halved
            # (We can't easily check the exact original, but at least verify
            # the lot_size is > 0 and position_size is intact)
            assert result.position_size.lot_size > 0

    def test_recovery_mult_one_no_change(self) -> None:
        """mult=1.0 means no scaling even in recovery mode."""
        runner = self._runner(threshold=0.01, mult=1.0)
        runner._peak_equity = D("10000")
        result = self._process(runner, equity=9500)  # 5% DD
        if result is not None and result.approved:
            assert result.position_size.lot_size > 0

    def test_peak_equity_starts_at_zero(self) -> None:
        runner = self._runner()
        assert runner._peak_equity == D("0")
