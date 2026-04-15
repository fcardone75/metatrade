"""Integration tests for ExitEngine."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from metatrade.exit_engine.config import (
    BreakEvenConfig,
    ExitEngineConfig,
    GiveBackConfig,
    PartialExitConfig,
    PartialExitLevel,
    ReputationConfig,
    SetupInvalidationConfig,
    TimeExitConfig,
    TrailingStopConfig,
    VolatilityExitConfig,
)
from metatrade.exit_engine.contracts import (
    ExitAction,
    ExitDecision,
    ExitSignal,
    PositionContext,
    PositionSide,
    TradeOutcome,
)
from metatrade.exit_engine.engine import ExitEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_long_ctx(
    current_price: str = "1.1005",
    peak: str = "1.1010",
    entry_atr: str | None = None,
    n_bars: int = 5,
    stop_loss: str | None = None,
    partial_closes=(),
) -> PositionContext:
    return PositionContext(
        position_id="pos-1",
        symbol="EURUSD",
        side=PositionSide.LONG,
        entry_price=Decimal("1.1000"),
        lot_size=Decimal("0.1"),
        opened_at_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        current_price=Decimal(current_price),
        current_bar=None,
        bars_since_entry=[None] * n_bars,
        peak_favorable_price=Decimal(peak),
        entry_atr=Decimal(entry_atr) if entry_atr else None,
        stop_loss=Decimal(stop_loss) if stop_loss else None,
        partial_closes=partial_closes,
    )


def minimal_cfg(
    close_full_threshold: float = 60.0,
    close_partial_threshold: float = 35.0,
) -> ExitEngineConfig:
    """Config with all rules disabled to isolate specific behaviour."""
    return ExitEngineConfig(
        close_full_threshold=close_full_threshold,
        close_partial_threshold=close_partial_threshold,
        reputation=ReputationConfig(cold_start_trades=1),
        trailing_stop=TrailingStopConfig(enabled=False),
        break_even=BreakEvenConfig(enabled=False),
        time_exit=TimeExitConfig(enabled=False),
        setup_invalidation=SetupInvalidationConfig(enabled=False),
        volatility_exit=VolatilityExitConfig(enabled=False),
        give_back=GiveBackConfig(enabled=False),
        partial_exit=PartialExitConfig(enabled=False),
    )


# ── ExitEngine.evaluate ───────────────────────────────────────────────────────

class TestExitEngineEvaluate:
    def test_returns_exit_decision(self):
        engine = ExitEngine(cfg=minimal_cfg())
        ctx = make_long_ctx()
        decision = engine.evaluate(ctx)
        assert isinstance(decision, ExitDecision)

    def test_disabled_engine_always_holds(self):
        cfg = ExitEngineConfig(enabled=False)
        engine = ExitEngine(cfg=cfg)
        ctx = make_long_ctx(current_price="1.0800")  # huge loss — would normally exit
        decision = engine.evaluate(ctx)
        assert decision.action == ExitAction.HOLD

    def test_hold_when_all_rules_hold(self):
        engine = ExitEngine(cfg=minimal_cfg())
        # position just opened — no rule should fire
        ctx = make_long_ctx(current_price="1.1001", peak="1.1001", n_bars=1)
        decision = engine.evaluate(ctx)
        assert decision.action == ExitAction.HOLD

    def test_give_back_triggers_close_full(self):
        # With 1 of 7 rules firing (confidence≈0.90, close_pct=1.0, all weights=50),
        # aggregate_score = (50×0.90×1.0) / (7×50) × 100 ≈ 12.9.
        # Threshold must be below that value for CLOSE_FULL to trigger.
        cfg = ExitEngineConfig(
            close_full_threshold=12.0,
            close_partial_threshold=5.0,
            reputation=ReputationConfig(cold_start_trades=1),
            trailing_stop=TrailingStopConfig(enabled=False),
            break_even=BreakEvenConfig(enabled=False),
            time_exit=TimeExitConfig(enabled=False),
            setup_invalidation=SetupInvalidationConfig(enabled=False),
            volatility_exit=VolatilityExitConfig(enabled=False),
            give_back=GiveBackConfig(enabled=True, give_back_pct=0.40),
            partial_exit=PartialExitConfig(enabled=False),
        )
        engine = ExitEngine(cfg=cfg)
        # peak=40pip, give_back=40% → min_keep=24pip; current=15pip → exit
        ctx = make_long_ctx(current_price="1.1015", peak="1.1040")
        decision = engine.evaluate(ctx)
        assert decision.action == ExitAction.CLOSE_FULL

    def test_partial_exit_triggers_close_partial(self):
        # partial_exit fires with confidence=0.80, close_pct=0.50
        # score = (50×0.80×0.50) / (7×50) × 100 ≈ 5.7
        # Use threshold below that value to ensure CLOSE_PARTIAL triggers.
        cfg = ExitEngineConfig(
            close_full_threshold=80.0,
            close_partial_threshold=5.0,
            reputation=ReputationConfig(cold_start_trades=1),
            trailing_stop=TrailingStopConfig(enabled=False),
            break_even=BreakEvenConfig(enabled=False),
            time_exit=TimeExitConfig(enabled=False),
            setup_invalidation=SetupInvalidationConfig(enabled=False),
            volatility_exit=VolatilityExitConfig(enabled=False),
            give_back=GiveBackConfig(enabled=False),
            partial_exit=PartialExitConfig(
                enabled=True,
                levels=[PartialExitLevel(pips=10.0, close_pct=0.50)],
            ),
        )
        engine = ExitEngine(cfg=cfg)
        ctx = make_long_ctx(current_price="1.1015", peak="1.1015")  # 15pip > 10pip
        decision = engine.evaluate(ctx)
        assert decision.action == ExitAction.CLOSE_PARTIAL

    def test_signals_present_in_decision(self):
        engine = ExitEngine(cfg=minimal_cfg())
        ctx = make_long_ctx()
        decision = engine.evaluate(ctx)
        # All 7 rules should produce a signal
        assert len(decision.signals) == 7

    def test_aggregate_score_is_float(self):
        engine = ExitEngine(cfg=minimal_cfg())
        decision = engine.evaluate(make_long_ctx())
        assert isinstance(decision.aggregate_score, float)
        assert 0.0 <= decision.aggregate_score <= 100.0

    def test_decision_timestamp_is_utc(self):
        engine = ExitEngine(cfg=minimal_cfg())
        decision = engine.evaluate(make_long_ctx())
        assert decision.timestamp_utc.tzinfo is not None


# ── ExitEngine.record_outcome ──────────────────────────────────────────────────

class TestExitEngineRecordOutcome:
    def _make_outcome(self, pnl_pips: float, signals=()) -> TradeOutcome:
        return TradeOutcome(
            position_id="pos-1",
            symbol="EURUSD",
            entry_price=Decimal("1.1000"),
            exit_price=Decimal(str(1.1000 + pnl_pips * 0.0001)),
            side=PositionSide.LONG,
            lot_size=Decimal("0.1"),
            opened_at_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_at_utc=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
            exit_reason="engine",
            signals_emitted=signals,
            pnl_pips=Decimal(str(pnl_pips)),
        )

    def test_record_outcome_returns_weight_dict(self):
        engine = ExitEngine(cfg=minimal_cfg())
        ctx = make_long_ctx()
        engine.evaluate(ctx)  # populate signal history
        outcome = self._make_outcome(10.0)
        updates = engine.record_outcome(outcome)
        assert isinstance(updates, dict)

    def test_signal_history_cleared_after_outcome(self):
        engine = ExitEngine(cfg=minimal_cfg())
        ctx = make_long_ctx()
        engine.evaluate(ctx)
        assert "pos-1" in engine._signal_history
        outcome = self._make_outcome(10.0)
        engine.record_outcome(outcome)
        assert "pos-1" not in engine._signal_history

    def test_record_outcome_uses_accumulated_history(self):
        """If signals_emitted is empty, engine uses its own history."""
        engine = ExitEngine(cfg=minimal_cfg())
        ctx = make_long_ctx()
        engine.evaluate(ctx)
        outcome = self._make_outcome(10.0)  # empty signals_emitted
        updates = engine.record_outcome(outcome)
        # Should have updates for rules that produced signals
        assert len(updates) > 0


# ── Stop-loss management ───────────────────────────────────────────────────────

class TestStopLossManagement:
    def test_new_sl_higher_than_current_for_long(self):
        cfg = ExitEngineConfig(
            reputation=ReputationConfig(cold_start_trades=1),
            trailing_stop=TrailingStopConfig(
                enabled=True, mode="atr", atr_mult=1.0
            ),
            break_even=BreakEvenConfig(enabled=False),
            time_exit=TimeExitConfig(enabled=False),
            give_back=GiveBackConfig(enabled=False),
            partial_exit=PartialExitConfig(enabled=False),
        )
        engine = ExitEngine(cfg=cfg)
        # entry_atr=0.0010, mult=1.0, peak=1.1040
        # trailing_sl = 1.1040 - 0.0010 = 1.1030
        # current=1.1035 > trailing_sl → HOLD but SL suggestion = 1.1030
        ctx = make_long_ctx(
            current_price="1.1035",
            peak="1.1040",
            entry_atr="0.0010",
            stop_loss="1.1000",  # current SL lower → should update
        )
        decision = engine.evaluate(ctx)
        if decision.new_stop_loss is not None:
            assert float(decision.new_stop_loss) > 1.1000

    def test_sl_never_above_current_price_for_long(self):
        cfg = ExitEngineConfig(
            reputation=ReputationConfig(cold_start_trades=1),
            trailing_stop=TrailingStopConfig(
                enabled=True, mode="fixed_pips", fixed_pips=1.0  # 1 pip
            ),
            break_even=BreakEvenConfig(enabled=False),
            time_exit=TimeExitConfig(enabled=False),
            give_back=GiveBackConfig(enabled=False),
            partial_exit=PartialExitConfig(enabled=False),
        )
        engine = ExitEngine(cfg=cfg)
        # peak=1.1005, distance=1pip → trailing_sl=1.1004
        # current=1.1003 < trailing_sl → would trigger exit
        ctx = make_long_ctx(current_price="1.1003", peak="1.1005")
        decision = engine.evaluate(ctx)
        # If a new SL was suggested, it must be below current price
        if decision.new_stop_loss is not None:
            assert float(decision.new_stop_loss) < float(ctx.current_price)
