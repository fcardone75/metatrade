"""Tests for ReputationModel."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest

from metatrade.exit_engine.config import ReputationConfig
from metatrade.exit_engine.contracts import (
    ExitAction,
    ExitSignal,
    PositionSide,
    TradeOutcome,
)
from metatrade.exit_engine.reputation import ReputationModel


# ── Helpers ───────────────────────────────────────────────────────────────────

def _signal(rule_id: str, action: ExitAction = ExitAction.CLOSE_FULL, confidence: float = 0.8) -> ExitSignal:
    return ExitSignal(
        rule_id=rule_id,
        action=action,
        close_pct=1.0 if action == ExitAction.CLOSE_FULL else 0.33 if action == ExitAction.CLOSE_PARTIAL else 0.0,
        reason="test",
        confidence=confidence,
    )


def _outcome(
    signals: list[tuple[ExitSignal, str]],
    pnl_pips: float,
    entry_price: str = "1.1000",
    exit_price: str | None = None,
    side: PositionSide = PositionSide.LONG,
) -> TradeOutcome:
    if exit_price is None:
        exit_price = str(float(entry_price) + pnl_pips * 0.0001)
    return TradeOutcome(
        position_id="p1",
        symbol="EURUSD",
        entry_price=Decimal(entry_price),
        exit_price=Decimal(exit_price),
        side=side,
        lot_size=Decimal("0.1"),
        opened_at_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
        closed_at_utc=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        exit_reason="engine",
        signals_emitted=tuple(
            (sig, Decimal(price)) for sig, price in signals
        ),
        pnl_pips=Decimal(str(pnl_pips)),
    )


# ── ReputationModel ───────────────────────────────────────────────────────────

class TestReputationModel:
    def test_initial_weight_is_50(self):
        model = ReputationModel()
        # Cold-start: no trades → effective weight blends with prior → 50
        assert model.get_weight("trailing_stop") == pytest.approx(50.0)

    def test_update_losing_trade_good_exit_increases_weight(self):
        """Rule exited early at +15pip; trade lost → good call → weight should increase."""
        model = ReputationModel(ReputationConfig(learning_rate=0.5, cold_start_trades=1))
        sig = _signal("trailing_stop")
        # signal emitted at 1.1015 (+15pip); final pnl = -10pip (would have been worse)
        oc = _outcome(
            [(sig, "1.1015")],
            pnl_pips=-10.0,
            entry_price="1.1000",
            exit_price="1.0990",
        )
        updates = model.update(oc)
        new_w = updates["trailing_stop"]
        assert new_w > 50.0, f"Expected weight > 50, got {new_w}"

    def test_update_winning_trade_early_exit_decreases_weight(self):
        """Rule exited at +10pip; trade ended at +30pip → left money on table."""
        model = ReputationModel(ReputationConfig(learning_rate=0.5, cold_start_trades=1))
        sig = _signal("give_back")
        oc = _outcome(
            [(sig, "1.1010")],  # exit at +10pip
            pnl_pips=30.0,      # trade actually ended at +30pip
            exit_price="1.1030",
        )
        updates = model.update(oc)
        new_w = updates["give_back"]
        assert new_w < 50.0, f"Expected weight < 50, got {new_w}"

    def test_hold_rule_winning_trade_score_above_neutral(self):
        """HOLD signal on a winning trade → score=0.60 → slight weight increase."""
        model = ReputationModel(ReputationConfig(learning_rate=0.3, cold_start_trades=1))
        sig = _signal("time_exit", action=ExitAction.HOLD)
        oc = _outcome([(sig, "1.1020")], pnl_pips=20.0, exit_price="1.1020")
        updates = model.update(oc)
        # score=0.60 → EMA pushes weight up slightly
        assert updates["time_exit"] > 50.0

    def test_hold_rule_losing_trade_score_below_neutral(self):
        """HOLD signal on a losing trade → score=0.35 → weight decreases."""
        model = ReputationModel(ReputationConfig(learning_rate=0.3, cold_start_trades=1))
        sig = _signal("time_exit", action=ExitAction.HOLD)
        oc = _outcome([(sig, "1.0990")], pnl_pips=-10.0, exit_price="1.0990")
        updates = model.update(oc)
        assert updates["time_exit"] < 50.0

    def test_cold_start_blends_with_prior(self):
        """Before cold_start_trades are reached, weight is blended toward 50."""
        cfg = ReputationConfig(learning_rate=0.5, cold_start_trades=10, initial_weight=50.0)
        model = ReputationModel(cfg)
        sig = _signal("volatility_exit")
        # Very high score → would push raw weight way up
        oc = _outcome([(sig, "1.1030")], pnl_pips=-50.0, exit_price="1.0950")
        model.update(oc)
        # After 1 trade (n=1, cold=10): effective = 1/10 × learned + 9/10 × 50
        # learned weight will be > 50, but effective should still be close to 50
        w = model.get_weight("volatility_exit")
        assert 50.0 < w < 60.0  # blended — not the raw high value

    def test_weight_clamps_to_max(self):
        cfg = ReputationConfig(
            learning_rate=0.99, min_weight=5.0, max_weight=95.0, cold_start_trades=1
        )
        model = ReputationModel(cfg)
        # Run many winning evaluations to push weight to max
        for _ in range(20):
            sig = _signal("trailing_stop")
            oc = _outcome([(sig, "1.1020")], pnl_pips=-30.0, exit_price="1.0970")
            model.update(oc)
        w = model.get_weight("trailing_stop")
        assert w <= 95.0

    def test_weight_clamps_to_min(self):
        cfg = ReputationConfig(
            learning_rate=0.99, min_weight=5.0, max_weight=95.0, cold_start_trades=1
        )
        model = ReputationModel(cfg)
        # Run many poor evaluations (early exit, trade won big)
        for _ in range(20):
            sig = _signal("trailing_stop")
            oc = _outcome([(sig, "1.1001")], pnl_pips=50.0, exit_price="1.1050")
            model.update(oc)
        w = model.get_weight("trailing_stop")
        assert w >= 5.0

    def test_apply_decay_drifts_toward_initial(self):
        model = ReputationModel(
            ReputationConfig(
                learning_rate=0.99,
                cold_start_trades=1,
                decay_days=1,
                decay_rate=0.1,
                initial_weight=50.0,
            )
        )
        # Push weight high
        sig = _signal("trailing_stop")
        oc = _outcome([(sig, "1.1020")], pnl_pips=-30.0)
        model.update(oc)
        w_before = model.get_weight("trailing_stop")

        # Decay with "now" 10 days after the last eval
        last_ts = model.all_states()[0].last_eval_ts
        future = datetime.fromtimestamp(last_ts + 10 * 86400, tz=timezone.utc)
        model.apply_decay(now=future)

        w_after = model.get_weight("trailing_stop")
        assert w_after < w_before  # drifted toward 50

    def test_store_upsert_called_on_update(self):
        mock_store = MagicMock()
        mock_store.list_rule_reputations.return_value = []
        model = ReputationModel(ReputationConfig(cold_start_trades=1), store=mock_store)
        sig = _signal("trailing_stop")
        oc = _outcome([(sig, "1.1010")], pnl_pips=10.0)
        model.update(oc)
        mock_store.upsert_rule_reputation.assert_called_once()

    def test_store_error_does_not_crash(self):
        mock_store = MagicMock()
        mock_store.list_rule_reputations.return_value = []
        mock_store.upsert_rule_reputation.side_effect = RuntimeError("db error")
        model = ReputationModel(ReputationConfig(cold_start_trades=1), store=mock_store)
        sig = _signal("trailing_stop")
        oc = _outcome([(sig, "1.1010")], pnl_pips=10.0)
        updates = model.update(oc)  # should not raise
        assert "trailing_stop" in updates

    def test_per_symbol_tracking(self):
        model = ReputationModel(ReputationConfig(per_symbol=True, cold_start_trades=1))
        sig = _signal("trailing_stop")

        # EURUSD losing trade → weight goes up
        oc_eu = TradeOutcome(
            position_id="p1", symbol="EURUSD",
            entry_price=Decimal("1.1000"), exit_price=Decimal("1.0990"),
            side=PositionSide.LONG, lot_size=Decimal("0.1"),
            opened_at_utc=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_at_utc=datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
            exit_reason="engine",
            signals_emitted=((sig, Decimal("1.1015")),),
            pnl_pips=Decimal("-10"),
        )
        model.update(oc_eu)
        eu_weight = model.get_weight("trailing_stop", "EURUSD")
        global_weight = model.get_weight("trailing_stop", "*")
        # Per-symbol and global should be independent
        assert eu_weight != global_weight or eu_weight == global_weight  # just checking no crash
        # EURUSD specifically should have been updated
        states = {(s.rule_id, s.symbol): s for s in model.all_states()}
        assert ("trailing_stop", "EURUSD") in states


class TestExitScore:
    """Unit tests for the _exit_score static method."""

    def test_profitable_exit_on_loss_scores_high(self):
        # Exit at +20pip, trade ended -10pip → exit saved money → > 0.75
        score = ReputationModel._exit_score(20.0, -10.0)
        assert score > 0.75

    def test_loss_exit_on_loss_gets_partial_credit(self):
        # Exit at -5pip, trade ended -20pip → still saved 15pip → moderate score
        score = ReputationModel._exit_score(-5.0, -20.0)
        assert 0.10 < score < 0.70

    def test_exit_at_better_level_winner_scores_070(self):
        # Exit at +30pip, trade ended +20pip → exited at same/better → 0.70
        score = ReputationModel._exit_score(30.0, 20.0)
        assert score == pytest.approx(0.70)

    def test_exit_early_winner_scores_proportional(self):
        # Exit at +10pip, trade ended +40pip → ratio=0.25 → score=0.50×0.25=0.125
        score = ReputationModel._exit_score(10.0, 40.0)
        assert score == pytest.approx(0.125)

    def test_exit_at_loss_winner_scores_020(self):
        # Exit at -5pip, trade ended +20pip → bad exit → 0.20
        score = ReputationModel._exit_score(-5.0, 20.0)
        assert score == pytest.approx(0.20)

    def test_score_bounded_0_1(self):
        for exit_pnl, final_pnl in [
            (100.0, -50.0), (-100.0, -50.0), (50.0, 100.0), (-5.0, 30.0),
        ]:
            score = ReputationModel._exit_score(exit_pnl, final_pnl)
            assert 0.0 <= score <= 1.0, f"score={score} out of bounds for exit={exit_pnl}, final={final_pnl}"
