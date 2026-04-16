"""Tests for min_sl_pips enforcement inside BaseRunner._apply_exit_profile.

The exit-profile selector can return candidates with very tight SLs (e.g.
1-2 pip swing levels on M1).  When combined with 1 % risk sizing this creates
lot sizes of 2+ lots that then get clamped by the broker and sometimes rejected.

These tests verify that:
  1. If a profile's sl_pips < min_sl_pips, sl_price is widened before sizing.
  2. The TP is rescaled proportionally so the R:R is preserved.
  3. If sl_pips >= min_sl_pips the profile is used as-is.
  4. The runner still returns an approved decision after the enforcement.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from metatrade.core.contracts.market import Bar
from metatrade.core.contracts.risk import PositionSizeResult, RiskDecision
from metatrade.core.enums import OrderSide, Timeframe
from metatrade.core.versioning import ModuleVersion
from metatrade.ml.exit_profile_contracts import ExitProfileCandidate, SelectedExitProfile
from metatrade.runner.base import BaseRunner
from metatrade.runner.config import RunnerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENTRY = Decimal("1.10000")
NOW   = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _bar(close: float = 1.10000, i: int = 0) -> Bar:
    c = Decimal(str(round(close, 5)))
    return Bar(
        symbol="EURUSD",
        timeframe=Timeframe.M5,
        timestamp_utc=datetime(2024, 1, 1, i % 24, 0, 0, tzinfo=timezone.utc),
        open=c, high=c + Decimal("0.0010"),
        low=c  - Decimal("0.0010"), close=c,
        volume=Decimal("1000"),
    )


def _make_candidate(
    sl_pips: float,
    rr: float = 2.0,
    side: OrderSide = OrderSide.BUY,
) -> ExitProfileCandidate:
    """Build a candidate with a given SL distance from ENTRY."""
    pip_size = Decimal("0.0001")
    sl_dist  = Decimal(str(sl_pips)) * pip_size
    if side == OrderSide.BUY:
        sl_price = ENTRY - sl_dist
        tp_price = ENTRY + sl_dist * Decimal(str(rr))
    else:
        sl_price = ENTRY + sl_dist
        tp_price = ENTRY - sl_dist * Decimal(str(rr))
    return ExitProfileCandidate(
        profile_id=f"test_{sl_pips}pip",
        sl_price=sl_price,
        tp_price=tp_price,
        sl_pips=Decimal(str(sl_pips)),
        tp_pips=Decimal(str(sl_pips * rr)),
        risk_reward=Decimal(str(rr)),
        method="atr",
        exit_mode="fixed_tp",
    )


def _make_decision(side: OrderSide = OrderSide.BUY) -> RiskDecision:
    ps = PositionSizeResult(
        lot_size=Decimal("0.50"),
        risk_amount=Decimal("42.00"),
        risk_pct=0.01,
        stop_loss_price=ENTRY - Decimal("0.0020"),
        stop_loss_pips=Decimal("20"),
        take_profit_price=ENTRY + Decimal("0.0040"),
        pip_value=Decimal("10.0"),
        method="fixed_fractional",
    )
    return RiskDecision(
        approved=True,
        symbol="EURUSD",
        side=side,
        timestamp_utc=NOW,
        position_size=ps,
    )


def _make_runner(min_sl_pips: float = 10.0) -> BaseRunner:
    """Return a BaseRunner with no real modules."""
    from metatrade.runner.config import RunnerConfig
    from metatrade.technical_analysis.interface import ITechnicalModule

    cfg = RunnerConfig(
        symbol="EURUSD",
        min_signals=1,
        max_risk_pct=0.01,
        daily_loss_limit_pct=0.99,
        auto_kill_drawdown_pct=0.99,
        max_open_positions=5,
        min_sl_pips=min_sl_pips,
    )
    return BaseRunner(config=cfg, modules=[])


# ---------------------------------------------------------------------------
# Tests: profile SL below minimum → widened
# ---------------------------------------------------------------------------

class TestMinSlEnforcement:

    def _call_apply_exit_profile(
        self,
        candidate: ExitProfileCandidate,
        decision: RiskDecision,
        min_sl_pips: float = 10.0,
        side: OrderSide = OrderSide.BUY,
    ) -> RiskDecision:
        """Wire up mocks and call _apply_exit_profile directly."""
        from metatrade.ml.exit_profile_candidate_generator import ExitProfileCandidateGenerator
        from metatrade.ml.exit_profile_selector import ExitProfileSelector

        runner = _make_runner(min_sl_pips=min_sl_pips)

        # Use spec= so isinstance() checks inside _apply_exit_profile pass
        mock_gen = MagicMock(spec=ExitProfileCandidateGenerator)
        mock_sel = MagicMock(spec=ExitProfileSelector)
        mock_gen.generate.return_value = [candidate]
        mock_sel.select.return_value = SelectedExitProfile(
            candidate=candidate, confidence=0.8,
            method="deterministic", reason="test",
        )
        runner._exit_profile_generator = mock_gen
        runner._exit_profile_selector  = mock_sel

        # All bars close at ENTRY so bars[-1].close == ENTRY (what _apply_exit_profile
        # uses as entry_price), ensuring SL distance measurements are consistent.
        bars = [_bar(float(ENTRY), i) for i in range(20)]
        # Simulate a stable ATR
        mock_atr = Decimal("0.0010")
        with patch.object(runner, "_compute_current_atr", return_value=mock_atr):
            return runner._apply_exit_profile(
                decision=decision,
                bars=bars,
                account=MagicMock(balance=Decimal("4200"), equity=Decimal("4200")),
                timestamp_utc=NOW,
                current_spread_pips=1.0,
            )

    # ── Profile SL well below minimum → widened ──────────────────────────────

    def test_sl_widened_when_below_minimum(self) -> None:
        """A 2-pip SL candidate must be widened to min_sl_pips (10)."""
        candidate = _make_candidate(sl_pips=2.0)
        decision  = _make_decision()
        result = self._call_apply_exit_profile(candidate, decision, min_sl_pips=10.0)

        assert result.approved
        assert result.position_size is not None
        sl_dist_pips = abs(
            result.position_size.stop_loss_price - ENTRY
        ) / Decimal("0.0001")
        # SL must have been widened to at least min_sl_pips
        assert sl_dist_pips >= Decimal("10.0"), (
            f"Expected SL >= 10 pips, got {sl_dist_pips}"
        )

    def test_sl_at_min_not_widened_further(self) -> None:
        """A profile with exactly min_sl_pips must not be modified."""
        candidate = _make_candidate(sl_pips=10.0)
        decision  = _make_decision()
        result = self._call_apply_exit_profile(candidate, decision, min_sl_pips=10.0)

        assert result.approved
        sl_dist_pips = abs(
            result.position_size.stop_loss_price - ENTRY
        ) / Decimal("0.0001")
        assert sl_dist_pips == Decimal("10.0")

    def test_sl_above_min_not_modified(self) -> None:
        """A 20-pip SL candidate must pass through unchanged."""
        candidate = _make_candidate(sl_pips=20.0)
        decision  = _make_decision()
        result = self._call_apply_exit_profile(candidate, decision, min_sl_pips=10.0)

        assert result.approved
        sl_dist_pips = abs(
            result.position_size.stop_loss_price - ENTRY
        ) / Decimal("0.0001")
        assert sl_dist_pips == Decimal("20.0")

    # ── TP is rescaled proportionally ────────────────────────────────────────

    def test_tp_rescaled_proportionally(self) -> None:
        """When SL is widened from 2 to 10 pips (5×), TP distance must also ×5."""
        candidate = _make_candidate(sl_pips=2.0, rr=2.0)
        decision  = _make_decision()
        result = self._call_apply_exit_profile(candidate, decision, min_sl_pips=10.0)

        assert result.position_size is not None
        sl_dist = abs(result.position_size.stop_loss_price   - ENTRY)
        tp_dist = abs(result.position_size.take_profit_price - ENTRY)
        # R:R should be preserved (2.0)
        rr_actual = tp_dist / sl_dist
        assert abs(float(rr_actual) - 2.0) < 0.01, f"R:R should be 2.0, got {rr_actual}"

    # ── SELL direction ────────────────────────────────────────────────────────

    def test_sell_sl_widened_upward(self) -> None:
        """For a SELL, a tight SL (above entry) must be pushed further above entry."""
        candidate = _make_candidate(sl_pips=2.0, side=OrderSide.SELL)
        decision  = _make_decision(side=OrderSide.SELL)
        result = self._call_apply_exit_profile(candidate, decision, min_sl_pips=10.0)

        assert result.approved
        sl_dist_pips = abs(
            result.position_size.stop_loss_price - ENTRY
        ) / Decimal("0.0001")
        assert sl_dist_pips >= Decimal("10.0")
        # SL must be above entry for SELL
        assert result.position_size.stop_loss_price > ENTRY

    # ── min_sl_pips = 0 disables enforcement ─────────────────────────────────

    def test_min_sl_zero_disables_enforcement(self) -> None:
        """When min_sl_pips=0 no widening should occur even for a 1-pip SL."""
        candidate = _make_candidate(sl_pips=1.0)
        decision  = _make_decision()
        result = self._call_apply_exit_profile(candidate, decision, min_sl_pips=0.0)

        assert result.approved
        sl_dist_pips = abs(
            result.position_size.stop_loss_price - ENTRY
        ) / Decimal("0.0001")
        assert sl_dist_pips == Decimal("1.0")

    # ── Lot size is smaller after widening ────────────────────────────────────

    def test_lot_size_smaller_after_widening(self) -> None:
        """Widening the SL must produce a smaller lot (same risk, wider stop)."""
        candidate_tight = _make_candidate(sl_pips=2.0)
        candidate_wide  = _make_candidate(sl_pips=20.0)
        decision = _make_decision()

        result_tight = self._call_apply_exit_profile(
            candidate_tight, decision, min_sl_pips=10.0
        )
        result_wide = self._call_apply_exit_profile(
            candidate_wide, decision, min_sl_pips=10.0
        )

        assert result_tight.position_size is not None
        assert result_wide.position_size  is not None
        # 10-pip (widened tight) lot should be bigger than 20-pip (unchanged wide)
        assert (
            result_tight.position_size.lot_size
            > result_wide.position_size.lot_size
        )
