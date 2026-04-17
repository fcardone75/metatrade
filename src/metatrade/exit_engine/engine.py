"""ExitEngine — weighted-vote orchestrator for all exit rules.

Architecture (see CLAUDE.md for full rationale):

  1. Each bar, evaluate all enabled rules against the current PositionContext.
  2. Collect ExitSignals and suggested stop-loss updates.
  3. Compute a weighted aggregate score:
       score = Σ(w_i × vote_i) / Σ(w_i) × 100
     where vote_i = confidence × close_pct for each signal.
     Rules that vote HOLD contribute 0.
  4. Map score to ExitDecision:
       score >= close_full_threshold    → CLOSE_FULL
       score >= close_partial_threshold → CLOSE_PARTIAL
       otherwise                        → HOLD
  5. For CLOSE_PARTIAL, use the highest close_pct requested by any
     partial-exit rule that fired.
  6. New SL: pick the most favourable suggested stop-loss across all rules
     (highest for LONG, lowest for SHORT).
  7. After a position closes, call record_outcome() to update reputations.

The engine is stateless between calls; all context is passed via PositionContext.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from metatrade.core.log import get_logger
from metatrade.exit_engine.config import ExitEngineConfig
from metatrade.exit_engine.contracts import (
    ExitAction,
    ExitDecision,
    ExitSignal,
    PositionContext,
    PositionSide,
    TradeOutcome,
)
from metatrade.exit_engine.reputation import IReputationStore, ReputationModel
from metatrade.exit_engine.rules.base import IExitRule
from metatrade.exit_engine.rules.break_even import BreakEvenRule
from metatrade.exit_engine.rules.give_back import GiveBackRule
from metatrade.exit_engine.rules.partial_exit import PartialExitRule
from metatrade.exit_engine.rules.setup_invalidation import SetupInvalidationRule
from metatrade.exit_engine.rules.time_exit import TimeExitRule
from metatrade.exit_engine.rules.trailing_stop import TrailingStopRule
from metatrade.exit_engine.rules.volatility_exit import VolatilityExitRule

log = get_logger(__name__)


class ExitEngine:
    """Orchestrates all exit rules and produces a weighted ExitDecision.

    Args:
        cfg:   ExitEngineConfig (all defaults if None).
        store: Optional persistence backend for rule reputations.
        extra_rules: Additional IExitRule instances to include beyond the
                     built-in 7.  Useful for custom rules in backtests.
    """

    def __init__(
        self,
        cfg: ExitEngineConfig | None = None,
        store: IReputationStore | None = None,
        extra_rules: list[IExitRule] | None = None,
    ) -> None:
        self._cfg = cfg or ExitEngineConfig()
        self._reputation = ReputationModel(cfg=self._cfg.reputation, store=store)

        self._rules: list[IExitRule] = [
            TrailingStopRule(cfg=self._cfg.trailing_stop),
            BreakEvenRule(cfg=self._cfg.break_even),
            TimeExitRule(cfg=self._cfg.time_exit),
            SetupInvalidationRule(cfg=self._cfg.setup_invalidation),
            VolatilityExitRule(cfg=self._cfg.volatility_exit),
            GiveBackRule(cfg=self._cfg.give_back),
            PartialExitRule(cfg=self._cfg.partial_exit),
        ]
        if extra_rules:
            self._rules.extend(extra_rules)

        # Signals accumulated during the lifetime of each open position.
        # Keyed by position_id → list of (ExitSignal, price_at_signal).
        self._signal_history: dict[str, list[tuple[ExitSignal, Decimal]]] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(self, ctx: PositionContext) -> ExitDecision:
        """Run all rules and return an aggregated ExitDecision.

        This is the hot path — called every bar for each open position.

        Returns an ExitDecision with action HOLD, CLOSE_PARTIAL, or CLOSE_FULL.
        """
        if not self._cfg.enabled:
            return self._hold_decision(ctx, signals=(), reason="ExitEngine disabled")

        signals: list[ExitSignal] = []
        suggested_sls: list[float] = []

        for rule in self._rules:
            try:
                signal = rule.evaluate(ctx)
                signals.append(signal)
                sl = rule.suggested_sl(ctx)
                if sl is not None:
                    suggested_sls.append(sl)
            except Exception as exc:
                log.warning(
                    "exit_rule_error",
                    rule_id=rule.rule_id,
                    position_id=ctx.position_id,
                    error=str(exc),
                )

        # Accumulate signal history for reputation update later
        history = self._signal_history.setdefault(ctx.position_id, [])
        for signal in signals:
            history.append((signal, ctx.current_price))

        # Weighted vote aggregation
        score = self._aggregate_score(signals, ctx.symbol)

        # Determine action
        action, close_pct = self._map_score_to_action(score, signals)

        # Best SL update
        new_sl = self._best_sl(ctx, suggested_sls)

        # Build explanation
        fired = [s for s in signals if s.action != ExitAction.HOLD]
        explanation = self._build_explanation(score, action, fired, new_sl)

        decision = ExitDecision(
            action=action,
            close_pct=close_pct,
            aggregate_score=round(score, 2),
            signals=tuple(signals),
            explanation=explanation,
            timestamp_utc=datetime.now(UTC),
            new_stop_loss=new_sl,
        )

        log.debug(
            "exit_engine_decision",
            position_id=ctx.position_id,
            symbol=ctx.symbol,
            action=action.value,
            score=round(score, 2),
            rules_fired=len(fired),
        )

        return decision

    def record_outcome(self, outcome: TradeOutcome) -> dict[str, float]:
        """Update rule reputations after a position closes.

        Pass the completed TradeOutcome; the engine fills in signals_emitted
        from the accumulated history if the caller passes an empty tuple.

        Returns {rule_id: new_effective_weight} for logging.
        """
        # If the caller didn't populate signals_emitted, use our history
        if not outcome.signals_emitted:
            history = self._signal_history.get(outcome.position_id, [])
            outcome = TradeOutcome(
                position_id=outcome.position_id,
                symbol=outcome.symbol,
                entry_price=outcome.entry_price,
                exit_price=outcome.exit_price,
                side=outcome.side,
                lot_size=outcome.lot_size,
                opened_at_utc=outcome.opened_at_utc,
                closed_at_utc=outcome.closed_at_utc,
                exit_reason=outcome.exit_reason,
                signals_emitted=tuple(history),
                pnl_pips=outcome.pnl_pips,
                pip_value_per_lot=outcome.pip_value_per_lot,
            )

        updates = self._reputation.update(outcome)

        # Clean up history for this position
        self._signal_history.pop(outcome.position_id, None)

        log.info(
            "exit_engine_reputation_updated",
            position_id=outcome.position_id,
            pnl_pips=float(outcome.pnl_pips),
            rule_updates=updates,
        )
        return updates

    def apply_decay(self, now: datetime | None = None) -> None:
        """Decay stale rule weights.  Call once per session start or daily."""
        self._reputation.apply_decay(now)

    def get_weight(self, rule_id: str, symbol: str = "*") -> float:
        """Return the current effective weight for a rule."""
        return self._reputation.get_weight(rule_id, symbol)

    def all_reputation_states(self) -> list[Any]:
        """Return all RuleReputationState objects (for dashboard/logging)."""
        return self._reputation.all_states()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _aggregate_score(self, signals: list[ExitSignal], symbol: str) -> float:
        """Weighted vote: score = Σ(w_i × vote_i) / Σ(w_i) × 100."""
        total_weight = 0.0
        weighted_vote = 0.0

        for signal in signals:
            w = self._reputation.get_weight(signal.rule_id, symbol)
            total_weight += w
            if signal.action != ExitAction.HOLD:
                # vote_i ∈ [0, 1]: confidence × close_pct
                vote = signal.confidence * signal.close_pct
                weighted_vote += w * vote

        if total_weight == 0.0:
            return 0.0
        return (weighted_vote / total_weight) * 100.0

    def _map_score_to_action(
        self,
        score: float,
        signals: list[ExitSignal],
    ) -> tuple[ExitAction, float]:
        """Map aggregate score to (action, close_pct)."""
        if score >= self._cfg.close_full_threshold:
            return ExitAction.CLOSE_FULL, 1.0

        if score >= self._cfg.close_partial_threshold:
            # Use the highest close_pct from any partial signal
            partial_pcts = [
                s.close_pct
                for s in signals
                if s.action == ExitAction.CLOSE_PARTIAL
            ]
            close_pct = max(partial_pcts) if partial_pcts else 0.33
            return ExitAction.CLOSE_PARTIAL, close_pct

        return ExitAction.HOLD, 0.0

    def _best_sl(
        self,
        ctx: PositionContext,
        suggested_sls: list[float],
    ) -> Decimal | None:
        """Return the most favourable new stop-loss, or None if unchanged."""
        if not suggested_sls:
            return None

        current_sl = float(ctx.stop_loss) if ctx.stop_loss is not None else None

        if ctx.side == PositionSide.LONG:
            # LONG: SL moves up — we want the highest SL that's below current price
            best = max(suggested_sls)
            if current_sl is None or best > current_sl:
                # Never move SL above current price
                if best < float(ctx.current_price):
                    return Decimal(str(round(best, 5)))
        else:
            # SHORT: SL moves down — we want the lowest SL that's above current price
            best = min(suggested_sls)
            if current_sl is None or best < current_sl:
                if best > float(ctx.current_price):
                    return Decimal(str(round(best, 5)))

        return None

    def _hold_decision(
        self,
        ctx: PositionContext,
        signals: tuple[ExitSignal, ...],
        reason: str,
    ) -> ExitDecision:
        return ExitDecision(
            action=ExitAction.HOLD,
            close_pct=0.0,
            aggregate_score=0.0,
            signals=signals,
            explanation=reason,
            timestamp_utc=datetime.now(UTC),
            new_stop_loss=None,
        )

    @staticmethod
    def _build_explanation(
        score: float,
        action: ExitAction,
        fired: list[ExitSignal],
        new_sl: Decimal | None,
    ) -> str:
        parts = [f"score={score:.1f} -> {action.value}"]
        if fired:
            rule_names = ", ".join(s.rule_id for s in fired)
            parts.append(f"rules fired: [{rule_names}]")
        if new_sl is not None:
            parts.append(f"new_sl={new_sl}")
        return " | ".join(parts)
