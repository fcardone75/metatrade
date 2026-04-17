"""ReputationModel — per-rule weight management with EMA updates.

Design (see CLAUDE.md for full rationale):

  - Each rule has weight w_i ∈ [min_weight, max_weight] (default [5, 95]).
  - Initial weight: initial_weight (default 50 — cold-start neutral).
  - After each trade closes:
      contribution_score ∈ [0, 1] is computed per rule.
      w_new = clip(α × score × 100 + (1−α) × w_old, min, max)
  - Cold-start blend: for the first cold_start_trades trades, weight is blended
      with the initial prior so early noise cannot dominate.
  - Decay: if a rule has not been evaluated for decay_days, weight drifts
      toward the initial value at decay_rate per inactive day.

Contribution score formula (given TradeOutcome):
  For each EXIT signal emitted by rule R at price P_signal:
    theoretical_pnl = pips_at_signal (positive = profit at that moment)
    final_pnl       = outcome.pnl_pips

    If trade ended badly (final_pnl ≤ 0):
      score = sigmoid(theoretical_pnl × scale) — higher if rule exited profitably
    If trade ended well (final_pnl > 0):
      if theoretical_pnl < final_pnl:
        score = 0.5 × (theoretical_pnl / final_pnl)  — penalty for too-early exit
      else:
        score = 0.7  — rule correctly exited before profit reduced

  For rules that only emitted HOLD throughout:
    score = 0.6 if final_pnl > 0 (good to hold)
    score = 0.35 if final_pnl ≤ 0 (should have exited)

Per-symbol support: weights stored as {symbol: {rule_id: state}}.
If symbol not found, falls back to global ("*").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from metatrade.exit_engine.config import ReputationConfig
from metatrade.exit_engine.contracts import ExitAction, TradeOutcome

# ── Storage protocol ──────────────────────────────────────────────────────────

class IReputationStore(Protocol):
    def upsert_rule_reputation(
        self,
        rule_id: str,
        symbol: str,
        weight: float,
        eval_count: int,
        mean_score: float,
        last_eval_ts: int,
    ) -> None: ...

    def list_rule_reputations(self) -> list[dict[str, Any]]: ...


# ── State ─────────────────────────────────────────────────────────────────────

@dataclass
class RuleReputationState:
    rule_id:    str
    symbol:     str    # "*" for global
    weight:     float
    eval_count: int   = 0
    mean_score: float = 0.5
    last_eval_ts: int = 0   # Unix timestamp of last trade close


# ── Model ─────────────────────────────────────────────────────────────────────

class ReputationModel:
    """Manages per-rule (optionally per-symbol) reputation weights.

    Args:
        cfg:   ReputationConfig.
        store: Optional persistence back-end. None = in-memory only.
    """

    def __init__(
        self,
        cfg: ReputationConfig | None = None,
        store: IReputationStore | None = None,
    ) -> None:
        self._cfg = cfg or ReputationConfig()
        self._store = store
        # {symbol -> {rule_id -> state}}
        self._states: dict[str, dict[str, RuleReputationState]] = {}
        if store is not None:
            self._load_from_db()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_weight(self, rule_id: str, symbol: str = "*") -> float:
        """Return the effective weight for a rule, applying cold-start blend.

        Falls back from symbol-specific to global ("*") if needed.
        """
        state = self._get_state(rule_id, symbol if self._cfg.per_symbol else "*")
        return self._effective_weight(state)

    def update(self, outcome: TradeOutcome) -> dict[str, float]:
        """Update weights for all rules that emitted signals during outcome.

        Returns a dict of {rule_id: new_weight} for logging/dashboard.
        """
        symbol = outcome.symbol if self._cfg.per_symbol else "*"
        scores = self._compute_scores(outcome)
        updates: dict[str, float] = {}
        now_ts = int(outcome.closed_at_utc.timestamp())

        for rule_id, score in scores.items():
            state = self._get_state(rule_id, symbol)
            state.eval_count += 1
            state.mean_score = (
                self._cfg.learning_rate * score
                + (1.0 - self._cfg.learning_rate) * state.mean_score
            )
            raw_new = (
                self._cfg.learning_rate * score * 100.0
                + (1.0 - self._cfg.learning_rate) * state.weight
            )
            state.weight = max(self._cfg.min_weight, min(self._cfg.max_weight, raw_new))
            state.last_eval_ts = now_ts
            updates[rule_id] = self._effective_weight(state)

            if self._store is not None:
                try:
                    self._store.upsert_rule_reputation(
                        rule_id=rule_id,
                        symbol=symbol,
                        weight=round(state.weight, 4),
                        eval_count=state.eval_count,
                        mean_score=round(state.mean_score, 4),
                        last_eval_ts=now_ts,
                    )
                except Exception:
                    pass

        return updates

    def apply_decay(self, now: datetime | None = None) -> None:
        """Decay weights toward initial_weight for rules inactive for > decay_days.

        Call once per session start (or daily) to prevent stale dominance.
        """
        if self._cfg.decay_days <= 0:
            return
        now_ts = int((now or datetime.now(UTC)).timestamp())
        decay_threshold_secs = self._cfg.decay_days * 86400

        for sym_states in self._states.values():
            for state in sym_states.values():
                if state.last_eval_ts == 0:
                    continue
                inactive_secs = now_ts - state.last_eval_ts
                if inactive_secs < decay_threshold_secs:
                    continue
                inactive_days = inactive_secs / 86400
                # Drift toward initial_weight
                gap = self._cfg.initial_weight - state.weight
                state.weight += gap * self._cfg.decay_rate * inactive_days
                state.weight = max(self._cfg.min_weight, min(self._cfg.max_weight, state.weight))

    def all_states(self) -> list[RuleReputationState]:
        return [
            state
            for sym_states in self._states.values()
            for state in sym_states.values()
        ]

    # ── Score computation ──────────────────────────────────────────────────────

    def _compute_scores(self, outcome: TradeOutcome) -> dict[str, float]:
        """Map each rule that participated in this trade to a score [0, 1]."""
        final_pnl = float(outcome.pnl_pips)
        rule_exit_signals: dict[str, list[float]] = {}   # rule_id → [pnl_at_signal]

        for signal, price_at_signal in outcome.signals_emitted:
            if signal.action == ExitAction.HOLD:
                continue
            if signal.rule_id not in rule_exit_signals:
                rule_exit_signals[signal.rule_id] = []
            # Theoretical pnl in pips if we'd closed at this price
            if outcome.side.value == "LONG":
                pnl_at = float((price_at_signal - outcome.entry_price) / Decimal("0.0001"))
            else:
                pnl_at = float((outcome.entry_price - price_at_signal) / Decimal("0.0001"))
            rule_exit_signals[signal.rule_id].append(pnl_at)

        # Collect all rule IDs that participated (EXIT or HOLD)
        all_rule_ids = {sig.rule_id for sig, _ in outcome.signals_emitted}
        scores: dict[str, float] = {}

        for rule_id in all_rule_ids:
            if rule_id in rule_exit_signals:
                # Best theoretical exit for this rule
                best_exit_pnl = max(rule_exit_signals[rule_id])
                scores[rule_id] = self._exit_score(best_exit_pnl, final_pnl)
            else:
                # Rule only emitted HOLD
                scores[rule_id] = 0.60 if final_pnl > 0 else 0.35

        return scores

    @staticmethod
    def _exit_score(exit_pnl: float, final_pnl: float) -> float:
        """Score an exit signal given theoretical and final P&L in pips.

        Scale ∈ [0, 1]:
          - Rule exited before trade turned bad: → high score
          - Rule exited too early on a winner:   → moderate penalty
          - Rule exited a losing trade early:    → full credit (saved from worse)
        """
        if final_pnl <= 0:
            # Trade was a loss: any positive exit_pnl is a win; use sigmoid
            if exit_pnl > 0:
                return (1.0 + math.tanh(exit_pnl * 0.05)) / 2.0
            # Exit also at a loss but maybe less bad
            improvement = final_pnl - exit_pnl   # positive if we'd have lost less
            return max(0.10, (1.0 + math.tanh(improvement * 0.05)) / 2.0)
        else:
            # Trade was a winner
            if exit_pnl >= final_pnl:
                # Exited at same or better level — 0.70 (good call, maybe not best)
                return 0.70
            elif exit_pnl > 0:
                # Exited profitably but left money on the table
                ratio = exit_pnl / final_pnl   # 0 < ratio < 1
                return 0.50 * ratio             # penalty for early exit
            else:
                # Exited at a loss even though trade ultimately recovered
                return 0.20

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _effective_weight(self, state: RuleReputationState) -> float:
        """Apply cold-start blend with prior."""
        n = state.eval_count
        cold = self._cfg.cold_start_trades
        if n >= cold:
            return state.weight
        # Blend: n/cold × learned + (1 - n/cold) × prior
        blend = n / cold
        return blend * state.weight + (1.0 - blend) * self._cfg.initial_weight

    def _get_state(self, rule_id: str, symbol: str) -> RuleReputationState:
        if symbol not in self._states:
            self._states[symbol] = {}
        if rule_id not in self._states[symbol]:
            self._states[symbol][rule_id] = RuleReputationState(
                rule_id=rule_id,
                symbol=symbol,
                weight=self._cfg.initial_weight,
            )
        return self._states[symbol][rule_id]

    def _load_from_db(self) -> None:
        if self._store is None:
            return
        try:
            rows = self._store.list_rule_reputations()
        except Exception:
            return
        for row in rows:
            rule_id = row.get("rule_id", "")
            symbol  = row.get("symbol", "*")
            if not rule_id:
                continue
            if symbol not in self._states:
                self._states[symbol] = {}
            self._states[symbol][rule_id] = RuleReputationState(
                rule_id=rule_id,
                symbol=symbol,
                weight=float(row.get("weight", self._cfg.initial_weight)),
                eval_count=int(row.get("eval_count", 0)),
                mean_score=float(row.get("mean_score", 0.5)),
                last_eval_ts=int(row.get("last_eval_ts", 0)),
            )
