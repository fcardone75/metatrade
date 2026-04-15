"""IExitRule — abstract interface for all exit rules."""

from __future__ import annotations

from abc import ABC, abstractmethod

from metatrade.exit_engine.contracts import ExitSignal, PositionContext


class IExitRule(ABC):
    """Contract every exit rule must satisfy.

    A rule receives a PositionContext (snapshot of the open position + market)
    and returns an ExitSignal.  Rules are stateless by design: all context
    they need is passed in at evaluation time.

    Rules may also return an updated stop-loss price via ``suggested_sl``.
    The ExitEngine collects these and applies the most favourable one.
    """

    @property
    @abstractmethod
    def rule_id(self) -> str:
        """Stable unique identifier (e.g. "trailing_stop")."""

    @abstractmethod
    def evaluate(self, ctx: PositionContext) -> ExitSignal:
        """Evaluate whether the position should be exited.

        Args:
            ctx: Full position context at the current bar.

        Returns:
            ExitSignal with action HOLD, CLOSE_PARTIAL, or CLOSE_FULL.
        """

    def suggested_sl(self, ctx: PositionContext) -> float | None:
        """Return a new stop-loss price if this rule wants to update it.

        Returns None (default) if the rule does not manage the SL.
        The ExitEngine picks the most favourable suggested SL across all rules.
        """
        return None

    def _hold(self, reason: str, **metadata) -> ExitSignal:
        """Convenience: return a HOLD signal."""
        from metatrade.exit_engine.contracts import ExitAction
        return ExitSignal(
            rule_id=self.rule_id,
            action=ExitAction.HOLD,
            close_pct=0.0,
            reason=reason,
            confidence=0.5,
            metadata=dict(metadata),
        )

    def _exit_full(self, reason: str, confidence: float = 0.8, **metadata) -> ExitSignal:
        from metatrade.exit_engine.contracts import ExitAction
        return ExitSignal(
            rule_id=self.rule_id,
            action=ExitAction.CLOSE_FULL,
            close_pct=1.0,
            reason=reason,
            confidence=confidence,
            metadata=dict(metadata),
        )

    def _exit_partial(self, close_pct: float, reason: str, confidence: float = 0.7, **metadata) -> ExitSignal:
        from metatrade.exit_engine.contracts import ExitAction
        return ExitSignal(
            rule_id=self.rule_id,
            action=ExitAction.CLOSE_PARTIAL,
            close_pct=close_pct,
            reason=reason,
            confidence=confidence,
            metadata=dict(metadata),
        )
