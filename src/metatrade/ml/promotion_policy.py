"""Promotion policy — decides when to switch from the current model to a candidate.

Pure logic with no I/O or side effects.  The policy evaluates two independent
trigger conditions:

UPGRADE trigger
    The candidate's holdout accuracy is strictly better than the current
    deployed model's holdout accuracy AND the candidate clears the minimum
    holdout bar.  Applied regardless of live accuracy reliability.

EMERGENCY trigger
    The live accuracy of the current model has fallen below the degradation
    threshold AND the minimum number of evaluated samples has been reached
    (so the metric is statistically reliable).  If a candidate exists, switch
    to it; otherwise just flag the degradation.

Switch is deferred until there are no open positions (enforced by the caller).
"""

from __future__ import annotations

from dataclasses import dataclass

from metatrade.ml.live_tracker import LiveAccuracyStats


@dataclass
class PromotionDecision:
    """Result of a single policy evaluation."""

    should_switch: bool
    reason: str
    trigger: str = "none"          # "upgrade" | "emergency" | "none"
    candidate_version: str | None = None


class PromotionPolicy:
    """Evaluate whether the active model should be replaced by a candidate.

    Args:
        min_holdout:            Minimum holdout accuracy a candidate must have.
        degradation_threshold:  Live accuracy below which an emergency switch
                                is triggered (only when stats are reliable).
    """

    def __init__(
        self,
        *,
        min_holdout: float = 0.52,
        degradation_threshold: float = 0.50,
    ) -> None:
        self._min_holdout = min_holdout
        self._degradation_threshold = degradation_threshold

    def evaluate(
        self,
        *,
        current_holdout: float | None,
        candidate_version: str,
        candidate_holdout: float,
        live_stats: LiveAccuracyStats | None,
    ) -> PromotionDecision:
        """Decide whether to promote the candidate.

        Args:
            current_holdout:    Holdout accuracy recorded when the currently
                                deployed model was trained.  None if unknown.
            candidate_version:  Version string of the candidate model.
            candidate_holdout:  Holdout accuracy of the candidate.
            live_stats:         Current live accuracy stats for the deployed
                                model (None if tracker not available).

        Returns:
            PromotionDecision with should_switch=True/False and the reason.
        """
        # Candidate must clear the minimum holdout bar
        if candidate_holdout < self._min_holdout:
            return PromotionDecision(
                should_switch=False,
                reason=(
                    f"candidate holdout {candidate_holdout:.1%} < "
                    f"minimum {self._min_holdout:.1%}"
                ),
            )

        # ── Emergency trigger: live accuracy degraded ─────────────────────────
        if (
            live_stats is not None
            and live_stats.is_reliable
            and live_stats.accuracy is not None
            and live_stats.accuracy < self._degradation_threshold
        ):
            return PromotionDecision(
                should_switch=True,
                trigger="emergency",
                reason=(
                    f"emergency switch: live accuracy {live_stats.accuracy:.1%} "
                    f"< {self._degradation_threshold:.1%} "
                    f"({live_stats.n_evaluated} samples)"
                ),
                candidate_version=candidate_version,
            )

        # ── Upgrade trigger: candidate is strictly better ─────────────────────
        if current_holdout is None or candidate_holdout > current_holdout:
            return PromotionDecision(
                should_switch=True,
                trigger="upgrade",
                reason=(
                    f"upgrade: candidate holdout {candidate_holdout:.1%} > "
                    f"current {current_holdout:.1%}"
                    if current_holdout is not None
                    else f"upgrade: no current model, candidate holdout {candidate_holdout:.1%}"
                ),
                candidate_version=candidate_version,
            )

        return PromotionDecision(
            should_switch=False,
            reason=(
                f"no trigger: candidate {candidate_holdout:.1%} "
                f"<= current {current_holdout:.1%}"
            ),
        )
