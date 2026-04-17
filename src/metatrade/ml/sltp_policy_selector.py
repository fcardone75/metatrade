"""SL/TP policy selector.

Chooses the best candidate from those produced by SlTpCandidateGenerator.

Two operating modes
-------------------
1. Deterministic fallback (always available, no dependencies):
   Selects the candidate with the highest R:R ratio.  ATR-based candidates
   are preferred over swing-based when R:R is tied.  This mode is
   transparent, reproducible, and sufficient for production.

2. ML model (optional):
   Uses a trained multiclass classifier to predict the optimal policy_id
   given the current entry context.  If the predicted policy is not in the
   candidate list (e.g. it was filtered by validation), the selector falls
   back to deterministic mode automatically.

Invariants
----------
- ``select()`` returns None only when the candidates list is empty.
  In that case the caller should veto the trade.
- When a model is provided and its prediction is invalid or raises,
  the selector silently falls back to deterministic — it never propagates
  exceptions.
- The deterministic path is always the final safety net.
"""

from __future__ import annotations

import structlog

from metatrade.ml.sltp_contracts import SelectedSlTpPolicy, SlTpCandidate, SlTpContext

log = structlog.get_logger(__name__)


class SlTpPolicySelector:
    """Selects the best SL/TP policy from a list of candidates.

    Args:
        model: Optional trained classifier.  Must implement ``predict(X)``
               returning a list of policy_id strings, and optionally
               ``predict_proba(X)`` and ``classes_`` for confidence.
               If None, only deterministic mode is used.
        feature_builder: Optional ``SlTpFeatureBuilder``.  Required when
               ``model`` is not None.  Must implement ``build_vector(ctx)``.
    """

    def __init__(
        self,
        model: object | None = None,
        feature_builder: object | None = None,  # SlTpFeatureBuilder | None
    ) -> None:
        self._model = model
        self._feature_builder = feature_builder

    # ── Public ────────────────────────────────────────────────────────────────

    def select(
        self,
        candidates: list[SlTpCandidate],
        context: SlTpContext,
    ) -> SelectedSlTpPolicy | None:
        """Select the best candidate for the current context.

        Returns None if and only if ``candidates`` is empty.

        Args:
            candidates: Output of SlTpCandidateGenerator.generate().
            context:    Current entry context.

        Returns:
            SelectedSlTpPolicy, or None when no valid candidate exists.
        """
        if not candidates:
            return None

        # Try ML path if a model is configured
        if self._model is not None and self._feature_builder is not None:
            result = self._select_ml(candidates, context)
            if result is not None:
                return result
            log.debug("sltp_ml_fallback_to_deterministic")

        return self._select_deterministic(candidates)

    # ── Deterministic ─────────────────────────────────────────────────────────

    def _select_deterministic(
        self, candidates: list[SlTpCandidate]
    ) -> SelectedSlTpPolicy | None:
        """Select the candidate with the highest R:R.

        The generator already sorts by R:R descending, so candidates[0]
        is always the best.  This method is O(1).
        """
        if not candidates:
            return None
        best = candidates[0]
        return SelectedSlTpPolicy(
            candidate=best,
            confidence=1.0,
            method="deterministic_fallback",
            reason=f"best_rr:{best.risk_reward}:{best.policy_id}",
            alternatives=tuple(candidates[1:]),
        )

    # ── ML ────────────────────────────────────────────────────────────────────

    def _select_ml(
        self,
        candidates: list[SlTpCandidate],
        context: SlTpContext,
    ) -> SelectedSlTpPolicy | None:
        """Use the trained model to predict the best policy.

        Returns None on any exception or when the predicted policy is not
        present in the candidate list.
        """
        try:
            from metatrade.ml.sltp_feature_builder import SlTpFeatureBuilder
            if not isinstance(self._feature_builder, SlTpFeatureBuilder):
                return None

            vec = self._feature_builder.build_vector(context)
            X = [vec]

            predicted_ids = self._model.predict(X)  # type: ignore[union-attr]
            if not predicted_ids:
                return None
            predicted_id: str = str(predicted_ids[0])

            matched = next(
                (c for c in candidates if c.policy_id == predicted_id), None
            )
            if matched is None:
                log.debug(
                    "sltp_ml_predicted_not_in_candidates",
                    predicted=predicted_id,
                    available=[c.policy_id for c in candidates],
                )
                return None

            # Extract confidence from predict_proba if available
            confidence = 0.5
            try:
                proba = self._model.predict_proba(X)  # type: ignore[union-attr]
                classes = list(self._model.classes_)  # type: ignore[union-attr]
                if predicted_id in classes:
                    idx = classes.index(predicted_id)
                    confidence = float(proba[0][idx])
            except Exception:
                pass  # confidence stays 0.5

            return SelectedSlTpPolicy(
                candidate=matched,
                confidence=confidence,
                method="ml_model",
                reason=f"ml_predicted:{predicted_id}",
                alternatives=tuple(c for c in candidates if c.policy_id != predicted_id),
            )
        except Exception as exc:
            log.warning("sltp_ml_select_failed", error=str(exc))
            return None
